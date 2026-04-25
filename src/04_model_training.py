import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PREPROCESSED, ARTIFACTS_DIR, FEATURE_COLS,
    X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH,
    X_UNSEEN_PATH, Y_UNSEEN_PATH, MODEL_PATH, OOF_PATH,
    RANDOM_STATE, UNSEEN_FRAC, TEST_FRAC, CV_FOLDS,
    ES_VALIDATION_FRAC, EARLY_STOPPING_ROUNDS, XGB_PARAMS,
)

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold
)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# 1. Group-Based Splitting 

def group_based_split(df: pd.DataFrame):
    """
    Split into train / test / unseen with guaranteed mother-level separation.
    """
    X = df[FEATURE_COLS]
    y = df['LBW_Risk']
    g = df['mother_id']

    # Step 1: Hold out unseen (sealed)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=UNSEEN_FRAC,
                              random_state=RANDOM_STATE)
    main_idx, unseen_idx = next(gss1.split(X, y, groups=g))

    X_main, y_main, g_main = X.iloc[main_idx], y.iloc[main_idx], g.iloc[main_idx]
    X_unseen = X.iloc[unseen_idx].reset_index(drop=True)
    y_unseen = y.iloc[unseen_idx].reset_index(drop=True)

    # Step 2: Split main into train + test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                              random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss2.split(X_main, y_main, groups=g_main))

    X_train  = X_main.iloc[tr_idx].reset_index(drop=True)
    y_train  = y_main.iloc[tr_idx].reset_index(drop=True)
    X_test   = X_main.iloc[te_idx].reset_index(drop=True)
    y_test   = y_main.iloc[te_idx].reset_index(drop=True)
    g_train  = g_main.iloc[tr_idx]
    g_test   = g_main.iloc[te_idx]
    g_unseen = g.iloc[unseen_idx]

    # Verify zero mother-level overlap (assertion — will raise on leakage)
    assert len(set(g_train) & set(g_test))   == 0, "LEAKAGE: Train-Test overlap!"
    assert len(set(g_train) & set(g_unseen)) == 0, "LEAKAGE: Train-Unseen overlap!"
    assert len(set(g_test)  & set(g_unseen)) == 0, "LEAKAGE: Test-Unseen overlap!"

    print(f"\n  [SPLIT] Mother-level GroupShuffleSplit — VERIFIED LEAK-FREE")
    print(f"  Training set  : {len(X_train):,} rows | {len(g_train.unique()):,} mothers | LBW: {y_train.mean()*100:.1f}%")
    print(f"  Test set      : {len(X_test):,}  rows | {len(g_test.unique()):,} mothers | LBW: {y_test.mean()*100:.1f}%")
    print(f"  Unseen holdout: {len(X_unseen):,}  rows | {len(g_unseen.unique()):,} mothers | LBW: {y_unseen.mean()*100:.1f}%")

    return X_train, X_test, X_unseen, y_train, y_test, y_unseen, g_train, g_test

# 2. CV with StratifiedGroupKFold (OOF Probabilities) 

def generate_oof_probabilities(X_train: pd.DataFrame, y_train: pd.Series,
                                g_train: pd.Series, xgb_params: dict) -> np.ndarray:
    """
    Generate out-of-fold (OOF) probabilities using StratifiedGroupKFold.
    """
    sgkf     = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                     random_state=RANDOM_STATE)
    oof      = np.zeros(len(X_train))
    fold_aucs = []

    print(f"\n  [CV] {CV_FOLDS}-fold StratifiedGroupKFold — mother-level separation")
    print(f"  [CV] Early stopping: {EARLY_STOPPING_ROUNDS} rounds, eval_metric=aucpr")