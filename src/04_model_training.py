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