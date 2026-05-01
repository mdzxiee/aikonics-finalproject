# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/04_model_training.py
#
# Purpose : Group-based splitting → XGBoost training with StratifiedGroupKFold
#           CV + early stopping → OOF probability generation → model artifact
#           export + 6 CSV splits export.
#
# Input  : artifacts/preprocessed.pkl
# Output : artifacts/model.pkl
#          artifacts/oof_probabilities.pkl
#          artifacts/X_train.csv + y_train.csv
#          artifacts/X_test.csv  + y_test.csv
#          artifacts/X_unseen.csv+ y_unseen.csv
#          (threshold.pkl and features.json are produced by 05_threshold_validation.py)
#
# KEY CORRECTIONS vs. monolithic version:
#   1. GroupShuffleSplit: mother-level separation for all 3 partitions
#   2. StratifiedGroupKFold: mother-level separation within CV folds
#   3. No SimpleImputer in pipeline — XGBoost handles NaN natively
#   4. Early stopping: uses 10% of training data as ES validation set
#   5. eval_metric='aucpr': correct monitoring criterion for imbalanced data
#   6. scale_pos_weight: computed dynamically from actual class ratio
#   7. 6 CSV exports for transparency and reproducibility
#
# Connects to: 05_threshold_validation.py, 02_eda.py, 03_feature_selection.py
# ---------------------------------------------------------------------------

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

    RATIONALE:
    NDHS Kids Recode has multiple birth rows per mother. Row-level splitting
    (train_test_split) allows the same mother in both training and test sets.
    Because each mother shares household, region, wealth, and education across
    her birth records, her test records are not independent of her training
    records — this inflates performance metrics. GroupShuffleSplit prevents
    this by assigning ALL of one mother's records to exactly ONE partition.

    Verified in PHKR82FL.csv: confirmed multiple birth records per mother.
    Overlap assertions (below) will raise if any mother leaks across sets.

    Partition sizes:
      Unseen: 10% (sealed until 06_evaluation.py)
      Test:   20% of remaining 90%
      Train:  ~72% (used for CV, threshold selection, final model fit)
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

    X_train = X_main.iloc[tr_idx].reset_index(drop=True)
    y_train = y_main.iloc[tr_idx].reset_index(drop=True)
    X_test = X_main.iloc[te_idx].reset_index(drop=True)
    y_test = y_main.iloc[te_idx].reset_index(drop=True)
    g_train = g_main.iloc[tr_idx]
    g_test = g_main.iloc[te_idx]
    g_unseen = g.iloc[unseen_idx]

    # Verify zero mother-level overlap (assertion — will raise on leakage)
    assert len(set(g_train) & set(g_test)) == 0, "LEAKAGE: Train-Test overlap!"
    assert len(set(g_train) & set(g_unseen)) == 0, "LEAKAGE: Train-Unseen overlap!"
    assert len(set(g_test) & set(g_unseen)) == 0, "LEAKAGE: Test-Unseen overlap!"

    print(f"\n  [SPLIT] Mother-level GroupShuffleSplit — VERIFIED LEAK-FREE")
    print(
        f"  Training set  : {len(X_train):,} rows | {len(g_train.unique()):,} mothers | LBW: {y_train.mean() * 100:.1f}%")
    print(
        f"  Test set      : {len(X_test):,}  rows | {len(g_test.unique()):,} mothers | LBW: {y_test.mean() * 100:.1f}%")
    print(
        f"  Unseen holdout: {len(X_unseen):,}  rows | {len(g_unseen.unique()):,} mothers | LBW: {y_unseen.mean() * 100:.1f}%")

    return X_train, X_test, X_unseen, y_train, y_test, y_unseen, g_train, g_test


# 2. CV with StratifiedGroupKFold (OOF Probabilities) 

def generate_oof_probabilities(X_train: pd.DataFrame, y_train: pd.Series,
                               g_train: pd.Series, xgb_params: dict) -> np.ndarray:
    """
    Generate out-of-fold (OOF) probabilities using StratifiedGroupKFold.

    WHY StratifiedGroupKFold (not StratifiedKFold):
      StratifiedKFold splits by ROW — the same mother can appear in both
      the CV training fold and the CV validation fold. This inflates
      within-CV performance estimates for the same reason as described in
      group_based_split(). StratifiedGroupKFold guarantees mother-level
      separation within each CV fold.

    WHY cross_val_predict-equivalent (manual OOF loop):
      sklearn's cross_val_predict does not accept custom groups in the same
      call signature as StratifiedGroupKFold. We implement the OOF loop
      manually for full control over early stopping per-fold.

    WHY OOF probabilities (not fold-level predict_classes):
      OOF raw probabilities are the INPUT to threshold selection in
      05_threshold_validation.py. If we apply the default 0.50 threshold
      internally (as cross_validate does), the threshold cannot be selected
      on OOF data — it becomes a test-set decision, which is leakage.

    Early stopping within CV:
      Each fold's validation data serves as the early stopping set.
      This means n_estimators is adaptively chosen per fold. The OOF
      probabilities therefore come from models of varying n_estimators —
      which is the correct behavior (each fold's model is independently tuned).
    """
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train))
    fold_aucs = []
    best_iters = []

    print(f"\n  [CV] {CV_FOLDS}-fold StratifiedGroupKFold — mother-level separation")
    print(f"  [CV] Early stopping: {EARLY_STOPPING_ROUNDS} rounds, eval_metric=aucpr")

    for fold_n, (tr_idx, va_idx) in enumerate(
            sgkf.split(X_train, y_train, groups=g_train), start=1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # Verify no group leakage within this fold
        g_tr_fold = g_train.iloc[tr_idx]
        g_va_fold = g_train.iloc[va_idx]
        assert len(set(g_tr_fold) & set(g_va_fold)) == 0, \
            f"LEAKAGE in fold {fold_n}!"

        fold_model = XGBClassifier(**xgb_params)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )

        va_proba = fold_model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_proba
        fold_auc = roc_auc_score(y_va, va_proba)
        fold_aucs.append(fold_auc)

        best_it = fold_model.best_iteration
        best_iters.append(best_it)
        print(f"    Fold {fold_n:2d}: AUC={fold_auc:.4f} | best_iter={best_it}")

    mean_oof_auc = roc_auc_score(y_train, oof)
    print(f"\n  [CV] OOF ROC-AUC: {mean_oof_auc:.4f}")
    print(f"  [CV] Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  [CV] Avg best n_estimators across folds: {np.mean(best_iters):.0f}")

    return oof


# 3. Final Model Training with Early Stopping 

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series,
                      g_train: pd.Series, xgb_params: dict) -> XGBClassifier:
    """
    Train final XGBClassifier on full training set with early stopping.

    Early stopping validation set: 10% of training data, held from the
    full training set using GroupShuffleSplit to prevent mother-level leakage.
    This is the ONLY data used to determine when to stop.
    Test and unseen sets are NEVER touched for this purpose.

    ARCHITECTURE NOTE: No sklearn Pipeline is used here.
    We removed SimpleImputer (see Error 4.1 verdict). XGBoost handles NaN
    natively via sparsity-aware split finding. The Pipeline is not needed.

    The model object saved is the raw XGBClassifier (not a Pipeline).
    Downstream modules (06_evaluation, 07_shap, prototype/predictor.py)
    use this model directly on raw NaN-containing data.
    """
    # Hold out 10% of training for early stopping via GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=ES_VALIDATION_FRAC,
                            random_state=RANDOM_STATE)
    tr_idx, es_idx = next(gss.split(X_train, y_train, groups=g_train))

    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_es, y_es = X_train.iloc[es_idx], y_train.iloc[es_idx]

    # Optional sanity check for early stopping leakage
    g_tr, g_es = g_train.iloc[tr_idx], g_train.iloc[es_idx]
    assert len(set(g_tr) & set(g_es)) == 0, "LEAKAGE in Early Stopping split!"

    print(f"\n  [FINAL] Early stopping validation: {len(X_es):,} rows from training")
    print(f"  [FINAL] Actual training rows: {len(X_tr):,}")

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_es, y_es)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    best_it = model.best_iteration
    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    es_val_auc = roc_auc_score(y_es, model.predict_proba(X_es)[:, 1])

    print(f"  [FINAL] Best iteration (early stopping): {best_it}")
    print(f"  [FINAL] Train AUC (optimistic upper bound): {train_auc:.4f}")
    print(f"  [FINAL] ES-val AUC: {es_val_auc:.4f}")
    gap = train_auc - es_val_auc
    print(
        f"  [FINAL] Train-ESval gap: {gap:.4f} {'(acceptable)' if gap < 0.10 else '(⚠ high — consider more regularization)'}")
    return model


# 4. Save Outputs 

def save_splits_as_csv(X_train, X_test, X_unseen,
                       y_train, y_test, y_unseen) -> None:
    """
    Export all 6 split CSVs for transparency and interoperability.
    These are consumed by 02_eda.py, 03_feature_selection.py, and
    06_evaluation.py. They also allow instructors to inspect splits
    without running Python.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False, header=True)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False, header=True)
    X_unseen.to_csv(X_UNSEEN_PATH, index=False)
    y_unseen.to_csv(Y_UNSEEN_PATH, index=False, header=True)
    print(f"\n  [SAVED] 6 split CSVs → {ARTIFACTS_DIR}")
    for name, df in [('X_train', X_train), ('X_test', X_test), ('X_unseen', X_unseen)]:
        print(f"          {name}: {df.shape}")


# Main 

def run_model_training() -> None:

    print("STAGE 4: MODEL TRAINING")

    # Load preprocessed data
    pkg = joblib.load(PREPROCESSED)
    df = pkg['df']
    imbalance_ratio = pkg['imbalance_ratio']

    # Dynamic scale_pos_weight (NEVER hardcoded)
    spw = round(imbalance_ratio)
    xgb_params = {**XGB_PARAMS, 'scale_pos_weight': spw}
    print(f"\n  [CONFIG] scale_pos_weight = {spw} (dynamic from ratio={imbalance_ratio:.2f}:1)")
    print(f"  [CONFIG] eval_metric = aucpr (PR-AUC monitoring, not logloss)")
    print(f"  [CONFIG] No SimpleImputer — XGBoost handles NaN natively")

    # Split
    X_train, X_test, X_unseen, y_train, y_test, y_unseen, g_train, g_test = \
        group_based_split(df)

    # Save 6 CSVs immediately after splitting
    save_splits_as_csv(X_train, X_test, X_unseen, y_train, y_test, y_unseen)

    # OOF probabilities via StratifiedGroupKFold CV
    oof_proba = generate_oof_probabilities(
        X_train, y_train, g_train, xgb_params
    )

    # Train final model on full training set
    final_model = train_final_model(X_train, y_train, g_train, xgb_params)

    # Save model artifact
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\n  [SAVED] model.pkl → {MODEL_PATH}")

    # Save OOF probabilities for threshold selection
    joblib.dump({'oof_proba': oof_proba, 'y_train': y_train}, OOF_PATH)
    print(f"  [SAVED] oof_probabilities.pkl → {OOF_PATH}")

    # Save split data for evaluation
    split_path = os.path.join(ARTIFACTS_DIR, "split_data.pkl")
    joblib.dump({
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'X_unseen': X_unseen, 'y_unseen': y_unseen,
    }, split_path)
    print(f"  [SAVED] split_data.pkl → {split_path}")

    print(f"\n  [NEXT] Run 05_threshold_validation.py to select and save threshold.")


if __name__ == "__main__":
    run_model_training()