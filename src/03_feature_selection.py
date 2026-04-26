# PURPOSE: Feature selection DOCUMENTATION AND ANALYSIS module.
#          This file does NOT drop any features from the model.
#          The final feature set is fixed in config.py (FEATURE_COLS)
#          based on domain knowledge and BHW collectibility criteria.
#
# WHY THIS FILE EXISTS (against the "delete it" recommendation):
#   A thesis panel WILL ask: "Why these 13 features and not 20?"
#   Without documented feature analysis, the answer is "because we said so."
#   This file provides the STATISTICAL EVIDENCE that supports the domain-
#   based feature set — permutation importance, multicollinearity (VIF),
#   and cross-referencing with SHAP global rankings.
#
# WHY NO AUTOMATED DROPPING:
#   Automated feature selection (RFECV, VIF pruning, etc.) breaks the
#   chain of reasoning between domain knowledge and model design.
#   BHW collectibility is a hard constraint — if a feature cannot be
#   collected by a BHW during a prenatal home visit, it is excluded
#   regardless of its AUC or VIF. This constraint is applied in config.py,
#   not here.
#
# Input  : artifacts/X_train.csv, artifacts/y_train.csv
#          artifacts/model.pkl  (optional — for permutation importance)
# Output : outputs/feature_vif.csv
#          outputs/feature_permutation_importance.csv
#          outputs/correlation_heatmap.png
#
# Connects to: 07_shap_analysis.py (cross-validate SHAP vs permutation rankings)
# ===========================================================================

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, OUTPUTS_DIR, FEATURE_COLS, MODEL_PATH
)

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import joblib

# ─── Variance Inflation Factor ──────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for each feature to detect multicollinearity.
    VIF > 10: strong multicollinearity — potential redundancy.
    VIF > 5 : moderate multicollinearity — monitor.
    VIF < 5 : acceptable.

    Note: VIF is an advisory metric only. XGBoost is robust to moderate
    multicollinearity because it builds trees on feature subsets
    (colsample_bytree=0.8). We do NOT drop features solely based on VIF.
    """
    from sklearn.linear_model import LinearRegression
    X_imp = X.copy().fillna(X.median())
    vif_data = []
    for i, col in enumerate(X_imp.columns):
        X_other = X_imp.drop(columns=[col])
        y_col   = X_imp[col]
        r2 = LinearRegression().fit(X_other, y_col).score(X_other, y_col)
        vif = 1 / (1 - r2) if r2 < 1.0 else float('inf')
        vif_data.append({'Feature': col, 'VIF': round(vif, 3)})
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

# ─── Permutation Importance ──────────────────────────────────────────────────

def compute_permutation_importance(X_train: pd.DataFrame,
                                   y_train: pd.Series) -> pd.DataFrame:
    """
    Permutation importance from the trained model.
    More reliable than XGBoost's built-in gain importance for correlated features.
    Uses 10 repeats to estimate variance of importance estimates.
    """
    if not os.path.exists(MODEL_PATH):
        print("  [SKIP] model.pkl not found — run 04_model_training.py first.")
        return pd.DataFrame()

    model = joblib.load(MODEL_PATH)
    result = permutation_importance(
        model, X_train, y_train,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='roc_auc'
    )
    df = pd.DataFrame({
        'Feature':   FEATURE_COLS,
        'PI_mean':   result.importances_mean.round(4),
        'PI_std':    result.importances_std.round(4),
    }).sort_values('PI_mean', ascending=False).reset_index(drop=True)
    df['Rank_PI'] = range(1, len(df) + 1)
    return df
