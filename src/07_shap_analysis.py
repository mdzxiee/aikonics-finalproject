# Purpose : SHAP explainability for RQ1 (global feature ranking) and
#           RQ4 (individual decision pathways for BHWs).
#
# Connects to: 08_export_artifacts.py
# ------------------------------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OUTPUTS_DIR, FEATURE_COLS,
    X_TEST_PATH, Y_TEST_PATH, X_UNSEEN_PATH, Y_UNSEEN_PATH
)

# 1. TreeExplainer Setup 

def get_explainer(model) -> shap.TreeExplainer:
    """Build SHAP TreeExplainer with tree_path_dependent perturbation."""
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation='tree_path_dependent'
    )
    return explainer

# 2. Global SHAP Importance (RQ1) 

def compute_global_shap(model, X_test: pd.DataFrame,
                         y_test: pd.Series) -> tuple:
    """Compute global SHAP importance on the TEST SET."""
    explainer   = get_explainer(model)
    shap_values = explainer(X_test)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    ranking_df = pd.DataFrame({
        'Feature':     FEATURE_COLS,
        'Mean_|SHAP|': mean_abs_shap,
    }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)

    print(f"\n  [SHAP GLOBAL] Computed on X_test ({len(X_test)} rows, "
          f"{y_test.sum()} LBW cases)")
    print(f"  {'Rank':<6} {'Feature':<24} {'Mean |SHAP|'}")
    print(f"  {'-'*45}")
    for _, row in ranking_df.iterrows():
        print(f"  {int(row['Rank']):<6} {row['Feature']:<24} {row['Mean_|SHAP|']:.5f}")

    return shap_values, ranking_df


