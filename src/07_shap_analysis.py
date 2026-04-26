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

# 3. Individual SHAP Waterfall (RQ4) 

def plot_individual_waterfall(model, X_unseen: pd.DataFrame,
                               y_unseen: pd.Series, threshold: float,
                               label: str, filename: str) -> None:
    """Plot a SHAP waterfall for a specific case from the UNSEEN HOLDOUT."""
    explainer   = get_explainer(model)
    shap_values = explainer(X_unseen)

    proba = model.predict_proba(X_unseen)[:, 1]
    pred  = (proba >= threshold).astype(int)
    y_arr = np.array(y_unseen)

    if label == 'tp':
        candidates = np.where((y_arr == 1) & (pred == 1))[0]
        title = 'True Positive — Correctly Identified LBW Case (Unseen Holdout)'
    elif label == 'fn':
        candidates = np.where((y_arr == 1) & (pred == 0))[0]
        title = 'False Negative — Missed LBW Case (Unseen Holdout)\n(Supports Limitations Section)'
    else:
        candidates = np.arange(len(y_unseen))
        title = f'SHAP Waterfall — {label}'

    if len(candidates) == 0:
        print(f"  [SHAP] No {label.upper()} cases found in unseen holdout — skipping.")
        return

    idx      = candidates[np.argmax(proba[candidates])]
    case_proba = proba[idx]
    actual    = y_arr[idx]

    print(f"\n  [SHAP WATERFALL] {label.upper()} case: "
          f"prob={case_proba:.4f}, actual={actual}, pred={pred[idx]}")

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=13, show=False)
    plt.title(title, fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")

# 4. Beeswarm Plot 

def plot_beeswarm(shap_values, X_test: pd.DataFrame) -> None:
    """SHAP beeswarm plot: each dot is one observation in X_test."""
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values.values,
        X_test,
        feature_names=FEATURE_COLS,
        max_display=13,
        show=False,
        plot_type='dot'
    )
    plt.title(
        'SHAP Beeswarm — Feature Impact on LBW Probability\n'
        'Each dot = one case in test set | Red = high value | Blue = low value\n'
        '(RQ4: Individual-level decision pathways)',
        fontweight='bold', fontsize=10
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] shap_beeswarm.png")
