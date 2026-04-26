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

# 5. Global Importance Bar Plot (RQ1) 

def plot_shap_bar(ranking_df: pd.DataFrame) -> None:
    """Horizontal bar plot of mean |SHAP| per feature."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sorted_df = ranking_df.sort_values('Mean_|SHAP|')
    colors    = ['#C00000' if i == len(sorted_df) - 1 else '#4472C4'
                 for i in range(len(sorted_df))]
    ax.barh(sorted_df['Feature'], sorted_df['Mean_|SHAP|'],
            color=colors, edgecolor='white')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title(
        'XGBoost Global Feature Importance via SHAP\n'
        '(Answers RQ1: Which variables have highest predictive utility?)\n'
        f'Computed on X_test ({sum(ranking_df["Rank"] > 0)} features)',
        fontweight='bold', fontsize=11
    )
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    for i, (_, row) in enumerate(sorted_df.tail(3).iterrows()):
        ax.text(row['Mean_|SHAP|'] + 0.0002, i + len(sorted_df) - 3,
                f'#{int(row["Rank"])}', va='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] shap_bar.png")

# 6. SHAP vs. Permutation Importance Cross-Validation

def cross_validate_importance(ranking_df: pd.DataFrame) -> None:
    """Compare SHAP global ranking with permutation importance from 03."""
    pi_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'feature_permutation_importance.csv'
    )
    if not os.path.exists(pi_path):
        print(f"\n  [CROSS-VAL] feature_permutation_importance.csv not found.")
        return

    pi_df = pd.read_csv(pi_path)
    merged = ranking_df[['Feature', 'Rank']].merge(
        pi_df[['Feature', 'Rank_PI']], on='Feature', how='left'
    )
    merged['Rank_diff'] = (merged['Rank'] - merged['Rank_PI']).abs()
    merged = merged.sort_values('Rank')

    print(f"\n  [CROSS-VAL] SHAP vs Permutation Importance Ranking:")
    print(f"  {'Feature':<24} {'SHAP Rank':>10} {'PI Rank':>10} {'|Diff|':>8}")
    print(f"  {'-'*55}")
    for _, row in merged.iterrows():
        flag = ' ⚠' if row['Rank_diff'] > 5 else ''
        pi_r = int(row['Rank_PI']) if not pd.isna(row['Rank_PI']) else 'N/A'
        print(f"  {row['Feature']:<24} {int(row['Rank']):>10} {str(pi_r):>10} "
              f"{int(row['Rank_diff']) if not pd.isna(row['Rank_diff']) else 'N/A':>8}{flag}")

    large_diffs = merged[merged['Rank_diff'] > 5]
    if len(large_diffs) > 0:
        print(f"\n  ⚠  {len(large_diffs)} features with rank difference > 5.")
    else:
        print(f"\n  ✓ SHAP and PI rankings are broadly consistent (max diff ≤ 5).")

#  Main 

def run_shap_analysis() -> None:
    print("=" * 65)
    print("STAGE 7: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 65)

    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    X_test   = pd.read_csv(X_TEST_PATH)
    y_test   = pd.read_csv(Y_TEST_PATH).squeeze()
    X_unseen = pd.read_csv(X_UNSEEN_PATH)
    y_unseen = pd.read_csv(Y_UNSEEN_PATH).squeeze()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
