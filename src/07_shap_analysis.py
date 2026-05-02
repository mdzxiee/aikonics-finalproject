# Purpose : SHAP explainability for RQ1 (global feature ranking) and
#           RQ4 (individual decision pathways for BHWs).
#
# Input  : artifacts/model.pkl
#          artifacts/threshold.pkl
#          artifacts/X_test.csv  + y_test.csv    ← global SHAP (larger, stable)
#          artifacts/X_unseen.csv + y_unseen.csv  ← waterfall (sealed partition)
#
# Output : outputs/shap_bar.png                  ← RQ1 evidence (paper)
#          outputs/shap_beeswarm.png              ← RQ4 population patterns
#          outputs/shap_waterfall_tp.png          ← RQ4 True Positive (unseen)
#          outputs/shap_waterfall_fn.png          ← False Negative (limitations)
#          outputs/shap_global_ranking.csv        ← RQ1 ranked feature table
#          outputs/shap_vs_pi_comparison.png      ← cross-validation figure
#
#  KEY TECHNICAL DECISIONS 
#
#   feature_perturbation='tree_path_dependent':
#     REQUIRED because the model was trained without SimpleImputer.
#     The model receives raw NaN values at inference time.
#     Default 'interventional' mode substitutes background values at split
#     points — it cannot account for the NaN branch direction learned
#     during training, producing incorrect Shapley attributions.
#     tree_path_dependent follows the ACTUAL PATH each observation takes
#     through every tree, including the NaN direction.
#     Verified working with NaN inputs on this XGBoost version.
#
#   Global SHAP → X_test (~317 rows, ~34 LBW cases):
#     More LBW cases = more stable mean |SHAP| estimates.
#     Using X_unseen (~176 rows, ~18 LBW cases) would produce noisier
#     global importance rankings because fewer minority-class examples
#     contribute to the average.
#
#   Individual waterfall → X_unseen (sealed partition):
#     Waterfall plots explain a specific prediction from data the model
#     has NEVER seen in any form — not during training, not during
#     threshold selection. This is the strongest available methodological
#     claim: "here is a correct individual explanation on a truly new case."
#
# Connects to: 08_export_artifacts.py
# ===========================================================================

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OUTPUTS_DIR, FEATURE_COLS,
    X_TEST_PATH, Y_TEST_PATH, X_UNSEEN_PATH, Y_UNSEEN_PATH
)

# SECTION 1 — SHAP EXPLAINER SETUP

def build_explainer(model) -> shap.TreeExplainer:
    """
    Build SHAP TreeExplainer with tree_path_dependent perturbation.

    WHY tree_path_dependent (not default 'interventional'):
      The model is a raw XGBClassifier trained without a SimpleImputer.
      Observations can contain NaN values (birth_interval for non-first-borns,
      etc.). The default 'interventional' SHAP substitutes background-dataset
      values at split nodes to compute counterfactuals. When the actual split
      direction for NaN observations is different from the substituted value's
      direction, the counterfactual is wrong — producing incorrect Shapley
      attributions.

      tree_path_dependent computes Shapley values by following the exact path
      each observation takes through every tree in the ensemble, including
      the learned NaN direction. This is both correct AND computationally
      faster for tree-based models.

    Verified working with NaN inputs on XGBoost 3.x.
    """
    return shap.TreeExplainer(
        model,
        feature_perturbation='tree_path_dependent'
    )

# SECTION 2 — GLOBAL SHAP IMPORTANCE (RQ1)

def compute_global_shap(model, X_test: pd.DataFrame,
                         y_test: pd.Series) -> tuple:
    """
    Compute global SHAP importance on X_test.

    WHY X_test (not X_unseen) for global SHAP:
      Global mean |SHAP| is an average across all observations.
      X_test has ~317 rows and ~34 LBW cases.
      X_unseen has ~176 rows and ~18 LBW cases.
      With fewer LBW examples, the mean |SHAP| for LBW-related features
      is estimated from a smaller, noisier sample.
      X_test provides 2× more stable global importance estimates.

    The use of test data here is for EXPLANATION purposes only.
    The model was never trained on test data — SHAP explains what the
    model learned from training data, not from test data.
    """
    explainer   = build_explainer(model)
    shap_values = explainer(X_test)

    # shap_values.values shape: (n_samples, n_features)
    # Positive SHAP = feature pushed prediction toward LBW
    # Negative SHAP = feature pushed prediction away from LBW
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    ranking_df = pd.DataFrame({
        'Feature':     FEATURE_COLS,
        'Mean_|SHAP|': mean_abs_shap.round(6),
    }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)

    print(f"\n  [SHAP] Global importance on X_test ({len(X_test)} rows, "
          f"{y_test.sum()} LBW cases)")
    print(f"  {'Rank':<6} {'Feature':<24} {'Mean |SHAP|':>13}  "
          f"{'Clinical meaning'}")
    print(f"  {'-'*75}")

    for _, row in ranking_df.iterrows():
        feat_idx  = FEATURE_COLS.index(row['Feature'])
        feat_vals = X_test.iloc[:, feat_idx].values
        sv_col    = shap_values.values[:, feat_idx]
        
        if np.std(feat_vals) == 0 or np.std(sv_col) == 0:
            direction = 'neutral / constant'
        else:
            corr = np.corrcoef(feat_vals, sv_col)[0, 1]
            direction = 'High value ↑ Risk' if corr > 0 else 'High value ↓ Risk'
            
        print(f"  {int(row['Rank']):<6} {row['Feature']:<24} {row['Mean_|SHAP|']:>13.6f}  {direction}")

    return shap_values, ranking_df

# SECTION 3 — SHAP BAR PLOT (RQ1 — paper figure)

def plot_shap_bar(ranking_df: pd.DataFrame, shap_values,
                   X_test: pd.DataFrame) -> None:
    """
    Horizontal bar chart of mean |SHAP| per feature.
    This is the PRIMARY figure for answering RQ1 in the paper.
    Color indicates dominant direction: red = increases LBW risk,
    blue = decreases LBW risk.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sorted_df = ranking_df.sort_values('Mean_|SHAP|', ascending=True)

    # Determine direction per feature
    bar_colors = []
    for feat in sorted_df['Feature']:
        feat_idx  = FEATURE_COLS.index(feat)
        feat_vals = X_test.iloc[:, feat_idx].values
        sv_col    = shap_values.values[:, feat_idx]
        
        if np.std(feat_vals) == 0 or np.std(sv_col) == 0:
            bar_colors.append('gray')
        else:
            corr = np.corrcoef(feat_vals, sv_col)[0, 1]
            bar_colors.append('#C00000' if corr > 0 else '#4472C4')

    bars = ax.barh(sorted_df['Feature'], sorted_df['Mean_|SHAP|'],
                   color=bar_colors, edgecolor='white', height=0.65)

    # Annotations
    for bar, val in zip(bars, sorted_df['Mean_|SHAP|']):
        ax.text(val + 0.0001, bar.get_y() + bar.get_height() / 2,
                f'{val:.5f}', va='center', fontsize=8)

    red_patch  = mpatches.Patch(color='#C00000',
                                 label='Increases LBW risk (positive SHAP)')
    blue_patch = mpatches.Patch(color='#4472C4',
                                 label='Decreases LBW risk (negative SHAP)')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')

    ax.set_xlabel('Mean |SHAP Value| (log-odds contribution)', fontsize=11)
    ax.set_title(
        'XGBoost Global Feature Importance via SHAP\n'
        '(Answers RQ1: Which variables have highest predictive utility for LBW?)\n'
        f'Computed on X_test ({len(X_test)} rows)',
        fontweight='bold', fontsize=11
    )
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] shap_bar.png")

# SECTION 4 — SHAP BEESWARM (RQ4 — population patterns)

def plot_shap_beeswarm(shap_values, X_test: pd.DataFrame) -> None:
    """
    SHAP beeswarm plot: each dot = one test-set observation.
    Red = high feature value | Blue = low feature value.
    Horizontal position = SHAP value (impact on LBW log-odds).

    Answers RQ4: "What individual-level decision pathways and population-level
    feature rankings does SHAP reveal?"
    Shows DIRECTION of effects that the bar chart cannot show.
    E.g., HIGH anc_first_timing (late ANC) → positive SHAP → increases LBW risk.
    """
    plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=13,
        show=False,
    )
    plt.title(
        'SHAP Beeswarm — Feature Value vs. Impact on LBW Probability\n'
        'Each dot = one test-set case | Red = high value | Blue = low value\n'
        '(RQ4: Individual-level decision pathways and population patterns)',
        fontweight='bold', fontsize=10
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] shap_beeswarm.png")


# SECTION 5 — SHAP WATERFALL PLOTS (RQ4 — individual cases from X_unseen)

def plot_waterfall(model, X_unseen: pd.DataFrame,
                   y_unseen: pd.Series, threshold: float,
                   case_type: str, filename: str) -> None:
    """
    Plot SHAP waterfall for one specific case from the SEALED UNSEEN HOLDOUT.

    WHY X_unseen (not X_test) for waterfall plots:
      Waterfall plots demonstrate an individual-level explanation for a case
      the model has TRULY NEVER seen — not in training, not during threshold
      selection, not during SHAP calibration. Using X_unseen provides the
      strongest possible generalization claim for the panel defense:
      "This is an explanation for a completely new case."

    Case types:
      'tp' = True Positive  → correctly identified LBW case (primary demo)
      'fn' = False Negative → missed LBW case (supports limitations section)
    """
    explainer   = build_explainer(model)
    shap_values = explainer(X_unseen)

    proba = model.predict_proba(X_unseen)[:, 1]
    pred  = (proba >= threshold).astype(int)
    y_arr = np.array(y_unseen)

    if case_type == 'tp':
        candidates = np.where((y_arr == 1) & (pred == 1))[0]
        title      = ('SHAP Waterfall — True Positive (Correctly Identified LBW)\n'
                      'From sealed unseen holdout — model never saw this case\n'
                      '(RQ4: Individual decision pathway)')
    else:
        candidates = np.where((y_arr == 1) & (pred == 0))[0]
        title      = ('SHAP Waterfall — False Negative (Missed LBW Case)\n'
                      'From sealed unseen holdout\n'
                      '(Supports Limitations: cases ML model fails to detect)')

    if len(candidates) == 0:
        print(f"  [SHAP] No {case_type.upper()} cases in unseen holdout — skipping.")
        return

    # Select the case with highest probability among candidates
    # (most interpretable — clearest signal)
    idx        = candidates[np.argmax(proba[candidates])]
    case_proba = proba[idx]
    case_true  = y_arr[idx]

    print(f"  [SHAP] {case_type.upper()} case: prob={case_proba:.4f} | "
          f"actual={case_true} | pred={pred[idx]}")

    # Print the top factors for this case
    sv          = shap_values.values[idx]
    feat_vals   = X_unseen.iloc[idx].values
    top5        = sorted(zip(FEATURE_COLS, sv, feat_vals),
                          key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  Top SHAP factors:")
    for feat, s, fv in top5:
        arrow = '↑ risk' if s > 0 else '↓ risk'
        print(f"    {feat:<24} SHAP={s:+.5f}  value={fv:.2f}  {arrow}")

    plt.figure(figsize=(11, 6))
    shap.plots.waterfall(shap_values[idx], max_display=13, show=False)
    plt.title(title, fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")

# SECTION 6 — SHAP vs PERMUTATION IMPORTANCE CROSS-VALIDATION

def plot_shap_vs_pi_comparison(ranking_df: pd.DataFrame) -> None:
    """
    Compare SHAP global ranking against permutation importance ranking
    from 03_feature_selection.py.

    WHY THIS MATTERS:
      If SHAP and PI rankings are broadly consistent → the feature importance
      story is robust across two independent methods.
      If they diverge significantly → likely due to feature interactions
      (XGBoost splits credit via Shapley; PI measures total permutation effect).
      Either way, both rankings should be reported and discrepancies explained.
    """
    pi_path = os.path.join(OUTPUTS_DIR, 'feature_permutation_importance.csv')
    if not os.path.exists(pi_path):
        print(f"  [SKIP] feature_permutation_importance.csv not found.")
        print(f"         Run 03_feature_selection.py after 04_model_training.py.")
        return

    pi_df  = pd.read_csv(pi_path)
    merged = ranking_df[['Feature', 'Rank']].rename(
        columns={'Rank': 'SHAP_Rank'}
    ).merge(
        pi_df[['Feature', 'Rank_PI']],
        on='Feature', how='left'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'SHAP Global Importance vs. Permutation Importance (PI)\n'
        'Cross-validation of feature ranking methods',
        fontsize=12, fontweight='bold'
    )

    # Left: SHAP ranks
    shap_sorted = ranking_df.sort_values('Mean_|SHAP|', ascending=True)
    axes[0].barh(shap_sorted['Feature'], shap_sorted['Mean_|SHAP|'],
                 color='#C00000', edgecolor='white', height=0.65)
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].set_title('SHAP Global Importance (X_test)', fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.35)

    # Right: Rank comparison scatter
    ax = axes[1]
    valid = merged.dropna(subset=['Rank_PI'])
    ax.scatter(valid['SHAP_Rank'], valid['Rank_PI'],
               color='#4472C4', s=90, zorder=5)
    for _, row in valid.iterrows():
        ax.annotate(row['Feature'],
                    (row['SHAP_Rank'], row['Rank_PI']),
                    fontsize=7.5, xytext=(3, 3),
                    textcoords='offset points')

    max_rank = max(valid['SHAP_Rank'].max(), valid['Rank_PI'].max()) + 1
    ax.plot([1, max_rank], [1, max_rank], 'k--', lw=1, alpha=0.5,
            label='Perfect agreement line')
    ax.set_xlabel('SHAP Rank (1=most important)', fontsize=10)
    ax.set_ylabel('Permutation Importance Rank', fontsize=10)
    ax.set_title('Rank Comparison: SHAP vs. Permutation Importance\n'
                 'Points near diagonal = consistent ranking',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.35)

    # Report discrepancies
    merged['Rank_diff'] = (merged['SHAP_Rank'] - merged['Rank_PI']).abs()
    large_diff = merged[merged['Rank_diff'] > 4].dropna()
    if len(large_diff) > 0:
        print(f"  [SHAP-PI] Features with rank difference > 4:")
        for _, row in large_diff.iterrows():
            print(f"    {row['Feature']:<24} "
                  f"SHAP={int(row['SHAP_Rank'])} | PI={int(row['Rank_PI'])} | "
                  f"diff={int(row['Rank_diff'])}")
        print(f"  Likely cause: collinearity distributing credit differently "
              f"between SHAP (Shapley weighting) and PI (permutation test).")
    else:
        print(f"  [SHAP-PI] Rankings broadly consistent (max diff ≤ 4) ✓")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_vs_pi_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] shap_vs_pi_comparison.png")

# MAIN

def run_shap_analysis() -> None:
    print("=" * 70)
    print("STAGE 7: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    print(f"\n  RQ1 (global importance) → X_test  (stable: more LBW cases)")
    print(f"  RQ4 (individual cases)  → X_unseen (sealed: strongest claim)")
    print(f"  Perturbation mode: tree_path_dependent (NaN-safe, required)")

    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    X_test   = pd.read_csv(X_TEST_PATH)
    y_test   = pd.read_csv(Y_TEST_PATH).squeeze()
    X_unseen = pd.read_csv(X_UNSEEN_PATH)
    y_unseen = pd.read_csv(Y_UNSEEN_PATH).squeeze()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"\n  X_test  : {X_test.shape}  | LBW: {y_test.sum()} cases")
    print(f"  X_unseen: {X_unseen.shape} | LBW: {y_unseen.sum()} cases")

    # 1. Global SHAP on X_test (RQ1)
    shap_values, ranking_df = compute_global_shap(model, X_test, y_test)

    # 2. Save global ranking table
    ranking_df.to_csv(
        os.path.join(OUTPUTS_DIR, 'shap_global_ranking.csv'), index=False
    )
    print(f"\n  [SAVED] shap_global_ranking.csv  (RQ1 evidence table)")

    # 3. SHAP bar chart (paper — RQ1)
    plot_shap_bar(ranking_df, shap_values, X_test)

    # 4. SHAP beeswarm (paper — RQ4 population patterns)
    plot_shap_beeswarm(shap_values, X_test)

    # 5. SHAP waterfall plots on X_unseen (paper — RQ4 individual cases)
    print(f"\n  [SHAP] Individual waterfall plots (X_unseen — sealed partition):")
    plot_waterfall(model, X_unseen, y_unseen, threshold,
                   case_type='tp', filename='shap_waterfall_tp.png')
    plot_waterfall(model, X_unseen, y_unseen, threshold,
                   case_type='fn', filename='shap_waterfall_fn.png')

    # 6. Cross-validate SHAP vs PI rankings
    print(f"\n  [SHAP] Cross-validation against Permutation Importance:")
    plot_shap_vs_pi_comparison(ranking_df)

    print(f"\n  {'═'*60}")
    print(f"  STAGE 7 COMPLETE")
    print(f"  Outputs:")
    print(f"    shap_bar.png              ← RQ1 (paper figure)")
    print(f"    shap_beeswarm.png         ← RQ4 population patterns")
    print(f"    shap_waterfall_tp.png     ← RQ4 True Positive (unseen)")
    print(f"    shap_waterfall_fn.png     ← Limitations discussion")
    print(f"    shap_global_ranking.csv   ← RQ1 evidence table")
    print(f"    shap_vs_pi_comparison.png ← method cross-validation")
    print(f"  Next: python src/08_export_artifacts.py")
    print(f"  {'═'*60}")


if __name__ == "__main__":
    run_shap_analysis()