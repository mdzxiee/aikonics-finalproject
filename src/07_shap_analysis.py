# Purpose : SHAP explainability for RQ1 (global feature ranking) and
#           RQ4 (individual decision pathways for BHWs).
#
# Input  : artifacts/model.pkl
#          artifacts/X_test.csv + y_test.csv    (global SHAP — larger sample)
#          artifacts/X_unseen.csv + y_unseen.csv (waterfall — methodologically sealed)
#          artifacts/threshold.pkl
#
# Output : outputs/shap_bar.png              (global importance — RQ1)
#          outputs/shap_beeswarm.png          (population patterns)
#          outputs/shap_waterfall_tp.png      (True Positive — RQ4)
#          outputs/shap_waterfall_fn.png      (False Negative — clinical insight)
#          outputs/shap_global_ranking.csv    (RQ1 evidence table)
#
# CORRECTIONS vs. monolithic version:
#   1. feature_perturbation='tree_path_dependent' — handles NaN correctly
#      (No SimpleImputer in pipeline means raw NaN inputs. Default 'interventional'
#       SHAP fails with NaN; tree_path_dependent follows actual tree paths.)
#   2. Global SHAP → X_test (larger, ~317 rows, more stable estimates)
#   3. Individual waterfall → X_unseen (methodologically sealed partition)
#      This distinction is explicitly reported in the thesis.
#   4. Cross-validation between SHAP and permutation importance rankings
#      (from 03_feature_selection.py) to confirm stability.
#
# WHY SPLIT USAGE (test vs. unseen for SHAP):
#   Global mean |SHAP| requires many samples to stabilize. X_unseen has
#   only ~175 rows and ~21 LBW cases — SHAP estimates would be noisy.
#   X_test has ~317 rows and ~38 LBW cases — 2× more stable.
#   Individual waterfall plots use X_unseen because citing a prediction from
#   a truly sealed partition strengthens the generalization claim.
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
   """
   Build SHAP TreeExplainer with tree_path_dependent perturbation.

   WHY tree_path_dependent (not default 'interventional'):
     The default 'interventional' SHAP computes counterfactual predictions
     by substituting background data values into the model. When features
     contain NaN (which XGBoost handles via learned default directions),
     interventional perturbation produces incorrect counterfactuals because
     it does not follow the tree's NaN path.

     tree_path_dependent follows the ACTUAL PATH each observation takes
     through every tree in the ensemble, including NaN branches. It is both
     correct and computationally efficient for tree models.

   VERIFIED: shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
   produces correct SHAP values on NaN-containing inputs. (Confirmed in
   environment testing before pipeline construction.)
   """
   explainer = shap.TreeExplainer(
       model,
       feature_perturbation='tree_path_dependent'
   )
   return explainer

# 2. Global SHAP Importance (RQ1)
def compute_global_shap(model, X_test: pd.DataFrame,
                        y_test: pd.Series) -> tuple:
   """
   Compute global SHAP importance on the TEST SET.

   Uses X_test (not X_unseen) because:
     - X_test has ~317 rows vs. ~175 in X_unseen
     - Mean |SHAP| stabilizes with more samples
     - LBW minority class: ~38 cases in test vs. ~21 in unseen
     - More LBW examples produce more stable SHAP estimates for the
       minority class — critical for correct feature attribution

   The use of test data here is limited to EXPLANATION purposes only.
   The model was not trained on test data. SHAP explains what the model
   learned, not what it learns from test data.
   """
   explainer   = get_explainer(model)
   shap_values = explainer(X_test)

   # shap_values.values shape: (n_samples, n_features)
   # For binary XGBoost, this is the log-odds SHAP values for class 1 (LBW)
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
   """
   Plot a SHAP waterfall for a specific case from the UNSEEN HOLDOUT.

   WHY X_unseen for waterfall plots:
     Waterfall plots are individual-level explanations used to demonstrate
     'Here is WHY this specific mother was flagged.' Using a case from
     the sealed unseen holdout strengthens the generalization claim:
     the model correctly explains a prediction for a mother it has
     NEVER seen in any form during training or threshold selection.

   Case selection:
     True Positive (TP): Model correctly flags a real LBW case.
       → Primary demonstration case for the thesis.
     False Negative (FN): Model misses a real LBW case.
       → Shows what patterns the model FAILS to detect.
       → Supports the limitations discussion.
   """
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

   # Select case with highest ML probability among candidates (most interpretable)
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
   """
   SHAP beeswarm plot: each dot is one observation in X_test.
   Red = high feature value, Blue = low.
   Horizontal position = SHAP value (impact on log-odds of LBW).
   Directly answers RQ4: 'What individual-level pathways does SHAP reveal?'
   """
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
   """
   Horizontal bar plot of mean |SHAP| per feature.
   Primary figure for answering RQ1 in the thesis.
   """
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
   # Annotate top 3
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
   """
   Compare SHAP global ranking with permutation importance from 03.
   If rankings are substantially different, flag for investigation.
   """
   pi_path = os.path.join(
       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
       'outputs', 'feature_permutation_importance.csv'
   )
   if not os.path.exists(pi_path):
       print(f"\n  [CROSS-VAL] feature_permutation_importance.csv not found.")
       print(f"  Run 03_feature_selection.py after 04_model_training.py to generate.")
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
       print(f"  Possible cause: collinearity splitting credit between features.")
       print(f"  SHAP distributes credit via Shapley values; PI uses random permutation.")
       print(f"  Both rankings are valid. Report both in thesis — discuss discrepancies.")
   else:
       print(f"\n  ✓ SHAP and PI rankings are broadly consistent (max diff ≤ 5).")

#  Main
def run_shap_analysis() -> None:
   print("=" * 65)
   print("STAGE 7: SHAP EXPLAINABILITY ANALYSIS")
   print("=" * 65)
   print("\n  RQ1: Global feature importance → X_test (more stable sample)")
   print("  RQ4: Individual waterfall   → X_unseen (sealed partition)")

   model     = joblib.load(MODEL_PATH)
   threshold = joblib.load(THRESHOLD_PATH)

   X_test   = pd.read_csv(X_TEST_PATH)
   y_test   = pd.read_csv(Y_TEST_PATH).squeeze()
   X_unseen = pd.read_csv(X_UNSEEN_PATH)
   y_unseen = pd.read_csv(Y_UNSEEN_PATH).squeeze()

   os.makedirs(OUTPUTS_DIR, exist_ok=True)

   print(f"\n  [DATA] X_test : {X_test.shape}  | LBW: {y_test.sum()} cases")
   print(f"  [DATA] X_unseen: {X_unseen.shape} | LBW: {y_unseen.sum()} cases")
   print(f"  [SHAP] feature_perturbation = tree_path_dependent (NaN-safe)")

   # Global SHAP on X_test
   shap_values, ranking_df = compute_global_shap(model, X_test, y_test)

   # Save global ranking
   ranking_df.to_csv(os.path.join(OUTPUTS_DIR, 'shap_global_ranking.csv'), index=False)
   print(f"\n  [SAVED] shap_global_ranking.csv  (RQ1 evidence)")

   # Plots
   plot_shap_bar(ranking_df)
   plot_beeswarm(shap_values, X_test)

   # Individual waterfall plots on X_unseen
   plot_individual_waterfall(
       model, X_unseen, y_unseen, threshold,
       label='tp', filename='shap_waterfall_tp.png'
   )
   plot_individual_waterfall(
       model, X_unseen, y_unseen, threshold,
       label='fn', filename='shap_waterfall_fn.png'
   )

   # Cross-validate with permutation importance from 03
   cross_validate_importance(ranking_df)

   print(f"\n  [SHAP ANALYSIS COMPLETE]")
   print(f"  Outputs: shap_bar.png, shap_beeswarm.png,")
   print(f"           shap_waterfall_tp.png, shap_waterfall_fn.png,")
   print(f"           shap_global_ranking.csv")

if __name__ == "__main__":
   run_shap_analysis()
