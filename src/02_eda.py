# Purpose : Exploratory Data Analysis on TRAINING SET ONLY.
#           Includes statistical feature relevance testing per professor's
#           instruction: "Test which factors are relevant to the target variable."
#
# Input  : artifacts/X_train.csv + artifacts/y_train.csv
#          (produced by 04_model_training.py — run 04 first)
#
# Output : outputs/eda_distributions.png
#          outputs/eda_class_comparison.png
#          outputs/feature_relevance_tests.csv   ← Statistical significance per feature
#          outputs/feature_relevance_plot.png    ← Visual evidence for paper
#          outputs/two_layer_rationale.txt       ← Why 2 layers (for paper documentation)
#
# WHY TRAINING SET ONLY:
#   EDA on the full dataset before splitting is data snooping. If we observe
#   that wealth_score is strongly associated with LBW in the full data and then
#   use that observation to keep it in the feature set, we have used information
#   from the test and unseen sets to make a modeling decision. EDA must be
#   restricted to training data to preserve the integrity of evaluation.
#
# Connects to: 03_feature_selection.py, 04_model_training.py (justification)
# ===========================================================================

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import X_TRAIN_PATH, Y_TRAIN_PATH, OUTPUTS_DIR, FEATURE_COLS

from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, pointbiserialr, chi2_contingency


# SECTION 1 — HELPER FUNCTIONS

def cohens_d(group1: pd.Series, group0: pd.Series) -> float:
    """
    Compute pooled Cohen's d effect size (standardized mean difference).
    Interpretation:
      d < 0.20 → negligible (classes heavily overlap — realistic for NDHS)
      0.20–0.50 → small
      0.50–0.80 → medium
      > 0.80   → large
    """
    n1, n0 = len(group1.dropna()), len(group0.dropna())
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = group1.dropna().std(), group0.dropna().std()
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    return float((group1.dropna().mean() - group0.dropna().mean()) / pooled) if pooled > 0 else np.nan


def classify_variable_type(feat: str, X: pd.DataFrame) -> str:
    """Classify feature strictly by explicit definition."""
    # Only true nominal/categorical variables go here.
    categorical = ['residence_type', 'region', 'iron_supplement', 'marital_status']
    
    # Even if features like tetanus_shots have <5 unique values, 
    # they are ordinal/continuous and belong in Mann-Whitney.
    if feat in categorical:
        return 'categorical'
    return 'continuous'

# SECTION 2 — STATISTICAL FEATURE RELEVANCE TESTS

def run_feature_relevance_tests(X_train: pd.DataFrame,
                                 y_train: pd.Series) -> pd.DataFrame:
    """
    For each feature, run appropriate statistical tests to determine
    whether it is significantly associated with LBW_Risk.

    Per professor's instruction: "Test which factors are relevant to
    the target variable."

    Tests applied:
    ─────────────────────────────────────────────────────────────────
    Continuous features (maternal_age, wealth_score, education_yrs, etc.):
      1. Mann-Whitney U test (non-parametric — correct because features
         are not normally distributed in LBW/Normal groups)
      2. Point-biserial correlation (r_pb, p-value)
      3. Individual ROC-AUC (threshold-free discriminative power)
      4. Cohen's d (effect size — tells us HOW different the groups are)

    Categorical features (residence_type, region, iron_supplement, etc.):
      1. Chi-square test of independence
      2. Cramér's V (effect size for chi-square)

    Significance threshold: p < 0.05 (two-tailed)
    ─────────────────────────────────────────────────────────────────

    WHY NON-PARAMETRIC (Mann-Whitney) INSTEAD OF t-test:
      The t-test assumes normal distribution within each group.
      NDHS sociodemographic variables (wealth_score, birth_interval,
      iron_days) are heavily skewed — the normality assumption is violated.
      Mann-Whitney U tests whether LBW and Normal groups have the same
      probability distribution, without assuming normality.
    """
    results = []

    lbw_mask  = y_train == 1
    norm_mask = y_train == 0
    n_lbw     = lbw_mask.sum()
    n_norm    = norm_mask.sum()

    print(f"\n  Feature Relevance Tests | n_LBW={n_lbw} | n_Normal={n_norm}")
    print(f"  {'Feature':<24} {'Type':<12} {'Test':<15} {'Statistic':>10} {'p-value':>10} "
          f"{'Effect':>8} {'AUC':>7} {'Significant':>12}")
    print(f"  {'-'*100}")

    for feat in FEATURE_COLS:
        if feat not in X_train.columns:
            continue

        s = X_train[feat]
        var_type = classify_variable_type(feat, X_train)

        # Create valid mask to drop NaNs for this specific feature WITHOUT filling them
        valid_mask = s.notna()
        s_valid = s[valid_mask]
        y_valid = y_train[valid_mask]
        
        lbw_vals  = s_valid[y_valid == 1]
        norm_vals = s_valid[y_valid == 0]

        if var_type == 'continuous':
            # Mann-Whitney U
            try:
                stat, p_mw = mannwhitneyu(lbw_vals, norm_vals, alternative='two-sided')
            except Exception:
                stat, p_mw = np.nan, np.nan

            # Point-biserial correlation (Using ONLY valid rows, NO median fill!)
            try:
                r_pb, p_pb = pointbiserialr(s_valid, y_valid)
            except Exception:
                r_pb, p_pb = np.nan, np.nan

            # Individual AUC (Using ONLY valid rows)
            try:
                auc = roc_auc_score(y_valid, s_valid)
                auc = max(auc, 1 - auc)   # always ≥ 0.50
            except Exception:
                auc = np.nan

            # Cohen's d
            d = cohens_d(lbw_vals, norm_vals)

            significant = '✓ YES' if (not np.isnan(p_mw) and p_mw < 0.05) else '✗ NO'
            print(f"  {feat:<24} {'continuous':<12} {'Mann-Whitney':<15} "
                  f"{stat:>10.2f} {p_mw:>10.4f} "
                  f"{d:>8.3f} {auc:>7.4f} {significant:>12}")

            results.append({
                'Feature':      feat,
                'Type':         'continuous',
                'Test':         'Mann-Whitney U',
                'Statistic':    round(float(stat), 3) if not np.isnan(stat) else np.nan,
                'p_value':      round(float(p_mw),  4) if not np.isnan(p_mw) else np.nan,
                'Effect_Size':  round(float(d),     3) if not np.isnan(d) else np.nan,
                'Effect_Label': 'Cohen\'s d',
                'Individual_AUC': round(float(auc), 4) if not np.isnan(auc) else np.nan,
                'r_pb':         round(float(r_pb),  4) if not np.isnan(r_pb) else np.nan,
                'Significant_p05': p_mw < 0.05 if not np.isnan(p_mw) else False,
                'LBW_Mean':     round(float(lbw_vals.mean()),  3) if len(lbw_vals) > 0 else np.nan,
                'Normal_Mean':  round(float(norm_vals.mean()), 3) if len(norm_vals) > 0 else np.nan,
            })

        else:  # categorical
            # Chi-square test (Using ONLY valid rows)
            try:
                ct = pd.crosstab(s_valid.round(0).astype(int), y_valid)
                chi2, p_chi, dof, expected = chi2_contingency(ct)
                n  = len(y_valid)
                cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else np.nan
            except Exception:
                chi2, p_chi, cramer_v = np.nan, np.nan, np.nan

            try:
                auc = roc_auc_score(y_valid, s_valid)
                auc = max(auc, 1 - auc)
            except Exception:
                auc = np.nan

            significant = '✓ YES' if (not np.isnan(p_chi) and p_chi < 0.05) else '✗ NO'
            print(f"  {feat:<24} {'categorical':<12} {'Chi-Square':<15} "
                  f"{chi2:>10.3f} {p_chi:>10.4f} "
                  f"{cramer_v:>8.3f} {auc:>7.4f} {significant:>12}")

            results.append({
                'Feature':      feat,
                'Type':         'categorical',
                'Test':         'Chi-Square',
                'Statistic':    round(float(chi2),     3) if not np.isnan(chi2) else np.nan,
                'p_value':      round(float(p_chi),    4) if not np.isnan(p_chi) else np.nan,
                'Effect_Size':  round(float(cramer_v), 3) if not np.isnan(cramer_v) else np.nan,
                'Effect_Label': "Cramér's V",
                'Individual_AUC': round(float(auc),   4) if not np.isnan(auc) else np.nan,
                'r_pb':         np.nan,
                'Significant_p05': p_chi < 0.05 if not np.isnan(p_chi) else False,
                'LBW_Mean':     round(float(lbw_vals.mean()),  3) if len(lbw_vals) > 0 else np.nan,
                'Normal_Mean':  round(float(norm_vals.mean()), 3) if len(norm_vals) > 0 else np.nan,
            })

    df = pd.DataFrame(results).sort_values('Individual_AUC', ascending=False).reset_index(drop=True)
    df['Rank_by_AUC'] = range(1, len(df) + 1)

    n_sig = df['Significant_p05'].sum()
    print(f"\n  SUMMARY: {n_sig}/{len(df)} features significantly associated with LBW (p < 0.05)")
    if n_sig < len(df) // 2:
        print(f"  NOTE: Low feature-level significance is expected when Cohen's d < 0.20.")
        print(f"        XGBoost can still learn from COMBINATIONS of weak features")
        print(f"        (non-linear interactions) even when individual tests are non-significant.")
    return df


# SECTION 3 — FEATURE RELEVANCE VISUALIZATION

def plot_feature_relevance(rel_df: pd.DataFrame) -> None:
    """
    Two-panel figure:
      Left:  Individual AUC per feature (bars), with significance markers
      Right: Effect size per feature (Cohen's d or Cramér's V)
    Used directly in the paper as evidence for RQ1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'Feature Relevance Analysis — Training Set\n'
        '(Answers RQ1: Which variables have highest predictive utility for LBW?)',
        fontsize=13, fontweight='bold'
    )

    sorted_df = rel_df.sort_values('Individual_AUC', ascending=True)

    #  Left: Individual AUC 
    ax = axes[0]
    colors = ['#C00000' if sig else '#4472C4' for sig in sorted_df['Significant_p05']]
    bars = ax.barh(sorted_df['Feature'], sorted_df['Individual_AUC'],
                   color=colors, edgecolor='white', height=0.65)
    ax.axvline(0.50, color='gray', linestyle='--', lw=1.5,
               label='Random baseline (0.50)', alpha=0.8)
    ax.set_xlabel('Individual Feature AUC', fontsize=10)
    ax.set_title('Individual Predictive Utility (AUC)\nRed = statistically significant (p<0.05)',
                 fontweight='bold', fontsize=10)
    ax.set_xlim(0.48, None)
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    sig_patch  = mpatches.Patch(color='#C00000', label='Significant (p<0.05)')
    nsig_patch = mpatches.Patch(color='#4472C4', label='Not significant')
    rand_line  = plt.Line2D([0],[0], color='gray', linestyle='--', label='Random (0.50)')
    ax.legend(handles=[sig_patch, nsig_patch, rand_line], fontsize=8, loc='lower right')

    for bar, val in zip(bars, sorted_df['Individual_AUC']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7.5)

    #  Right: Effect Size
    ax2 = axes[1]
    abs_effect = sorted_df['Effect_Size'].abs()
    eff_colors = ['#C00000' if sig else '#70AD47'
                  for sig in sorted_df['Significant_p05']]
    bars2 = ax2.barh(sorted_df['Feature'], abs_effect,
                     color=eff_colors, edgecolor='white', height=0.65)
    ax2.axvline(0.20, color='orange', linestyle='--', lw=1.5,
                label="Cohen's d = 0.20 (small effect boundary)", alpha=0.8)
    ax2.set_xlabel("Effect Size (|Cohen's d| or Cramér's V)", fontsize=10)
    ax2.set_title("Effect Size per Feature\nd < 0.20 = negligible overlap\n(Expected for NDHS sociodemographic data)",
                  fontweight='bold', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.35)

    # Reference lines for d interpretation
    for xv, lbl, clr in [(0.20,'small','orange'), (0.50,'medium','#FFC000')]:
        ax2.axvline(xv, color=clr, linestyle=':', lw=1.2, alpha=0.6)
        ax2.text(xv+0.005, len(sorted_df)-0.5, lbl, color=clr, fontsize=7)

    for bar, val in zip(bars2, abs_effect):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=7.5)

    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_relevance_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] feature_relevance_plot.png")


# SECTION 4 — DISTRIBUTION PLOTS

def plot_distributions(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Feature distributions stratified by LBW vs Normal — training set only."""
    n_feats = len(FEATURE_COLS)
    cols    = 4
    rows    = (n_feats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    fig.suptitle(
        'Feature Distributions — LBW vs Normal (Training Set Only)\n'
        'Confirms overlapping distributions (Cohen\'s d < 0.20 for most features)',
        fontsize=12, fontweight='bold'
    )
    axes_flat = axes.flatten()

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes_flat[i]
        if feat not in X_train.columns:
            ax.set_visible(False)
            continue
        lbw_vals  = X_train[feat][y_train == 1].dropna()
        norm_vals = X_train[feat][y_train == 0].dropna()
        bins = min(20, X_train[feat].nunique())
        ax.hist(norm_vals, bins=bins, alpha=0.55, color='#4472C4',
                label=f'Normal (n={len(norm_vals)})', density=True)
        ax.hist(lbw_vals,  bins=bins, alpha=0.55, color='#ED7D31',
                label=f'LBW (n={len(lbw_vals)})', density=True)
        ax.set_title(feat, fontweight='bold', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(linestyle='--', alpha=0.3)

    for j in range(n_feats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eda_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] eda_distributions.png")


def plot_class_comparison(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Box plots: LBW vs Normal for continuous features."""
    cont = ['wealth_score', 'education_yrs', 'birth_interval',
            'anc_first_timing', 'iron_days', 'maternal_age', 'birth_order']
    present = [f for f in cont if f in X_train.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(20, 5))
    fig.suptitle('LBW vs Normal — Feature Distribution Comparison (Training Set)',
                 fontsize=12, fontweight='bold')

    for ax, feat in zip(axes, present):
        data = [X_train[feat][y_train==0].dropna().values,
                X_train[feat][y_train==1].dropna().values]
        bp = ax.boxplot(data, labels=['Normal','LBW'], patch_artist=True, notch=False)
        bp['boxes'][0].set_facecolor('#4472C4')
        bp['boxes'][1].set_facecolor('#ED7D31')
        ax.set_title(feat, fontweight='bold', fontsize=9)
        ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eda_class_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] eda_class_comparison.png")

# MAIN

def run_eda() -> None:
    print("STAGE 2: EDA + FEATURE RELEVANCE TESTS (TRAINING SET ONLY)")
    print("\n  ⚠  Loading ONLY X_train.csv and y_train.csv")
    print("  ⚠  Test and unseen sets remain sealed throughout EDA")

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(
            f"X_train.csv not found at {X_TRAIN_PATH}.\n"
            "Run 04_model_training.py first."
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"\n  Training rows: {len(X_train):,} | "
          f"LBW: {y_train.sum()} ({y_train.mean()*100:.1f}%) | "
          f"Normal: {(y_train==0).sum()} ({(1-y_train.mean())*100:.1f}%)")

    # 1. Missing values
    print(f"\n[EDA 1] Missing Values (Training Set):")
    null_df = (X_train.isnull().mean()*100).round(2)
    remaining = null_df[null_df > 0]
    if len(remaining) > 0:
        print(remaining.to_string())
        print("  → These are handled by XGBoost's native NaN split finding.")
    else:
        print("  None remaining (all structural NaN resolved in preprocessing).")

    # 2. Feature relevance statistical tests
    print(f"\n[EDA 2] Statistical Feature Relevance Tests:")
    rel_df = run_feature_relevance_tests(X_train, y_train)
    rel_df.to_csv(os.path.join(OUTPUTS_DIR, 'feature_relevance_tests.csv'), index=False)
    print(f"  [SAVED] feature_relevance_tests.csv")

    # 3. Visualizations
    print(f"\n[EDA 3] Generating Figures:")
    plot_feature_relevance(rel_df)
    plot_distributions(X_train, y_train)
    plot_class_comparison(X_train, y_train)

    print(f"\n  {'═'*60}")
    print(f"  EDA COMPLETE — all outputs in {OUTPUTS_DIR}")
    print(f"  Next: Run 03_feature_selection.py or 05_threshold_validation.py")


if __name__ == "__main__":
    run_eda()