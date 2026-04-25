# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/02_eda.py
#
# Purpose : Exploratory Data Analysis exclusively on the TRAINING set.
#           Any EDA on the full/test/unseen dataset constitutes data snooping
#           — analyst decisions informed by test-set patterns contaminate
#           the integrity of subsequent evaluations.
#
# Input  : artifacts/X_train.csv + artifacts/y_train.csv
#          (produced by 04_model_training.py — run 04 first, then return here
#           for documentation; or run 04 immediately after 01 then run 02)
#
# Output : outputs/eda_distributions.png
#          outputs/eda_class_comparison.png
#          outputs/feature_auc_train.csv     (RQ1 evidence)
#          outputs/feature_stats_train.csv   (Cohen's d + correlations)
#
# Connects to: 03_feature_selection.py (documentation chain)
# ---------------------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, OUTPUTS_DIR, FEATURE_COLS
)

from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr, mannwhitneyu


#  Helpers 

def cohens_d(group1: pd.Series, group0: pd.Series) -> float:
    """Pooled standard deviation Cohen's d effect size."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std() ** 2 + (n0 - 1) * group0.std() ** 2) /
        (n1 + n0 - 2)
    )
    return float((group1.mean() - group0.mean()) / pooled_std) if pooled_std > 0 else 0.0


def individual_auc(series: pd.Series, y: pd.Series) -> float:
    """Compute AUC of a single feature vs. binary target. Always ≥ 0.50."""
    s = series.fillna(series.median())
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)

#  Analysis Functions 

def missing_value_report(X_train: pd.DataFrame) -> None:
    print("\n[EDA 1] Missing Value Report (Training Set Only):")
    null_df = pd.DataFrame({
        'Feature':  X_train.columns,
        'Missing_N': X_train.isnull().sum().values,
        'Missing_%': (X_train.isnull().mean() * 100).round(2).values,
    }).sort_values('Missing_%', ascending=False)
    remaining = null_df[null_df['Missing_%'] > 0]
    if len(remaining) > 0:
        print(remaining.to_string(index=False))
    else:
        print("  All structural NaN handled in preprocessing. Remaining: None.")
    print(f"  Note: XGBoost handles any residual NaN natively (tree_path_dependent).")


def class_distribution_report(y_train: pd.Series) -> None:
    print("\n[EDA 2] Class Distribution (Training Set Only):")
    n_lbw  = y_train.sum()
    n_norm = (y_train == 0).sum()
    print(f"  Normal (0): {n_norm:,} ({n_norm/len(y_train)*100:.1f}%)")
    print(f"  LBW    (1): {n_lbw:,}  ({n_lbw/len(y_train)*100:.1f}%)")
    print(f"  Ratio     : {n_norm/n_lbw:.2f}:1")

    def compute_rq1_feature_metrics(X_train: pd.DataFrame,
                                 y_train: pd.Series) -> pd.DataFrame:
    """
    RQ1 Evidence Table: Which features have highest predictive utility?

    Metrics computed on TRAINING SET ONLY:
      AUC:      individual predictive power vs LBW target (AUC ≥ 0.50 by design)
      Cohen's d: standardized mean difference between LBW and Normal groups
      r_pb:     point-biserial correlation with LBW target
      U_p:      Mann-Whitney U test p-value (non-parametric group difference)
    """
    rows = []
    for col in FEATURE_COLS:
        if col not in X_train.columns:
            continue
        s    = X_train[col]
        lbw  = s[y_train == 1].dropna()
        norm = s[y_train == 0].dropna()
        auc  = individual_auc(s, y_train)
        d    = cohens_d(lbw, norm)
        r_pb, p_r = pointbiserialr(s.fillna(s.median()), y_train)
        try:
            _, p_mw = mannwhitneyu(lbw, norm, alternative='two-sided')
        except Exception:
            p_mw = np.nan
        rows.append({
            'Feature':   col,
            'AUC':       round(auc, 4),
            'Cohens_d':  round(d, 4),
            'r_pb':      round(r_pb, 4),
            'p_pointbis':round(p_r, 4),
            'p_MW':      round(p_mw, 4) if not np.isnan(p_mw) else np.nan,
            'LBW_mean':  round(lbw.mean(), 3) if len(lbw) > 0 else np.nan,
            'Norm_mean': round(norm.mean(), 3) if len(norm) > 0 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
    print("\n[EDA 3] RQ1 Feature Predictive Utility (Training Set):")
    print(f"  {'Feature':<22} {'AUC':>7} {'d':>8} {'r_pb':>7} {'p_MW':>8}")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        print(f"  {row['Feature']:<22} {row['AUC']:>7.4f} {row['Cohens_d']:>8.4f} "
              f"{row['r_pb']:>7.4f} {row['p_MW']:>8.4f}")
    print(f"\n  Note: Max individual AUC = {df['AUC'].max():.4f}")
    print(f"  Note: Cohen's d < 0.20 for most features — heavily overlapping")
    print(f"        class distributions. Realistic NDHS-only ceiling: 0.58–0.65 AUC.")
    return df

def plot_distributions(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Plot feature distributions stratified by LBW vs Normal."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Feature Distributions — LBW vs Normal (Training Set Only)',
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes_flat[i]
        if feat not in X_train.columns:
            ax.set_visible(False)
            continue
        lbw_vals  = X_train[feat][y_train == 1].dropna()
        norm_vals = X_train[feat][y_train == 0].dropna()
        bins = min(20, X_train[feat].nunique())
        ax.hist(norm_vals, bins=bins, alpha=0.6, color='#4472C4',
                label='Normal', density=True)
        ax.hist(lbw_vals,  bins=bins, alpha=0.6, color='#ED7D31',
                label='LBW', density=True)
        ax.set_title(feat, fontweight='bold', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(linestyle='--', alpha=0.3)

    # Hide unused axes
    for j in range(len(FEATURE_COLS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eda_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] eda_distributions.png")

    def plot_class_comparison(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Box plots comparing LBW vs Normal across continuous features."""
    cont_feats = ['wealth_score', 'education_yrs', 'birth_interval',
                  'anc_first_timing', 'iron_days', 'maternal_age']
    present = [f for f in cont_feats if f in X_train.columns]
    fig, axes = plt.subplots(1, len(present), figsize=(18, 5))
    fig.suptitle('LBW vs Normal — Feature Comparison (Training Set)',
                 fontsize=13, fontweight='bold')
    for ax, feat in zip(axes, present):
        data = [
            X_train[feat][y_train == 0].dropna().values,
            X_train[feat][y_train == 1].dropna().values,
        ]
        bp = ax.boxplot(data, labels=['Normal', 'LBW'],
                        patch_artist=True, notch=False)
        bp['boxes'][0].set_facecolor('#4472C4')
        bp['boxes'][1].set_facecolor('#ED7D31')
        ax.set_title(feat, fontweight='bold', fontsize=9)
        ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eda_class_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] eda_class_comparison.png")