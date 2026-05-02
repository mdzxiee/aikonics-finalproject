# Purpose : Select and save the optimal classification threshold using
#           exclusively OOF probabilities. Also documents the FP problem
#           and why dual-purpose Layer 2 de-escalation addresses it.
#
# Input  : artifacts/oof_probabilities.pkl
# Output : artifacts/threshold.pkl       
#          artifacts/features.json       ← ordered feature list
#          outputs/threshold_analysis.png
#          outputs/threshold_sensitivity_table.csv
#          outputs/threshold_fp_analysis.png   ← FP problem documentation
#
# THRESHOLD LEAKAGE RULE:
#   The threshold is selected on OOF probabilities ONLY.
#   OOF probabilities come from 10-fold StratifiedGroupKFold on training data.
#   Test and unseen sets are NEVER used for threshold selection.
#   Using test data to select the threshold would optimize the threshold
#   for that specific test split — inflating test metrics.
#
# WHY threshold.pkl IS SEPARATE FROM model.pkl:
#   The threshold is a deployment decision, not a model parameter.
#   A different health district or a health authority with different risk
#   tolerance could recalibrate the threshold without retraining the model.
#   Keeping them separate makes the system modular and auditable.
#
# Connects to: 06_evaluation.py (applies this threshold to all 3 sets)
#              prototype/predictor.py (loads threshold at inference time)
# ===========================================================================

import os, sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OOF_PATH, ARTIFACTS_DIR, OUTPUTS_DIR,
    THRESHOLD_PATH, FEATURES_PATH, FEATURE_COLS,
    MIN_RECALL_FLOOR
)

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    average_precision_score, precision_score,
    recall_score, f1_score
)

# SECTION 1 — THRESHOLD SELECTION

def select_threshold(oof_proba: np.ndarray,
                     y_train: pd.Series) -> tuple:
    """
    Recall-prioritized threshold selection on OOF probabilities.

    SELECTION CRITERION:
      Find the HIGHEST threshold where OOF Recall >= MIN_RECALL_FLOOR (0.50).
      'Highest' = most conservative = fewer false positives while still
      meeting the recall floor.

    WHY RECALL FLOOR = 50%:
      A screening system that misses more than 50% of LBW mothers provides
      less benefit than simply monitoring all pregnant women equally.
      50% is the minimum level at which the system adds genuine triage value.

    WHY RECALL-FIRST (not F1-first):
      In prenatal LBW screening:
        False Negative = missed at-risk mother = no early intervention =
          higher probability of LBW delivery = elevated stunting risk.
          Stunting is largely IRREVERSIBLE after 1,000 days. This is
          preventable harm.
        False Positive = unnecessary RHU referral = one extra visit =
          inconvenient but clinically harmless.
      The asymmetric cost justifies prioritizing recall over precision.

    WHY HIGHEST VALID THRESHOLD (not lowest):
      The lowest threshold meeting recall ≥ 50% would flag nearly every
      mother — unworkable for rural RHUs. The highest threshold meeting
      the floor minimizes false positives while guaranteeing the recall
      floor. This is the most operationally efficient choice.
    """
    prec_c, rec_c, thr_c = precision_recall_curve(y_train, oof_proba)
    oof_pr_auc = average_precision_score(y_train, oof_proba)
    actual_roc_auc = roc_auc_score(y_train, oof_proba) 

    cands = []
    for p, r, t in zip(prec_c[:-1], rec_c[:-1], thr_c):
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        cands.append({'threshold': float(t), 'precision': float(p),
                      'recall': float(r), 'f1': float(f1)})
    cand_df = pd.DataFrame(cands)

    valid = cand_df[cand_df['recall'] >= MIN_RECALL_FLOOR]

    if len(valid) > 0:
        best   = valid.loc[valid['threshold'].idxmax()]
        method = f"Recall-prioritized (floor ≥ {MIN_RECALL_FLOOR})"
        fallback = False
    else:
        best     = cand_df.loc[cand_df['f1'].idxmax()]
        method   = f"F1-fallback (no threshold achieves recall ≥ {MIN_RECALL_FLOOR})"
        fallback = True
        print(f"\n  ⚠  WARNING: No threshold achieves Recall ≥ {MIN_RECALL_FLOOR}")
        print(f"  ⚠  Falling back to F1-maximizing threshold.")

    threshold = float(best['threshold'])
    t_prec    = float(best['precision'])
    t_rec     = float(best['recall'])
    t_f1      = float(best['f1'])

    print(f"\n  [THRESHOLD] Method    : {method}")
    print(f"  [THRESHOLD] Value     : {threshold:.4f}")
    print(f"  [THRESHOLD] OOF Recall: {t_rec:.4f}")
    print(f"  [THRESHOLD] OOF Prec  : {t_prec:.4f}")
    print(f"  [THRESHOLD] OOF PR-AUC: {oof_pr_auc:.4f}")

    if not fallback:
        pred   = (oof_proba >= threshold).astype(int)
        tp     = int(((pred==1) & (y_train==1)).sum())
        fn     = int(((pred==0) & (y_train==1)).sum())
        fp     = int(((pred==1) & (y_train==0)).sum())
        tn     = int(((pred==0) & (y_train==0)).sum())
        ref_rt = pred.mean() * 100
        print(f"\n  [OOF CM]  TP={tp} | FN={fn} | FP={fp} | TN={tn}")
        print(f"  [OOF CM]  Referral rate = {ref_rt:.1f}%")
        print(f"  [OOF CM]  FP/TP ratio  = {fp/tp:.1f}:1 ")
        print(f"\n  ⚠  NOTE ON HIGH FP RATE:")
        print(f"  At ROC-AUC {actual_roc_auc:.3f}, raising the threshold to cut FP also")
        print(f"  cuts TP proportionally — the two classes overlap too much.")
        print(f"  The dual-purpose Layer 2 de-escalation in clinical_flags.py")
        print(f"  addresses this by routing ML-flagged + clinically-stable")
        print(f"  mothers to Enhanced Monitoring instead of RHU referral.")

    return threshold, cand_df

# SECTION 2 — SENSITIVITY TABLE

def build_sensitivity_table(oof_proba: np.ndarray,
                              y_train: pd.Series,
                              selected: float) -> pd.DataFrame:
    """
    Show metrics across a range of thresholds.
    Used in the paper to justify the selection criterion.
    """
    n_total = len(y_train)
    n_lbw   = y_train.sum()
    rows    = []

    for t in np.arange(0.10, 0.75, 0.05):
        pred   = (oof_proba >= t).astype(int)
        prec   = precision_score(y_train, pred, zero_division=0)
        rec    = recall_score(y_train, pred, zero_division=0)
        f1     = f1_score(y_train, pred, zero_division=0)
        tp     = int(((pred==1)&(y_train==1)).sum())
        fn     = int(((pred==0)&(y_train==1)).sum())
        fp     = int(((pred==1)&(y_train==0)).sum())
        ref    = pred.mean() * 100
        fptr   = fp/tp if tp > 0 else float('inf')
        rows.append({
            'Threshold':  round(t, 2),
            'Recall':     round(rec,  4),
            'Precision':  round(prec, 4),
            'F1':         round(f1,   4),
            'TP':         tp,
            'FN':         fn,
            'FP':         fp,
            'FP_per_TP':  round(fptr, 1) if np.isfinite(fptr) else 'inf',
            'Referral_%': round(ref,  1),
            'Selected':   '← SELECTED' if abs(t - selected) < 0.026 else '',
        })

    return pd.DataFrame(rows)

# SECTION 3 — VISUALIZATIONS

def plot_threshold_analysis(oof_proba: np.ndarray,
                              y_train: pd.Series,
                              cand_df: pd.DataFrame,
                              threshold: float) -> None:
    """
    Three-panel threshold analysis figure:
      Panel 1: Precision-Recall curve with selected threshold marked
      Panel 2: Recall and F1 vs. Threshold
      Panel 3: FP and TP count vs. Threshold (FP problem visualization)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Threshold Selection Analysis — OOF Probabilities Only\n'
        f'Selected threshold = {threshold:.4f} '
        f'(recall ≥ {MIN_RECALL_FLOOR}, most conservative meeting floor)',
        fontsize=12, fontweight='bold'
    )

    #  Panel 1: PR Curve 
    prec_c, rec_c, thr_c = precision_recall_curve(y_train, oof_proba)
    ax = axes[0]
    ax.plot(rec_c, prec_c, '#4472C4', lw=2)
    sel = cand_df.iloc[(cand_df['threshold'] - threshold).abs().argsort()[:1]]
    if len(sel) > 0:
        ax.scatter(sel['recall'].values[0], sel['precision'].values[0],
                   color='red', s=150, zorder=5,
                   label=f'Selected t={threshold:.4f}')
    ax.axhline(y_train.mean(), color='gray', linestyle=':', lw=1.5, alpha=0.7,
               label=f'Random baseline ({y_train.mean():.3f})')
    ax.axvline(MIN_RECALL_FLOOR, color='orange', linestyle='--', lw=1.5,
               alpha=0.7, label=f'Recall floor = {MIN_RECALL_FLOOR}')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=10)
    ax.set_ylabel('Precision (PPV)', fontsize=10)
    ax.set_title('Precision-Recall Curve\n(OOF probabilities only)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.35)

    # Panel 2: Recall and F1 vs Threshold 
    ax2 = axes[1]
    ax2.plot(cand_df['threshold'], cand_df['recall'], '#C00000', lw=2.5,
             linestyle='--', label='Recall')
    ax2.plot(cand_df['threshold'], cand_df['f1'],     '#4472C4', lw=2,
             label='F1')
    ax2.plot(cand_df['threshold'], cand_df['precision'], '#70AD47', lw=1.5,
             linestyle=':', label='Precision', alpha=0.8)
    ax2.axvline(threshold, color='red', linestyle='--', lw=1.5,
                label=f'Selected t={threshold:.4f}')
    ax2.axhline(MIN_RECALL_FLOOR, color='orange', linestyle=':', lw=1.2,
                alpha=0.7, label=f'Recall floor={MIN_RECALL_FLOOR}')
    ax2.set_xlabel('Threshold', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Recall, F1, Precision vs. Threshold\n'
                  '(Recall prioritized — FN costs more than FP)',
                  fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(linestyle='--', alpha=0.35)

    #  Panel 3: FP vs TP tradeoff 
    ax3 = axes[2]
    n_lbw = y_train.sum()
    actual_roc_auc = roc_auc_score(y_train, oof_proba)
    
    cand_df['tp_calc'] = cand_df['recall'] * n_lbw
    cand_df['fp_calc'] = np.where(cand_df['precision'] > 0, 
                                 (cand_df['tp_calc'] / cand_df['precision']) - cand_df['tp_calc'], 
                                 0)

    ax3.plot(cand_df['threshold'], cand_df['fp_calc'], '#C00000', lw=2.5, label='FP (unnecessary referrals)')
    ax3.plot(cand_df['threshold'], cand_df['tp_calc'], '#4472C4', lw=2.5, label='TP (LBW correctly caught)')
    ax3.axvline(threshold, color='red', linestyle='--', lw=1.5,
                label=f'Selected t={threshold:.4f}')
    ax3.set_xlabel('Threshold', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('FP vs TP Trade-off vs. Threshold\n'
                  'Shows why raising threshold alone cannot fix high FP rate',
                  fontweight='bold', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(linestyle='--', alpha=0.35)
    
    ax3.text(0.6, 0.7,
             f"At AUC={actual_roc_auc:.3f}:\nCutting FP\nalso cuts TP\nat equal rate",
             transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'threshold_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] threshold_analysis.png")

# MAIN

def run_threshold_validation() -> float:
    print("=" * 70)
    print("STAGE 5: THRESHOLD SELECTION AND VALIDATION")
    print("=" * 70)
    print(f"\n  ⚠  Threshold selected on OOF probabilities ONLY.")
    print(f"  ⚠  Test and unseen data are NEVER used here.")

    pkg       = joblib.load(OOF_PATH)
    oof_proba = pkg['oof_proba']
    y_train   = pkg['y_train']

    oof_roc  = roc_auc_score(y_train, oof_proba)
    oof_pr   = average_precision_score(y_train, oof_proba)
    rand_pr  = y_train.mean()

    print(f"\n  [OOF] ROC-AUC : {oof_roc:.4f}")
    print(f"  [OOF] PR-AUC  : {oof_pr:.4f} (random baseline = {rand_pr:.4f})")
    print(f"  [OOF] Lift    : +{oof_pr - rand_pr:.4f} above random")
    print(f"  [OOF] LBW cases in training: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # Select threshold
    threshold, cand_df = select_threshold(oof_proba, y_train)

    # Sensitivity table
    sens_table = build_sensitivity_table(oof_proba, y_train, threshold)
    print(f"\n  Threshold Sensitivity (OOF):")
    print(f"  {'Threshold':>10} {'Recall':>8} {'Prec':>7} {'F1':>7} "
          f"{'TP':>5} {'FN':>5} {'FP':>5} {'FP/TP':>7} {'Ref%':>6}")
    print(f"  {'-'*70}")
    for _, row in sens_table.iterrows():
        marker = row['Selected'] if row['Selected'] else ''
        print(f"  {row['Threshold']:>10.2f} {row['Recall']:>8.4f} "
              f"{row['Precision']:>7.4f} {row['F1']:>7.4f} "
              f"{row['TP']:>5} {row['FN']:>5} {row['FP']:>5} "
              f"{str(row['FP_per_TP']):>7} {row['Referral_%']:>6.1f}% {marker}")

    # Visualizations
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plot_threshold_analysis(oof_proba, y_train, cand_df, threshold)

    sens_table.to_csv(
        os.path.join(OUTPUTS_DIR, 'threshold_sensitivity_table.csv'), index=False
    )
    print(f"  [SAVED] threshold_sensitivity_table.csv")

    # Save artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"\n  [SAVED] threshold.pkl = {threshold:.4f}")

    with open(FEATURES_PATH, 'w') as f:
        json.dump({'features': FEATURE_COLS, 'n_features': len(FEATURE_COLS)}, f, indent=2)
    print(f"  [SAVED] features.json ({len(FEATURE_COLS)} features)")

    print(f"\n  [SUMMARY]")
    print(f"  Selected threshold : {threshold:.4f}")
    print(f"  Applied to         : CV (OOF), Test Set, Unseen Holdout")
    print(f"  High FP rate note  : Addressed by Layer 2 de-escalation in")
    print(f"                       prototype/clinical_flags.py")
    print(f"  Next: Run 06_evaluation.py")

    return threshold


if __name__ == "__main__":
    run_threshold_validation()