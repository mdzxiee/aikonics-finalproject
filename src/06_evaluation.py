# Purpose : Full model evaluation across all three partitions (OOF / Test /
#           Unseen) using the same threshold from 05. Includes Academic
#           Layer 2 simulation with DUAL-PURPOSE fusion (escalation AND
#           de-escalation) to answer RQ3 quantitatively.
#
# Input  : artifacts/model.pkl
#          artifacts/threshold.pkl
#          artifacts/oof_probabilities.pkl
#          artifacts/split_data.pkl
#
# Output : outputs/xgb_performance_summary.csv
#          outputs/evaluation_panels.png
#          outputs/layer2_simulation.csv
#          outputs/layer2_simulation.png
#
# METRIC HIERARCHY (mandated)
#   CONTEXT    : Threshold, Referral Rate
#   PRIMARY    : Recall, PR-AUC, F1, TP, FN  ← minimize FN (missed LBW)
#   SECONDARY  : Precision, Specificity, ROC-AUC, FP, TN, NPV
#   REJECTED   : Accuracy  ← last; 87.8% trivial all-Normal baseline
#
# ACADEMIC vs PRODUCTION LAYER 2 
#   ACADEMIC (this file):
#     Statistical simulation using DOH/FNRI prevalence-based random data.
#     Strictly for quantifying RQ3 in the thesis.
#     Uses DUAL-PURPOSE fusion: escalation + de-escalation.
#   PRODUCTION (prototype/clinical_flags.py):
#     Handles real BHW measurements. Rule of 3 BP + MUAC re-measurement.
#     Same dual-purpose fusion logic but with real data.
#   DO NOT conflate these two implementations.
#
#  DUAL-PURPOSE LAYER 2 FUSION LOGIC 
#   Any L2 CRITICAL confirmed           → HIGH PRIORITY REFERRAL (escalation)
#   L2 WARNING + ML=HIGH                → HIGH PRIORITY REFERRAL (escalation)
#   L2 WARNING + ML=LOW/MEDIUM          → ELEVATED MONITORING
#   ML ≥ threshold + NO L2 flags        → ELEVATED MONITORING (de-escalation)
#   ML < threshold + NO L2 flags        → ROUTINE MONITORING
#
# Connects to: 07_shap_analysis.py
# ===========================================================================

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OOF_PATH,
    ARTIFACTS_DIR, OUTPUTS_DIR
)

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, average_precision_score,
)

# SECTION 1 — METRIC COMPUTATION

def compute_all_metrics(name: str, proba: np.ndarray,
                         y_true: pd.Series, threshold: float) -> dict:
    """
    Compute the full metric set for one partition.
    Metric ordering follows the mandated clinical hierarchy.

    PR-AUC is preferred over ROC-AUC as the primary discrimination metric
    because ROC-AUC is optimistic under severe class imbalance — the large
    TN pool inflates it even for weak minority-class detection.
    PR-AUC focuses exclusively on the LBW minority class performance.
    """
    pred    = (proba >= threshold).astype(int)
    cm      = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    recall  = recall_score(y_true, pred, zero_division=0)
    pr_auc  = average_precision_score(y_true, proba)
    f1      = f1_score(y_true, pred, zero_division=0)
    prec    = precision_score(y_true, pred, zero_division=0)
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, proba)
    npv     = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc     = accuracy_score(y_true, pred)

    return {
        'Partition':         name,
        # CONTEXT
        'Threshold':         round(float(threshold), 4),
        'Referral_Rate_%':   round(pred.mean() * 100, 1),
        # PRIMARY — Clinical Safety
        'Recall_(TPR)':      round(recall,  4),
        'PR_AUC':            round(pr_auc,  4),
        'F1_Score':          round(f1,      4),
        'TP':                int(tp),
        'FN_(Minimize!)':    int(fn),
        # SECONDARY — Logistical Balance
        'Precision_(PPV)':   round(prec,    4),
        'Specificity_(TNR)': round(spec,    4),
        'ROC_AUC':           round(roc_auc, 4),
        'FP':                int(fp),
        'TN':                int(tn),
        'NPV':               round(npv,     4),
        # REJECTED BASELINE
        'Accuracy_(last)':   round(acc,     4),
    }

# SECTION 2 — METRIC HIERARCHY TABLE

def print_metric_table(results: list) -> pd.DataFrame:
    """Print a professionally aligned metric table separated by clinical priority."""
    df = pd.DataFrame(results).set_index('Partition').T
    
    # Define the hierarchy groups
    groups = {
        'CONTEXT': ['Threshold', 'Referral_Rate_%'],
        'PRIMARY: CLINICAL SAFETY (↓ FN)': ['Recall_(TPR)', 'PR_AUC', 'F1_Score', 'TP', 'FN_(Minimize!)'],
        'SECONDARY: LOGISTICAL BALANCE': ['Precision_(PPV)', 'Specificity_(TNR)', 'ROC_AUC', 'FP', 'TN', 'NPV'],
        'REJECTED BASELINE': ['Accuracy_(last)']
    }

    # Format the table dynamically based on column names
    col_names = list(df.columns)
    metric_width = 25
    col_widths = [max(len(str(col)), 12) for col in col_names]
    
    header = f"  {'Metric'.ljust(metric_width)}" + "".join([f"{col.rjust(w)}  " for col, w in zip(col_names, col_widths)])
    
    # Print Header
    print(f"\n  {'═'*85}")
    print(f"  EVALUATION RESULTS — METRIC HIERARCHY")
    print(f"  Threshold: {results[0]['Threshold']} | Same threshold applied to all 3 partitions")
    print(f"  {'═'*85}")
    print(header)

    # Print Data Group by Group
    for group_name, metrics in groups.items():
        print(f"  {'─'*85}")
        print(f"  {group_name}")
        print(f"  {'─'*85}")
        
        for metric in metrics:
            if metric in df.index:
                row_str = f"  {metric.ljust(metric_width)}"
                for col, w in zip(col_names, col_widths):
                    val = df.loc[metric, col]
                    row_str += f"{str(val).rjust(w)}  "
                print(row_str)

    # Footer Math
    r0    = results[0]
    total = r0['TP'] + r0['FN_(Minimize!)'] + r0['FP'] + r0['TN']
    triv  = round(r0['TN'] / total * 100, 1) if total > 0 else 0.0

    print(f"  {'═'*85}")
    print(f"\n  NOTE: 'Accuracy_(last)' is placed last intentionally.")
    print(f"  Trivial all-Normal baseline = {triv}% — Accuracy is not the primary metric.")
    
    # Return the pure, unmodified DataFrame
    return df

# SECTION 3 — GENERALIZATION CHECKS

def check_generalization(results: list) -> None:
    """Explicit pass/fail generalization gap checks between partitions."""
    by_name = {r['Partition']: r for r in results}
    pairs   = [
        ('10-Fold CV (OOF)', 'Test Set',       0.10, 0.15),
        ('10-Fold CV (OOF)', 'Unseen Holdout', 0.15, 0.20),
        ('Test Set',          'Unseen Holdout', 0.10, 0.15),
    ]
    print(f"\n  {'═'*72}")
    print(f"  GENERALIZATION GAP ANALYSIS")
    print(f"  {'═'*72}")
    
    for a, b, auc_tol, rec_tol in pairs:
        if a not in by_name or b not in by_name:
            continue
            
        auc_gap   = abs(by_name[a]['ROC_AUC']      - by_name[b]['ROC_AUC'])
        prauc_gap = abs(by_name[a]['PR_AUC']       - by_name[b]['PR_AUC'])
        rec_gap   = abs(by_name[a]['Recall_(TPR)'] - by_name[b]['Recall_(TPR)'])
        f1_gap    = abs(by_name[a]['F1_Score']     - by_name[b]['F1_Score'])
        
        print(f"\n  {a}  →  {b}")
        print(f"    ROC-AUC gap : {auc_gap:.4f}  "
              f"{'✓ PASS' if auc_gap <= auc_tol else f'⚠  FAIL (>{auc_tol})'}")
              
        print(f"    PR-AUC gap  : {prauc_gap:.4f}  "
              f"{'✓ PASS' if prauc_gap <= auc_tol else f'⚠  FAIL (>{auc_tol})'}")
              
        print(f"    Recall gap  : {rec_gap:.4f}  "
              f"{'✓ PASS' if rec_gap <= rec_tol else f'⚠  FAIL (>{rec_tol})'}")
        print(f"    F1 gap      : {f1_gap:.4f}")

    print(f"\n  CONTEXT: A modest gap is expected at this sample size (~1,760 rows).")
    print(f"  The 3-partition sealed evaluation is the primary overfitting guard.")

# SECTION 4 — ACADEMIC LAYER 2 SIMULATION (RQ3) — DUAL-PURPOSE

def simulate_layer2_academic_rq3(proba_unseen: np.ndarray,
                                   y_unseen: pd.Series,
                                   threshold: float,
                                   random_seed: int = 42) -> tuple:
    """
    =========================================================================
    ACADEMIC LAYER 2 SIMULATION — FOR THESIS DEFENSE (RQ3) ONLY
    =========================================================================

    WHAT THIS IS:
      A statistical simulation that answers RQ3:
      "How does the Decision-Level Fusion framework influence the final
       LBW risk classification in alignment with DOH/WHO thresholds?"

      NDHS contains no real BP or MUAC measurements, so we simulate
      clinically realistic readings using DOH/FNRI prevalence-based
      distributions. Applied to X_unseen (sealed partition).

    WHAT THIS IS NOT:
      This is NOT the Production Layer 2 (prototype/clinical_flags.py).
      The production system handles real BHW-entered measurements with
      UI-enforced Rule of 3 and MUAC re-measurement. Do not conflate.

    DUAL-PURPOSE FUSION (updated logic):
      This simulation implements BOTH escalation AND de-escalation.

      ESCALATION PATH:
        If confirmed clinical danger sign → HIGH PRIORITY REFERRAL
        regardless of ML tier.

      DE-ESCALATION PATH (NEW):
        If ML flags (prob ≥ threshold) BUT no clinical danger signs
        confirmed → ELEVATED MONITORING (not RHU referral).
        This addresses the 39%+ referral rate from ML alone.

    SIMULATION PARAMETERS (calibrated to DOH/FNRI 2022 data):
      SBP ~ Normal(115, 20) clipped [88, 185]
      DBP ~ Normal(74,  13) clipped [55, 115]
      BP Rule of 3: ~8% confirmed hypertension after 3 readings
      MUAC ~ Normal(25.8, 2.6) clipped [19, 34]
      MUAC confirmed < 23.5: ~17% after re-measurement
    =========================================================================
    """
    np.random.seed(random_seed)
    n = len(proba_unseen)

    # STEP 1: Layer 1 ML baseline 
    pred_l1  = (proba_unseen >= threshold).astype(int)
    ml_tiers = []
    for p in proba_unseen:
        if p >= threshold:
            ml_tiers.append('HIGH')
        elif p >= (threshold / 2.0):
            ml_tiers.append('MEDIUM')
        else:
            ml_tiers.append('LOW')
    ml_tiers = np.array(ml_tiers)

    print(f"\n  [L2-SIM] STEP 1 — ML Baseline")
    for t in ['LOW', 'MEDIUM', 'HIGH']:
        c = (ml_tiers == t).sum()
        print(f"  [L2-SIM]   {t:<8}: {c:>4} ({c/n*100:.1f}%)")
    print(f"  [L2-SIM]   ML flagged: {pred_l1.sum()} ({pred_l1.mean()*100:.1f}%)")

    # STEP 2: BP Rule of 3
    sbp_r1 = np.clip(np.random.normal(115, 20, n), 88, 185)
    dbp_r1 = np.clip(np.random.normal(74,  13, n), 55, 115)
    bp_f1  = (sbp_r1 >= 140) | (dbp_r1 >= 90)

    sbp_r2 = sbp_r1 - np.random.uniform(3, 10, n)
    dbp_r2 = dbp_r1 - np.random.uniform(2,  6, n)
    bp_f2  = bp_f1 & ((sbp_r2 >= 140) | (dbp_r2 >= 90))

    sbp_r3 = sbp_r2 - np.random.uniform(2, 6, n)
    dbp_r3 = dbp_r2 - np.random.uniform(1, 4, n)
    bp_confirmed = bp_f2 & ((sbp_r3 >= 140) | (dbp_r3 >= 90))

    deesc_bp = int(bp_f1.sum() - bp_confirmed.sum())
    print(f"\n  [L2-SIM] STEP 2 — BP Rule of 3")
    print(f"  [L2-SIM]   R1 flagged: {bp_f1.sum()} ({bp_f1.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   R3 confirmed: {bp_confirmed.sum()} "
          f"({bp_confirmed.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   De-escalated by Rule of 3: {deesc_bp} "
          f"({deesc_bp/n*100:.1f}%)")

    # STEP 3: MUAC Verification 
    muac_r1   = np.clip(np.random.normal(25.8, 2.6, n), 19.0, 34.0)
    muac_f1   = muac_r1 < 23.5
    muac_r2   = muac_r1 + np.random.uniform(-0.5, 0.5, n)
    muac_conf = muac_f1 & (muac_r2 < 23.5)
    cleared   = int(muac_f1.sum() - muac_conf.sum())

    print(f"\n  [L2-SIM] STEP 3 — MUAC Verification")
    print(f"  [L2-SIM]   R1 flagged: {muac_f1.sum()} ({muac_f1.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   Confirmed: {muac_conf.sum()} "
          f"({muac_conf.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   Cleared by re-measure: {cleared} "
          f"({cleared/n*100:.1f}%)")

    # STEP 4: Dual-Purpose Decision Fusion
    any_critical = bp_confirmed | muac_conf
    final_levels = []
    for i, (ml_tier, ml_flag, crit) in enumerate(
            zip(ml_tiers, pred_l1, any_critical)):
        if crit:
            final_levels.append('HIGH PRIORITY REFERRAL (L2 Escalated)')
        elif ml_flag and not crit:
            # ML flagged but no clinical danger → DE-ESCALATION
            final_levels.append('ELEVATED MONITORING (ML flagged, L2 De-escalated)')
        elif ml_tier == 'HIGH' and not ml_flag:
            final_levels.append('ELEVATED MONITORING (ML High-tier, no L2 flag)')
        else:
            final_levels.append(f'ROUTINE MONITORING ({ml_tier} ML tier)')

    # Count outcomes
    total_l2_overrides  = int(any_critical.sum())
    escalated_from_low  = int((any_critical & (ml_tiers != 'HIGH')).sum())
    ml_flagged          = int(pred_l1.sum())
    
    # deescalated = ML flagged but no L2 danger → rerouted to monitoring
    deesc_count = int(np.sum(
        pred_l1.astype(bool) & ~any_critical.astype(bool)
    ))
    
    pct_altered = total_l2_overrides / n * 100
    pct_deesc   = deesc_count / n * 100

    #  RQ3 Answer ─
    box_w = 78  # Total width of our box
    inner = box_w - 4  # Inner text width

    def print_box_line(text):
        """Helper to perfectly align the right border of the box."""
        print(f"  │ {text.ljust(inner)} │")

    print(f"\n  {'═'*box_w}")
    print(f"  RQ3 — DECISION-LEVEL FUSION EFFECT (ACADEMIC SIMULATION)")
    print(f"  {'═'*box_w}")
    print(f"\n  RQ3: 'How does Decision-Level Fusion influence final LBW risk")
    print(f"        classification in alignment with DOH/WHO thresholds?'\n")

    print(f"  ┌{'─' * (box_w - 2)}┐")
    
    print_box_line("ESCALATION PATHWAY")
    print_box_line(f"  • {total_l2_overrides} of {n} ({pct_altered:.1f}%) cases received Layer 2 clinical override")
    print_box_line("    → HIGH PRIORITY REFERRAL.")
    print_box_line(f"  • {escalated_from_low} of {n} ({escalated_from_low/n*100:.1f}%) escalated from LOW/MEDIUM ML tier")
    print_box_line("    (would have been completely missed by ML alone).")
    
    print(f"  ├{'─' * (box_w - 2)}┤")
    
    print_box_line("DE-ESCALATION PATHWAY")
    print_box_line(f"  • {deesc_count} of {n} ({pct_deesc:.1f}%) ML-flagged cases had NO confirmed")
    print_box_line("    clinical danger → rerouted to ELEVATED MONITORING.")
    print_box_line("  • These are converted from unnecessary RHU referrals")
    print_box_line("    to appropriate enhanced home monitoring.")
    
    print(f"  ├{'─' * (box_w - 2)}┤")
    
    print_box_line("CLINICAL SAFETY NET & RESOURCE PROTECTION")
    print_box_line(f"  • BP RULE OF 3 PREVENTED {deesc_bp} unnecessary escalations")
    print_box_line(f"    ({deesc_bp/n*100:.1f}%) — white-coat effect eliminated by protocol.")
    print_box_line(f"  • MUAC VERIFICATION CLEARED {cleared} borderline readings")
    print_box_line(f"    ({cleared/n*100:.1f}%) — measurement error corrected by re-measure.")
    
    print(f"  └{'─' * (box_w - 2)}┘")
    print(f"\n  (Simulation — DOH/FNRI 2022 prevalence-based | seed={random_seed})")

    # Build output DataFrame
    sim_df = pd.DataFrame({
        'ml_probability':   proba_unseen,
        'ml_tier':          ml_tiers,
        'ml_pred':          pred_l1,
        'sbp_r1': sbp_r1.round(1), 'dbp_r1': dbp_r1.round(1),
        'sbp_r3': sbp_r3.round(1), 'dbp_r3': dbp_r3.round(1),
        'bp_r1_flag': bp_f1, 'bp_confirmed': bp_confirmed,
        'muac_r1': muac_r1.round(1), 'muac_r2': muac_r2.round(1),
        'muac_confirmed': muac_conf,
        'any_l2_critical':  any_critical,
        'final_level':      final_levels,
        'actual_lbw':       np.array(y_unseen),
    })

    stats = {
        'n_total':             n,
        'total_l2_escalated':  total_l2_overrides,
        'pct_escalated':       round(pct_altered, 2),
        'escalated_from_low':  escalated_from_low,
        'deescalated_count':   deesc_count,
        'pct_deescalated':     round(pct_deesc, 2),
        'bp_r1_flags':         int(bp_f1.sum()),
        'bp_r2_flags':         int(bp_f2.sum()),
        'bp_confirmed':        int(bp_confirmed.sum()),
        'bp_deescalated':      deesc_bp,
        'muac_r1_flags':       int(muac_f1.sum()),
        'muac_confirmed':      int(muac_conf.sum()),
        'muac_cleared':        cleared,
    }
    return sim_df, stats


# SECTION 5 — VISUALIZATIONS

def plot_evaluation_panels(results: list, oof_proba, proba_test, proba_unseen,
                            y_train, y_test, y_unseen, threshold: float) -> None:
    """Confusion matrices + ROC curves for all three partitions."""
    data_map = {
        '10-Fold CV (OOF)': (oof_proba,    y_train),
        'Test Set':          (proba_test,   y_test),
        'Unseen Holdout':    (proba_unseen, y_unseen),
    }
    colors  = ['#4472C4', '#ED7D31', '#70AD47']
    n_sets  = len(results)
    fig     = plt.figure(figsize=(7 * n_sets, 11))
    gs      = gridspec.GridSpec(2, n_sets, hspace=0.4, wspace=0.38)

    fig.suptitle(
        f'AIKONIC XGBoost — Evaluation Across All Three Partitions\n'
        f'Threshold = {threshold:.4f}  '
        f'(selected on OOF data; applied unchanged to all sets)',
        fontsize=12, fontweight='bold', y=1.01
    )

    for i, (res, color) in enumerate(zip(results, colors)):
        name      = res['Partition']
        proba, y  = data_map[name]
        pred      = (proba >= threshold).astype(int)
        cm        = confusion_matrix(y, pred)
        rec       = res['Recall_(TPR)']
        pr_a      = res['PR_AUC']
        roc_a     = res['ROC_AUC']
        spec      = res['Specificity_(TNR)']
        ref       = res['Referral_Rate_%']
        fn_c      = res['FN_(Minimize!)']

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[0, i])
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'LBW']).plot(
            ax=ax_cm, cmap='Blues', colorbar=False
        )
        ax_cm.set_title(
            f'{name}\nRecall={rec:.3f}  PR-AUC={pr_a:.3f}\n'
            f'FN={fn_c} (missed LBW)  Referral={ref}%',
            fontweight='bold', fontsize=8.5
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc       = fig.add_subplot(gs[1, i])
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f'ROC-AUC={roc_a:.4f}\nPR-AUC={pr_a:.4f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
        ax_roc.scatter([1 - spec], [rec], color='red', s=110, zorder=5,
                       label=f'Operating pt (t={threshold:.3f})\nRecall={rec:.3f}')
        ax_roc.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=8.5)
        ax_roc.set_ylabel('True Positive Rate (Recall)', fontsize=8.5)
        ax_roc.set_title(f'ROC — {name}', fontweight='bold', fontsize=9)
        ax_roc.legend(fontsize=7.5, loc='lower right')
        ax_roc.grid(linestyle='--', alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'evaluation_panels.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [SAVED] evaluation_panels.png")


def plot_layer2_simulation(sim_df: pd.DataFrame, stats: dict) -> None:
    """
    Two-panel RQ3 figure:
      Left : Classification before vs after fusion (escalation + de-escalation)
      Right: Clinical triage protocol funnel
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'RQ3 — Dual-Purpose Decision-Level Fusion Effect\n'
        '(Academic Simulation — DOH/FNRI 2022 Prevalence-Based | '
        'Production Layer 2 in prototype/clinical_flags.py)',
        fontsize=11, fontweight='bold'
    )
    n = stats['n_total']

    # Left: Before vs After 
    ax    = axes[0]
    ml_cts = sim_df['ml_tier'].value_counts()
    tier_order  = ['LOW', 'MEDIUM', 'HIGH']
    tier_colors = {'LOW': '#70AD47', 'MEDIUM': '#FFC000', 'HIGH': '#C00000'}

    # Before (ML only)
    bottom = 0
    for t in tier_order:
        v = ml_cts.get(t, 0)
        ax.bar(0, v, color=tier_colors[t], width=0.5,
               bottom=bottom, edgecolor='white', label=t)
        bottom += v

    # After (fusion — subtract escalated, add de-escalated note)
    bottom2 = 0
    for t in tier_order:
        escl = sim_df[(sim_df['ml_tier'] == t) & sim_df['any_l2_critical']].shape[0]
        v_after = max(ml_cts.get(t, 0) - escl, 0)
        ax.bar(1, v_after, color=tier_colors[t], width=0.5,
               bottom=bottom2, edgecolor='white', alpha=0.85)
        bottom2 += v_after

    # Add escalated block + de-escalated annotation
    ax.bar(1, stats['total_l2_escalated'], color='#7B0000', width=0.5,
           bottom=bottom2, edgecolor='white',
           label=f'L2 Escalated → REFERRAL ({stats["pct_escalated"]}%)')
    # Annotate de-escalation
    ax.annotate(
        f'De-escalated: {stats["deescalated_count"]}\n'
        f'({stats["pct_deescalated"]}%) ML-flagged\n→ Monitoring only',
        xy=(1, bottom2 / 2), xytext=(1.35, bottom2 / 2),
        fontsize=8, color='#003366',
        arrowprops=dict(arrowstyle='->', color='#003366'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4FD', alpha=0.8)
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Layer 1\n(ML Only)', 'Final\n(After Fusion)'], fontsize=10)
    ax.set_ylabel('Number of Cases')
    ax.set_title(
        f'Classification Before vs. After Layer 2\n'
        f'Escalation: {stats["pct_escalated"]}% | De-escalation: {stats["pct_deescalated"]}%',
        fontweight='bold', fontsize=10
    )
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # Right: Protocol Funnel 
    ax2 = axes[1]
    funnel_data = [
        (f'Total Simulated (n={n})',               n,                            '#4472C4'),
        (f'BP R1 ≥140/90 → {stats["bp_r1_flags"]} ({stats["bp_r1_flags"]/n*100:.1f}%)',
                                                   stats['bp_r1_flags'],         '#ED7D31'),
        (f'BP R3 Confirmed → {stats["bp_confirmed"]} ({stats["bp_confirmed"]/n*100:.1f}%)',
                                                   stats['bp_confirmed'],         '#C00000'),
        (f'MUAC R1 <23.5 → {stats["muac_r1_flags"]} ({stats["muac_r1_flags"]/n*100:.1f}%)',
                                                   stats['muac_r1_flags'],        '#FFC000'),
        (f'MUAC Confirmed → {stats["muac_confirmed"]} ({stats["muac_confirmed"]/n*100:.1f}%)',
                                                   stats['muac_confirmed'],       '#C00000'),
        (f'Total L2 Escalated → {stats["total_l2_escalated"]} ({stats["pct_escalated"]}%)',
                                                   stats['total_l2_escalated'],   '#7B0000'),
        (f'ML-only De-escalated → {stats["deescalated_count"]} ({stats["pct_deescalated"]}%)',
                                                   stats['deescalated_count'],    '#003366'),
    ]
    labels = [d[0] for d in funnel_data]
    values = [d[1] for d in funnel_data]
    fcolors= [d[2] for d in funnel_data]
    y_pos  = np.arange(len(labels))
    ax2.barh(y_pos, values, color=fcolors, edgecolor='white', height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of Cases')
    ax2.set_title('Clinical Triage Protocol Funnel\n(Rule of 3 + MUAC Verification + De-escalation)',
                  fontweight='bold', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.35)
    for yp, val in zip(y_pos, values):
        ax2.text(val + 0.3, yp, f'{val}', va='center', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'layer2_simulation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] layer2_simulation.png")

# MAIN

def run_evaluation() -> None:
    print("=" * 72)
    print("STAGE 6: FULL MODEL EVALUATION + DUAL-PURPOSE LAYER 2 SIM (RQ3)")
    print("=" * 72)

    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    oof_pkg   = joblib.load(OOF_PATH)
    splits    = joblib.load(os.path.join(ARTIFACTS_DIR, "split_data.pkl"))

    oof_proba  = oof_pkg['oof_proba']
    y_train    = oof_pkg['y_train']
    X_test     = splits['X_test'];  y_test   = splits['y_test']
    X_unseen   = splits['X_unseen'];y_unseen = splits['y_unseen']

    proba_test   = model.predict_proba(X_test)[:,   1]
    proba_unseen = model.predict_proba(X_unseen)[:, 1]

    print(f"\n  Threshold       : {threshold:.4f}  (from 05, OOF-based, unchanged)")
    print(f"  OOF  LBW rate   : {y_train.mean()*100:.1f}%  ({y_train.sum()} cases)")
    print(f"  Test LBW rate   : {y_test.mean()*100:.1f}%  ({y_test.sum()} cases)")
    print(f"  Unseen LBW rate : {y_unseen.mean()*100:.1f}%  ({y_unseen.sum()} cases)")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Compute metrics for all 3 partitions
    results = [
        compute_all_metrics('10-Fold CV (OOF)', oof_proba,    y_train,  threshold),
        compute_all_metrics('Test Set',          proba_test,   y_test,   threshold),
        compute_all_metrics('Unseen Holdout',    proba_unseen, y_unseen, threshold),
    ]

    # Print hierarchical table
    print_metric_table(results)

    # Generalization checks
    check_generalization(results)

    # Save CSV
    flat = pd.DataFrame(results)
    flat.to_csv(os.path.join(OUTPUTS_DIR, 'xgb_performance_summary.csv'), index=False)
    print(f"\n  [SAVED] xgb_performance_summary.csv")

    # Evaluation panels
    plot_evaluation_panels(results, oof_proba, proba_test, proba_unseen,
                            y_train, y_test, y_unseen, threshold)

    # Academic Layer 2 simulation (RQ3) — dual-purpose
    print(f"\n  {'═'*72}")
    print(f"  ACADEMIC LAYER 2 SIMULATION — RQ3 (dual-purpose fusion)")
    print(f"  NOT the Production Layer 2 (prototype/clinical_flags.py).")
    print(f"  {'═'*72}")

    sim_df, stats = simulate_layer2_academic_rq3(
        proba_unseen, y_unseen, threshold, random_seed=42
    )
    sim_df.to_csv(os.path.join(OUTPUTS_DIR, 'layer2_simulation.csv'), index=False)
    print(f"\n  [SAVED] layer2_simulation.csv")
    plot_layer2_simulation(sim_df, stats)

    print(f"\n  {'═'*72}")
    print(f"  STAGE 6 COMPLETE")
    print(f"  Outputs : xgb_performance_summary.csv | evaluation_panels.png")
    print(f"            layer2_simulation.csv        | layer2_simulation.png")
    print(f"  Next    : python src/07_shap_analysis.py")
    print(f"  {'═'*72}")


if __name__ == "__main__":
    run_evaluation()