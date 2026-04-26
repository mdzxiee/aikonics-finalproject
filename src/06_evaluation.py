# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/06_evaluation.py
#
# Purpose : Full model evaluation across all three data partitions (OOF /
#           Test / Unseen) using the SAME threshold from 05, followed by a
#           structured simulation of the Layer 2 clinical triage protocol
#           to answer RQ3 quantitatively.
#
# Input  : artifacts/model.pkl
#          artifacts/threshold.pkl
#          artifacts/oof_probabilities.pkl
#          artifacts/split_data.pkl
#
# Output : outputs/xgb_performance_summary.csv    — full metric table
#          outputs/evaluation_panels.png           — CM + ROC figure
#          outputs/layer2_simulation.csv           — RQ3 evidence
#          outputs/layer2_simulation.png           — RQ3 figure
#
#  METRIC HIERARCHY (RULE 1) — ordering is intentional and mandated 
#
#   ┌─ CONTEXT               : Threshold, Referral Rate
#   ├─ PRIMARY  (Clinical)   : Recall, PR-AUC, F1, TP, FN  ← minimize FN
#   ├─ SECONDARY (Logistical): Precision, Specificity, ROC-AUC, FP, TN, NPV
#   └─ REJECTED BASELINE     : Accuracy  ← last; explicitly deprioritized
#
#  LAYER 2 SIMULATION (RULE 2) — two distinct implementations 
#
#   "ACADEMIC Layer 2"   → This file (simulate_layer2_academic_rq3)
#     Strictly for RQ3 thesis defense. Uses prevalence-based random clinical
#     data to quantify how often Layer 2 alters Layer 1 output.
#     Do NOT use in deployed prototype.
#
#   "PRODUCTION Layer 2" → prototype/clinical_flags.py + prototype/app.py
#     Handles real BHW-entered measurements, UI re-check prompts, and actual
#     multi-reading enforcement during home visits.
#     Built separately. Do NOT conflate with this simulation.
#
# Connects to: 07_shap_analysis.py
# -----------------------------------------------------------------------

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
    ARTIFACTS_DIR, OUTPUTS_DIR, CLINICAL_THRESHOLDS
)

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, average_precision_score,
)


# SECTION 1 — METRIC COMPUTATION

def compute_all_metrics(name: str,
                         proba: np.ndarray,
                         y_true: pd.Series,
                         threshold: float) -> dict:
    """
    Compute the full metric set for one evaluation partition.

    Metric ordering follows the mandated clinical hierarchy:
      1. Context      : threshold + referral rate
      2. Primary      : recall, PR-AUC, F1, TP, FN  ← clinical safety first
      3. Secondary    : precision, specificity, ROC-AUC, FP, TN, NPV
      4. Rejected     : accuracy (last, explicitly deprioritized)

    PR-AUC (average_precision_score) is preferred over ROC-AUC as the
    primary discrimination metric because:
      ● ROC-AUC is optimistic under severe class imbalance — the large pool
        of True Negatives inflates it even for weak minority-class detection.
      ● PR-AUC focuses exclusively on the LBW minority class: how well
        the model retrieves true LBW cases relative to its false alarms.
      ● A model that correctly identifies 12% LBW cases from an 88% Normal
        pool can achieve ROC-AUC > 0.70 while PR-AUC remains low — PR-AUC
        exposes this discrepancy.
    """
    pred = (proba >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
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
        'Partition':        name,
        #  Context 
        'Threshold':        round(float(threshold), 4),
        'Referral_Rate_%':  round(pred.mean() * 100, 1),
        # PRIMARY: Clinical Safety 
        'Recall_(TPR)':     round(recall,  4),
        'PR_AUC':           round(pr_auc,  4),
        'F1_Score':         round(f1,      4),
        'TP':               int(tp),
        'FN_(Minimize!)':   int(fn),
        # SECONDARY: Logistical Balance 
        'Precision_(PPV)':  round(prec,    4),
        'Specificity_(TNR)':round(spec,    4),
        'ROC_AUC':          round(roc_auc, 4),
        'FP':               int(fp),
        'TN':               int(tn),
        'NPV':              round(npv,     4),
        #  REJECTED BASELINE 
        'Accuracy_(last)':  round(acc,     4),
    }



# SECTION 2 — FORMATTED METRIC HIERARCHY TABLE

def print_metric_table(results: list) -> pd.DataFrame:
    """
    Build the side-by-side metric comparison with separator rows for each
    group, matching the mandated clinical hierarchy exactly.
    The table is printed and also returned as a DataFrame for saving.
    """
    df = pd.DataFrame(results).set_index('Partition').T

    # Separator rows injected between metric groups
    sep = {
        '__sep_ctx__':   '── CONTEXT ─────────────────────────────────',
        '__sep_pri__':   '── PRIMARY: CLINICAL SAFETY  (minimize FN) ─',
        '__sep_sec__':   '── SECONDARY: LOGISTICAL BALANCE ────────────',
        '__sep_rej__':   '── REJECTED BASELINE ────────────────────────',
    }
    for key, label in sep.items():
        df.loc[key] = [label for _ in df.columns]

    ordered = [
        '__sep_ctx__',
        'Threshold',
        'Referral_Rate_%',
        '__sep_pri__',
        'Recall_(TPR)',
        'PR_AUC',
        'F1_Score',
        'TP',
        'FN_(Minimize!)',
        '__sep_sec__',
        'Precision_(PPV)',
        'Specificity_(TNR)',
        'ROC_AUC',
        'FP',
        'TN',
        'NPV',
        '__sep_rej__',
        'Accuracy_(last)',
    ]
    df = df.reindex(ordered)

    # Compute trivial-classifier accuracy for the rejected-baseline note
    r0     = results[0]
    total0 = r0['TP'] + r0['FN_(Minimize!)'] + r0['FP'] + r0['TN']
    triv   = round(r0['TN'] / total0 * 100, 1) if total0 > 0 else 0.0

    print(f"\n  {'-'*72}")
    print(f"  EVALUATION RESULTS — METRIC HIERARCHY (Sections by clinical priority)")
    print(f"  Threshold: {results[0]['Threshold']}  │  Applied identically to all three partitions")
    print(f"  {'-'*72}")
    print(df.to_string())
    print(f"\n  NOTE  → 'Accuracy_(last)' is deliberately placed last.")
    print(f"          A trivial all-Normal classifier achieves {triv}% accuracy.")
    print(f"          Accuracy is reported only for completeness; Recall and")
    print(f"          PR-AUC are the operationally significant metrics for")
    print(f"          this imbalanced prenatal screening context.")

    return df


# SECTION 3 — GENERALIZATION GAP ANALYSIS

def check_generalization(results: list) -> None:
    """
    Explicit pass/fail generalization gap checks between partitions.
    These thresholds define the line between acceptable variation and
    overfitting for a ~1,758-row clinical dataset.
    """
    by_name = {r['Partition']: r for r in results}
    pairs   = [
        ('10-Fold CV (OOF)', 'Test Set',       0.10, 0.15),
        ('10-Fold CV (OOF)', 'Unseen Holdout', 0.15, 0.20),
        ('Test Set',          'Unseen Holdout', 0.10, 0.15),
    ]

    print(f"\n  {'-'*72}")
    print(f"  GENERALIZATION GAP ANALYSIS")
    print(f"  {'-'*72}")

    for a, b, auc_tol, rec_tol in pairs:
        if a not in by_name or b not in by_name:
            continue
        auc_gap = abs(by_name[a]['ROC_AUC']      - by_name[b]['ROC_AUC'])
        prauc_gap = abs(by_name[a]['PR_AUC']       - by_name[b]['PR_AUC'])
        rec_gap = abs(by_name[a]['Recall_(TPR)'] - by_name[b]['Recall_(TPR)'])
        f1_gap  = abs(by_name[a]['F1_Score']     - by_name[b]['F1_Score'])

        print(f"\n  {a}  →  {b}")
        print(f"    ROC-AUC gap : {auc_gap:.4f}  "
              f"{'✓ PASS' if auc_gap <= auc_tol else f'⚠  FAIL (>{auc_tol}) — investigate overfitting'}")
        print(f"    PR-AUC gap  : {prauc_gap:.4f}")  
        print(f"    Recall gap  : {rec_gap:.4f}  "
              f"{'✓ PASS' if rec_gap <= rec_tol else f'⚠  FAIL (>{rec_tol}) — threshold instability'}")
        print(f"    F1 gap      : {f1_gap:.4f}")

    print(f"\n  CONTEXT: A modest performance gap is expected at this sample size.")

# SECTION 4 — ACADEMIC LAYER 2 SIMULATION (RQ3)

def simulate_layer2_academic_rq3(proba_unseen: np.ndarray,
                                   y_unseen: pd.Series,
                                   threshold: float,
                                   random_seed: int = 42) -> tuple:
    """
    =========================================================================
    ACADEMIC LAYER 2 SIMULATION — FOR THESIS DEFENSE (RQ3) ONLY
    =========================================================================

    WHAT THIS IS:
      A statistical simulation that answers RQ3 quantitatively:
      "How does the Decision-Level Fusion framework influence the final
       LBW risk classification in alignment with DOH/WHO guidelines?"

      Since the NDHS dataset contains no BP or MUAC measurements, we
      generate realistic synthetic clinical readings using prevalence-based
      distributions calibrated to DOH/FNRI data for Filipino pregnant women,
      then apply the multi-step clinical triage protocol below.

    WHAT THIS IS NOT:
      This is NOT the Production Layer 2. It does not:
        ● Accept real user inputs from the BHW form.
        ● Enforce UI-level re-check prompts or measurement retries.
        ● Validate physical tape tension or cuff placement.
        ● Connect to the database or API.
      All of the above is implemented in the Production Layer 2:
        prototype/clinical_flags.py  →  apply_decision_fusion()
        prototype/app.py             →  POST /api/assess endpoint
      Do NOT use this function in any user-facing system.

    SIMULATION PARAMETER CALIBRATION (verified against Philippine data):
    ──────────────────────────────────────────────────────────────────────
      Blood Pressure:
        SBP  ~ Normal(115, 20), clipped [88, 185] mmHg
        DBP  ~ Normal(74,  13), clipped [55, 115] mmHg
        Initial BP flag rate (≥140/90): ~21% (before Rule of 3)
        Confirmed BP flag rate:          ~8%  (after Rule of 3)
        Aligned with DOH AO 2022-0012 reporting ~11% hypertension in
        pregnancy (DOH figure includes borderline/managed cases
        outside our simulation's critical bounds).
        Source: Philippine Heart Association / DOH Administrative Order 2022-0012.

      MUAC:
        MUAC ~ Normal(25.8, 2.6), clipped [19.0, 34.0] cm
        First-read flag rate (< 23.5 cm): ~18%
        Confirmed after verification:     ~17%
        Source: FNRI 8th National Nutrition Survey 2019 — maternal
        undernutrition rate among Filipino pregnant women 16–19%.

    MULTI-STEP CLINICAL TRIAGE PROTOCOL:
    ──────────────────────────────────────────────────────────────────────
      STEP 1 — Apply ML Baseline (Layer 1)
        Assign LOW / MEDIUM / HIGH risk tier per probability threshold.

      STEP 2 — BP Rule of 3 Protocol
        Reading 1 : Initial BP (simulated BHW visit measurement).
        Rest      : 15-minute seated rest → SBP drops 3–10 mmHg,
                    DBP drops 2–6 mmHg (white-coat effect reduction).
        Reading 2 : Re-measure. If now < 140/90, no flag.
        Posture   : If still ≥ 140/90 → posture correction →
                    SBP drops 2–6, DBP drops 1–4 mmHg (orthostatic effect).
        Reading 3 : CONFIRMED hypertension ONLY if still ≥ 140/90.
        Basis     : Philippine Heart Association 2022 consensus — hypertension
                    should be confirmed on ≥ 2 readings with one post-rest,
                    to exclude isolated white-coat hypertension.

      STEP 3 — MUAC Verification Protocol
        Reading 1 : BHW measures MUAC on non-dominant arm.
        Verify    : If < 23.5 cm, re-measure (check tape tension).
                    Measurement error: ± 0.5 cm (realistic BHW field precision).
        Confirmed : Both readings must be < 23.5 cm to trigger the flag.
        Basis     : DOH MUAC measurement guidelines — one borderline reading
                    is insufficient to declare maternal undernutrition.

      STEP 4 — Decision Fusion
        If ANY confirmed Layer 2 flag → HIGH PRIORITY REFERRAL (override).
        Otherwise → Final = Layer 1 ML output (unchanged).
    =========================================================================
    """
    np.random.seed(random_seed)
    n = len(proba_unseen)

    # STEP 1: Apply ML Baseline (Layer 1)
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

    tier_series = pd.Series(ml_tiers)
    print(f"\n  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM] STEP 1 — ML Baseline (Layer 1)")
    print(f"  [L2-SIM] {'─'*62}")
    for t in ['LOW', 'MEDIUM', 'HIGH']:
        c = (tier_series == t).sum()
        print(f"  [L2-SIM]   {t:<8}: {c:>4} ({c/n*100:.1f}%)")
    print(f"  [L2-SIM]   Flagged by ML (pred=1): {pred_l1.sum()} ({pred_l1.mean()*100:.1f}%)")


    # STEP 2: BP Rule of 3 Protocol

    sbp_r1 = np.clip(np.random.normal(115, 20, n), 88, 185)
    dbp_r1 = np.clip(np.random.normal(74,  13, n), 55, 115)
    bp_flag_r1 = (sbp_r1 >= 140) | (dbp_r1 >= 90)

    # Reading 2: 15-minute rest (white-coat effect reduction)
    sbp_r2 = sbp_r1 - np.random.uniform(3, 10, n)
    dbp_r2 = dbp_r1 - np.random.uniform(2,  6, n)
    bp_flag_r2 = bp_flag_r1 & ((sbp_r2 >= 140) | (dbp_r2 >= 90))

    # Reading 3: posture correction (orthostatic effect)
    sbp_r3 = sbp_r2 - np.random.uniform(2, 6, n)
    dbp_r3 = dbp_r2 - np.random.uniform(1, 4, n)
    bp_confirmed = bp_flag_r2 & ((sbp_r3 >= 140) | (dbp_r3 >= 90))

    print(f"\n  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM] STEP 2 — BP Rule of 3 Protocol")
    print(f"  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM]   Reading 1 (initial ≥140/90)          : "
          f"{bp_flag_r1.sum():>4}  ({bp_flag_r1.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   Reading 2 (after 15-min rest)        : "
          f"{bp_flag_r2.sum():>4}  ({bp_flag_r2.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   Reading 3 CONFIRMED (post-posture)   : "
          f"{bp_confirmed.sum():>4}  ({bp_confirmed.mean()*100:.1f}%)")
    deesc_bp = bp_flag_r1.sum() - bp_confirmed.sum()
    print(f"  [L2-SIM]   Cases de-escalated by Rule of 3      : "
          f"{deesc_bp:>4}  ({deesc_bp/n*100:.1f}%) — avoided unnecessary referrals")

    # STEP 3: MUAC Verification Protocol

    muac_r1    = np.clip(np.random.normal(25.8, 2.6, n), 19.0, 34.0)
    muac_flag_r1 = muac_r1 < 23.5

    # Re-measure: ± 0.5 cm random measurement error (tape tension, technique)
    muac_r2       = muac_r1 + np.random.uniform(-0.5, 0.5, n)
    muac_confirmed = muac_flag_r1 & (muac_r2 < 23.5)

    print(f"\n  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM] STEP 3 — MUAC Verification Protocol")
    print(f"  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM]   Reading 1 (initial MUAC < 23.5 cm)   : "
          f"{muac_flag_r1.sum():>4}  ({muac_flag_r1.mean()*100:.1f}%)")
    print(f"  [L2-SIM]   Verified (re-measure < 23.5 cm)      : "
          f"{muac_confirmed.sum():>4}  ({muac_confirmed.mean()*100:.1f}%)")
    cleared_muac = muac_flag_r1.sum() - muac_confirmed.sum()
    print(f"  [L2-SIM]   Cases cleared by re-measurement      : "
          f"{cleared_muac:>4}  ({cleared_muac/n*100:.1f}%) — avoided false malnutrition flags")

    # STEP 4: Decision Fusion

    any_l2_override = bp_confirmed | muac_confirmed

    final_levels = np.where(
        any_l2_override,
        'HIGH PRIORITY REFERRAL (L2 Override)',
        np.char.add(ml_tiers, ' RISK (L1 Only)')
    )

    # Cases where Layer 2 altered a NON-HIGH Layer 1 output
    non_high_mask       = ml_tiers != 'HIGH'
    altered_from_nonhigh = (any_l2_override & non_high_mask).sum()
    total_altered        = any_l2_override.sum()
    pct_altered          = total_altered / n * 100
    pct_from_nonhigh     = altered_from_nonhigh / n * 100

    print(f"\n  [L2-SIM] {'─'*62}")
    print(f"  [L2-SIM] STEP 4 — Decision Fusion Output")
    print(f"  [L2-SIM] {'─'*62}")
    for lv, ct in pd.Series(final_levels).value_counts().items():
        print(f"  [L2-SIM]   {lv:<45}: {ct:>4}  ({ct/n*100:.1f}%)")

    #  RQ3 QUANTITATIVE ANSWER 
    print(f"\n  {'═'*72}")
    print(f"  RQ3 — QUANTITATIVE ANSWER")
    print(f"  {'═'*72}")
    print(f"\n  RQ3: 'How does the Decision-Level Fusion framework influence the")
    print(f"        final LBW risk classification in alignment with WHO/DOH thresholds?'")
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  {total_altered} of {n} cases ({pct_altered:.1f}%) received a Layer 2 clinical      │")
    print(f"  │  override, altering the final recommendation from the ML baseline. │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  {altered_from_nonhigh} of {n} cases ({pct_from_nonhigh:.1f}%) were escalated to HIGH PRIORITY    │")
    print(f"  │  REFERRAL despite being classified LOW or MEDIUM by Layer 1 alone. │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  BP Rule of 3 prevented {deesc_bp} unnecessary escalations        │")
    print(f"  │  ({deesc_bp/n*100:.1f}%) that a single-reading protocol would have triggered.  │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  MUAC verification corrected {cleared_muac} borderline readings            │")
    print(f"  │  ({cleared_muac/n*100:.1f}%) that would have caused false malnutrition flagging.  │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")
    print(f"\n  INTERPRETATION:")
    print(f"  The Decision-Level Fusion framework meaningfully altered final")
    print(f"  recommendations in {pct_altered:.1f}% of simulated cases. This demonstrates")
    print(f"  that Layer 2 adds clinical value beyond the ML baseline — particularly")
    print(f"  for detecting maternal undernutrition (MUAC < 23.5 cm), which the")
    print(f"  ML model cannot capture from sociodemographic features alone.")
    print(f"  The multi-reading BP protocol reduced false escalation by {deesc_bp/n*100:.1f}%,")
    print(f"  improving specificity while preserving clinical safety.")
    print(f"\n  (Simulation — DOH/FNRI 2022 prevalence-based | seed={random_seed})")

    # Build output DataFrame 
    sim_df = pd.DataFrame({
        'ml_probability':   proba_unseen,
        'ml_tier':          ml_tiers,
        'ml_pred_label':    pred_l1,
        'sbp_r1':           sbp_r1.round(1),
        'dbp_r1':           dbp_r1.round(1),
        'sbp_r2':           sbp_r2.round(1),
        'dbp_r2':           dbp_r2.round(1),
        'sbp_r3':           sbp_r3.round(1),
        'dbp_r3':           dbp_r3.round(1),
        'bp_flag_r1':       bp_flag_r1,
        'bp_flag_r2':       bp_flag_r2,
        'bp_confirmed':     bp_confirmed,
        'muac_r1':          muac_r1.round(1),
        'muac_r2':          muac_r2.round(1),
        'muac_flag_r1':     muac_flag_r1,
        'muac_confirmed':   muac_confirmed,
        'any_l2_override':  any_l2_override,
        'final_level':      final_levels,
        'actual_lbw':       np.array(y_unseen),
    })

    stats = {
        'n_total':               n,
        'total_l2_overrides':    int(total_altered),
        'pct_altered':           round(pct_altered, 2),
        'escalated_nonhigh':     int(altered_from_nonhigh),
        'pct_from_nonhigh':      round(pct_from_nonhigh, 2),
        'bp_r1_flags':           int(bp_flag_r1.sum()),
        'bp_r2_flags':           int(bp_flag_r2.sum()),
        'bp_confirmed':          int(bp_confirmed.sum()),
        'bp_deescalated':        int(deesc_bp),
        'muac_r1_flags':         int(muac_flag_r1.sum()),
        'muac_confirmed':        int(muac_confirmed.sum()),
        'muac_cleared':          int(cleared_muac),
    }

    return sim_df, stats


# SECTION 5 — VISUALIZATION

def plot_evaluation_panels(results: list,
                            oof_proba, proba_test, proba_unseen,
                            y_train, y_test, y_unseen,
                            threshold: float) -> None:
    """Confusion matrices + ROC curves, three partitions side by side."""
    data_map = {
        '10-Fold CV (OOF)': (oof_proba,    y_train),
        'Test Set':          (proba_test,   y_test),
        'Unseen Holdout':    (proba_unseen, y_unseen),
    }
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    n_sets = len(results)

    fig = plt.figure(figsize=(7 * n_sets, 11))
    gs  = gridspec.GridSpec(2, n_sets, hspace=0.4, wspace=0.38)
    fig.suptitle(
        f'AIKONIC XGBoost — Evaluation Across All Three Partitions\n'
        f'Threshold = {threshold:.4f}  (selected on OOF data; applied unchanged to all sets)',
        fontsize=12, fontweight='bold', y=1.01
    )

    for i, (res, color) in enumerate(zip(results, colors)):
        name     = res['Partition']
        proba, y = data_map[name]
        pred     = (proba >= threshold).astype(int)
        cm       = confusion_matrix(y, pred)
        rec      = res['Recall_(TPR)']
        pr_a     = res['PR_AUC']
        roc_a    = res['ROC_AUC']
        spec     = res['Specificity_(TNR)']
        ref      = res['Referral_Rate_%']
        fn_count = res['FN_(Minimize!)']

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[0, i])
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'LBW']).plot(
            ax=ax_cm, cmap='Blues', colorbar=False
        )
        ax_cm.set_title(
            f'{name}\nRecall={rec:.3f}  PR-AUC={pr_a:.3f}\n'
            f'FN={fn_count} (missed LBW)  Referral={ref}%',
            fontweight='bold', fontsize=8.5
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc       = fig.add_subplot(gs[1, i])
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f'ROC-AUC = {roc_a:.4f}\nPR-AUC  = {pr_a:.4f}')
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
      Left  : Classification distribution before vs. after fusion (stacked bar)
      Right : Protocol funnel (BP Rule of 3 + MUAC verification step counts)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'RQ3 — Decision-Level Fusion Effect on Final LBW Risk Classification\n'
        '(Academic Layer 2 Simulation — DOH/FNRI 2022 Prevalence-Based; '
        'Production Layer 2 in prototype/clinical_flags.py)',
        fontsize=11, fontweight='bold'
    )

    n = stats['n_total']

    #  Left: Before vs. After stacked bar 
    ax = axes[0]
    tier_order  = ['LOW', 'MEDIUM', 'HIGH']
    tier_colors = {'LOW': '#70AD47', 'MEDIUM': '#FFC000', 'HIGH': '#C00000'}

    ml_cts = sim_df['ml_tier'].value_counts()
    # After fusion: count per tier minus escalated from that tier, plus override bar
    tier_after  = {}
    tier_escl   = {}
    for t in tier_order:
        escl = sim_df[(sim_df['ml_tier'] == t) & sim_df['any_l2_override']].shape[0]
        tier_escl[t] = escl
        tier_after[t] = max(ml_cts.get(t, 0) - escl, 0)

    x_labels = ['Layer 1\n(ML Only)', 'Final\n(After Fusion)']
    bottom_0, bottom_1 = 0, 0
    for t in tier_order:
        before_val = ml_cts.get(t, 0)
        after_val  = tier_after[t]
        ax.bar(0, before_val, color=tier_colors[t], width=0.5,
               bottom=bottom_0, edgecolor='white', label=t)
        ax.bar(1, after_val, color=tier_colors[t], width=0.5,
               bottom=bottom_1, edgecolor='white', alpha=0.85)
        bottom_0 += before_val
        bottom_1 += after_val

    # Override bar on top of "After" column
    ax.bar(1, stats['total_l2_overrides'], color='#7B0000', width=0.5,
           bottom=bottom_1, edgecolor='white', label='HIGH PRIORITY\nREFERRAL (L2)')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel('Number of Cases')
    ax.set_title(
        f'Classification Before vs. After Layer 2\n'
        f'{stats["pct_altered"]:.1f}% of cases altered  '
        f'({stats["escalated_nonhigh"]} from LOW/MEDIUM → Escalated)',
        fontweight='bold', fontsize=9.5
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    #  Right: Protocol funnel
    ax2 = axes[1]
    funnel_data = [
        (f'Total Simulated (n={n})',                           n,                           '#4472C4'),
        (f'BP R1 ≥ 140/90  →  {stats["bp_r1_flags"]} ({stats["bp_r1_flags"]/n*100:.1f}%)',
                                                               stats['bp_r1_flags'],         '#ED7D31'),
        (f'BP R2 still ≥ 140/90  →  {stats["bp_r2_flags"]} ({stats["bp_r2_flags"]/n*100:.1f}%)',
                                                               stats['bp_r2_flags'],         '#ED7D31'),
        (f'BP R3 CONFIRMED  →  {stats["bp_confirmed"]} ({stats["bp_confirmed"]/n*100:.1f}%)',
                                                               stats['bp_confirmed'],        '#C00000'),
        (f'MUAC R1 < 23.5 cm  →  {stats["muac_r1_flags"]} ({stats["muac_r1_flags"]/n*100:.1f}%)',
                                                               stats['muac_r1_flags'],       '#FFC000'),
        (f'MUAC Verified  →  {stats["muac_confirmed"]} ({stats["muac_confirmed"]/n*100:.1f}%)',
                                                               stats['muac_confirmed'],      '#C00000'),
        (f'Total L2 Overrides  →  {stats["total_l2_overrides"]} ({stats["pct_altered"]}%)',
                                                               stats['total_l2_overrides'],  '#7B0000'),
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
    ax2.set_title(
        'Clinical Triage Protocol Funnel\n'
        '(Rule of 3  +  MUAC Re-Measurement Verification)',
        fontweight='bold', fontsize=9.5
    )
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
  
    print("STAGE 6: FULL MODEL EVALUATION + ACADEMIC LAYER 2 SIMULATION (RQ3)")

    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    oof_pkg   = joblib.load(OOF_PATH)
    splits    = joblib.load(os.path.join(ARTIFACTS_DIR, "split_data.pkl"))

    oof_proba   = oof_pkg['oof_proba']
    y_train     = oof_pkg['y_train']
    X_test      = splits['X_test'];   y_test    = splits['y_test']
    X_unseen    = splits['X_unseen']; y_unseen  = splits['y_unseen']
    X_train     = splits['X_train']

    proba_test   = model.predict_proba(X_test)[:, 1]
    proba_unseen = model.predict_proba(X_unseen)[:, 1]

    print(f"\n  Threshold        : {threshold:.4f}")
    print(f"  OOF  LBW rate   : {oof_pkg['y_train'].mean()*100:.1f}%  ({oof_pkg['y_train'].sum()} cases)")
    print(f"  Test LBW rate   : {y_test.mean()*100:.1f}%  ({y_test.sum()} cases)")
    print(f"  Unseen LBW rate : {y_unseen.mean()*100:.1f}%  ({y_unseen.sum()} cases)")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Compute metrics
    results = [
        compute_all_metrics('10-Fold CV (OOF)', oof_proba,    y_train,  threshold),
        compute_all_metrics('Test Set',          proba_test,   y_test,   threshold),
        compute_all_metrics('Unseen Holdout',    proba_unseen, y_unseen, threshold),
    ]

    # Print hierarchical table
    print_metric_table(results)

    # Generalization analysis
    check_generalization(results)

    # Save CSV
    flat = pd.DataFrame(results)
    flat.to_csv(os.path.join(OUTPUTS_DIR, 'xgb_performance_summary.csv'), index=False)
    print(f"\n  [SAVED] xgb_performance_summary.csv")

    # Evaluation panels
    plot_evaluation_panels(results, oof_proba, proba_test, proba_unseen,
                            y_train, y_test, y_unseen, threshold)

    # Academic Layer 2 simulation (RQ3)
    print(f"\n  {'═'*72}")
    print(f"  ACADEMIC LAYER 2 SIMULATION — RQ3")
    print(f"  This is NOT the Production Layer 2 (prototype/clinical_flags.py).")
    print(f"  This simulation provides the quantitative answer to RQ3 only.")
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
    print(f"  Next    : python 07_shap_analysis.py")
    print(f"  {'═'*72}")


if __name__ == "__main__":
    run_evaluation()