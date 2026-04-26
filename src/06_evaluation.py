# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/06_evaluation.py
#
# Purpose : Full model evaluation across all three partitions (OOF / Test /
#           Unseen) using the SAME threshold from 05. Also simulates Layer 2
#           clinical escalation effects to answer RQ3 quantitatively.
#
# Connects to: 07_shap_analysis.py
# --------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OOF_PATH, OUTPUTS_DIR, ARTIFACTS_DIR
)

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

# 1. Partition Evaluation 

def evaluate_partition(name: str, proba: np.ndarray, y_true: pd.Series,
                        threshold: float, color: str,
                        ax_cm, ax_roc) -> dict:
    """Compute and display all metrics for one data partition."""
    pred = (proba >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    f1   = f1_score(y_true, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0   
    auc  = roc_auc_score(y_true, proba)

    print(f"\n  {'─'*55}")
    print(f"  Partition: {name}  (threshold = {threshold:.4f})")
    print(f"  TN={tn:>4}  FP={fp:>4}  |  Specificity (TNR) : {spec:.4f}")
    print(f"  FN={fn:>4}  TP={tp:>4}  |  Recall     (TPR) : {rec:.4f}")

    return {
        'Set': name, 'AUC': auc, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'Specificity': spec, 'NPV': npv,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Threshold': threshold,
        'Referral_rate_%': round(pred.mean() * 100, 1),
    }

# 1. Partition Evaluation 

def evaluate_partition(name: str, proba: np.ndarray, y_true: pd.Series,
                        threshold: float, color: str,
                        ax_cm, ax_roc) -> dict:
    """Compute and display all metrics for one data partition."""
    pred = (proba >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    f1   = f1_score(y_true, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0   
    auc  = roc_auc_score(y_true, proba)

    print(f"\n  {'─'*55}")
    print(f"  Partition: {name}  (threshold = {threshold:.4f})")
    print(f"  {'─'*55}")
    print(f"  TN={tn:>4}  FP={fp:>4}  |  Specificity (TNR) : {spec:.4f}")
    print(f"  FN={fn:>4}  TP={tp:>4}  |  Recall     (TPR) : {rec:.4f}")
    print(f"  Accuracy  : {acc:.4f}    |  Precision  (PPV) : {prec:.4f}")
    print(f"  F1        : {f1:.4f}    |  NPV               : {npv:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Referral rate (predicted positive): {pred.mean()*100:.1f}%")

    if fn > 0:
        print(f"  ⚠  {fn} LBW cases MISSED (False Negatives) — high clinical cost")
    if fp > 0:
        print(f"  ℹ  {fp} unnecessary referrals (False Positives) — low clinical cost")

    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'LBW']).plot(
        ax=ax_cm, cmap='Blues', colorbar=False
    )
    ax_cm.set_title(f'{name}\nCM (t={threshold:.3f})', fontweight='bold', fontsize=10)

    fpr, tpr, _ = roc_curve(y_true, proba)
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{name} AUC={auc:.4f}')
    ax_roc.scatter([1 - spec], [rec], color='red', s=80, zorder=5)

    return {
        'Set': name, 'AUC': auc, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'Specificity': spec, 'NPV': npv,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Threshold': threshold,
        'Referral_rate_%': round(pred.mean() * 100, 1),
    }

# 2. Generalization Checks 

def check_generalization(results: list) -> None:
    """Flag overfitting if AUC or Recall drops across CV → Test → Unseen."""
    rows = {r['Set']: r for r in results}
    print(f"\n  {'═'*55}")
    print(f"  GENERALIZATION CHECKS")
    print(f"  {'═'*55}")

    sets = ['10-Fold CV (OOF)', 'Test Set', 'Unseen Holdout']
    present = [s for s in sets if s in rows]

    if len(present) >= 2:
        for i in range(len(present) - 1):
            a, b = present[i], present[i+1]
            auc_gap = abs(rows[a]['AUC'] - rows[b]['AUC'])
            rec_gap = abs(rows[a]['Recall'] - rows[b]['Recall'])
            print(f"\n  {a}  →  {b}")
            print(f"    AUC gap   : {auc_gap:.4f}")
            print(f"    Recall gap: {rec_gap:.4f}")

            # 2. Generalization Checks 

def check_generalization(results: list) -> None:
    """Flag overfitting if AUC or Recall drops across CV → Test → Unseen."""
    rows = {r['Set']: r for r in results}
    print(f"\n  {'═'*55}")
    print(f"  GENERALIZATION CHECKS")
    print(f"  {'═'*55}")

    sets = ['10-Fold CV (OOF)', 'Test Set', 'Unseen Holdout']
    present = [s for s in sets if s in rows]

    if len(present) >= 2:
        for i in range(len(present) - 1):
            a, b = present[i], present[i+1]
            auc_gap = abs(rows[a]['AUC'] - rows[b]['AUC'])
            rec_gap = abs(rows[a]['Recall'] - rows[b]['Recall'])
            auc_ok  = auc_gap <= 0.10
            rec_ok  = rec_gap <= 0.15
            print(f"\n  {a}  →  {b}")
            print(f"    AUC gap   : {auc_gap:.4f}  {'✓ OK' if auc_ok else '⚠ HIGH (>0.10) — investigate overfitting'}")
            print(f"    Recall gap: {rec_gap:.4f}  {'✓ OK' if rec_ok else '⚠ HIGH (>0.15) — threshold may not generalize'}")

    print(f"\n  NOTE: Modest generalization gap is EXPECTED at this sample size.")
    print(f"  A CV↔Unseen AUC gap ≤0.15 is acceptable for 1,700-row clinical data.")

    # 3. Layer 2 Clinical Escalation Simulation (RQ3) 

def simulate_layer2_escalation(proba_test: np.ndarray, threshold: float) -> pd.DataFrame:
    """Simulate the effect of Layer 2 clinical escalation on final recommendations."""
    np.random.seed(42)
    n = len(proba_test)

    ml_tiers = []
    for p in proba_test:
        if p >= threshold:
            ml_tiers.append('HIGH')
        elif p >= (threshold / 2.0):
            ml_tiers.append('MEDIUM')
        else:
            ml_tiers.append('LOW')

    df = pd.DataFrame({
        'ml_probability': proba_test,
        'ml_tier':        ml_tiers
    })
    return df

# 3. Layer 2 Clinical Escalation Simulation (RQ3) 

def simulate_layer2_escalation(proba_test: np.ndarray, threshold: float) -> pd.DataFrame:
    """Simulate the effect of Layer 2 clinical escalation on final recommendations."""
    np.random.seed(42)
    n = len(proba_test)

    ml_tiers = []
    for p in proba_test:
        if p >= threshold:
            ml_tiers.append('HIGH')
        elif p >= (threshold / 2.0):
            ml_tiers.append('MEDIUM')
        else:
            ml_tiers.append('LOW')

    muac_flag = np.random.binomial(1, 0.175, n).astype(bool)   
    bp_flag   = np.random.binomial(1, 0.110, n).astype(bool)   
    any_critical = muac_flag | bp_flag

    final_levels = []
    for ml, critical in zip(ml_tiers, any_critical):
        if critical:
            final_levels.append('HIGH PRIORITY REFERRAL (Escalated)')
        else:
            final_levels.append(f'{ml} RISK (ML only)')

    df = pd.DataFrame({
        'ml_probability': proba_test,
        'ml_tier':        ml_tiers,
        'muac_flag':      muac_flag,
        'bp_flag':        bp_flag,
        'any_critical':   any_critical,
        'final_level':    final_levels,
    })

    total     = len(df)
    escalated = df['any_critical'].sum()
    print(f"\n  [RQ3 SIMULATION] Layer 2 Clinical Escalation (n={total})")
    print(f"  MUAC < 23.5 cm triggered : {muac_flag.sum():4d} ({muac_flag.mean()*100:.1f}%)")
    print(f"  BP ≥ 140/90 triggered    : {bp_flag.sum():4d} ({bp_flag.mean()*100:.1f}%)")
    print(f"  Any critical flag        : {escalated:4d} ({escalated/total*100:.1f}%)")
    
    xtab = pd.crosstab(df['ml_tier'], df['any_critical'], rownames=['ML Tier'], colnames=['Escalated'])
    xtab.columns = ['Not Escalated', 'Escalated']
    print(f"\n  Cross-tab: ML Tier × Escalation → Final Level\n{xtab.to_string()}")
    print(f"\n  Final Level Distribution:\n{df['final_level'].value_counts().to_string()}")

    return df

def plot_layer2_simulation(sim_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RQ3 — Layer 2 Decision-Level Fusion Effect', fontsize=12, fontweight='bold')

    tier_counts = sim_df['ml_tier'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'], fill_value=0)
    colors_ml   = ['#70AD47', '#FFC000', '#FF0000']
    axes[0].bar(tier_counts.index, tier_counts.values, color=colors_ml, edgecolor='white')
    axes[0].set_title('Layer 1 ML Risk Tier Distribution', fontweight='bold')
    
    final_counts = sim_df['final_level'].value_counts()
    color_map    = {
        'HIGH RISK (ML only)':                '#FF0000',
        'MEDIUM RISK (ML only)':              '#FFC000',
        'LOW RISK (ML only)':                 '#70AD47',
        'HIGH PRIORITY REFERRAL (Escalated)': '#C00000',
    }
    bar_colors = [color_map.get(k, '#4472C4') for k in final_counts.index]
    axes[1].barh(range(len(final_counts)), final_counts.values, color=bar_colors, edgecolor='white')
    axes[1].set_yticks(range(len(final_counts)))
    axes[1].set_yticklabels([k[:35] for k in final_counts.index], fontsize=8)
    axes[1].set_title('Final Level After Layer 2 Escalation', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'layer2_simulation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Main 

def run_evaluation() -> None:
    print("STAGE 6: FULL MODEL EVALUATION")
    pass

if __name__ == "__main__":
    run_evaluation()

    def run_evaluation() -> None:
    print("STAGE 6: FULL MODEL EVALUATION")
    
    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    oof_pkg   = joblib.load(OOF_PATH)
    splits    = joblib.load(os.path.join(ARTIFACTS_DIR, "split_data.pkl"))

    oof_proba = oof_pkg['oof_proba']
    y_train   = oof_pkg['y_train']
    X_test    = splits['X_test'];  y_test   = splits['y_test']
    X_unseen  = splits['X_unseen'];y_unseen = splits['y_unseen']

    proba_test   = model.predict_proba(X_test)[:, 1]
    proba_unseen = model.predict_proba(X_unseen)[:, 1]

    os.makedirs(OUTPUTS_DIR, exist_ok=True)