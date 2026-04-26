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