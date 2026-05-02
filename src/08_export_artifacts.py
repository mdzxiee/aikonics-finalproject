# Purpose : Verify, bundle, and copy the 3-artifact set to prototype/.
#           Register the model version in the database.
#           Generate the artifact manifest for reproducibility documentation.
#
# Input  : artifacts/model.pkl
#          artifacts/threshold.pkl
#          artifacts/features.json
#          outputs/xgb_performance_summary.csv  (for manifest metadata)
#
# Output : prototype/model.pkl        ← copy (used by predictor.py)
#          prototype/threshold.pkl    ← copy (used by predictor.py)
#          prototype/features.json    ← copy (schema validation)
#          outputs/artifact_manifest.json
#
# WHY THREE SEPARATE ARTIFACTS (not one bundle) 
#
#   model.pkl      : Trained XGBClassifier. Large. Version-controlled.
#                    Replace only when model is retrained.
#
#   threshold.pkl  : A single float.
#                    Can be recalibrated without retraining the model.
#                    Example: a health district with higher LBW prevalence
#                    could lower the threshold for higher sensitivity,
#                    without touching the underlying XGBoost model.
#
#   features.json  : Ordered feature list with count.
#                    Validates that the BHW form submits features in the
#                    exact order and count the model was trained on.
#                    Prevents silent errors if the form schema changes.
#
# ─ DUAL-PURPOSE LAYER 2 NOTE
#   The prototype's clinical_flags.py implements BOTH:
#     ESCALATION: confirmed clinical danger → HIGH PRIORITY REFERRAL
#     DE-ESCALATION: ML-flagged + no clinical danger → ELEVATED MONITORING
#   This is the production implementation of the fusion logic.
#   It is separate from the academic simulation in 06_evaluation.py.
#
# Connects to: prototype/predictor.py, prototype/app.py, 09_database_seed.py
# ===========================================================================

import os, sys, json, shutil, warnings
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH,
    ARTIFACTS_DIR, OUTPUTS_DIR, PROTO_DIR, FEATURE_COLS
)

# SECTION 1 — ARTIFACT VERIFICATION

def verify_artifacts() -> dict:
    """
    Verify all three required artifacts exist and are internally consistent.

    Checks performed:
      1. Each file exists and is loadable
      2. threshold.pkl is a float in (0, 1)
      3. features.json feature list matches config.FEATURE_COLS exactly
      4. model.n_features_in_ matches len(FEATURE_COLS) where available

    Raises immediately if any check fails — prevents deploying a broken bundle.
    """
    required = {
        'model.pkl':     MODEL_PATH,
        'threshold.pkl': THRESHOLD_PATH,
        'features.json': FEATURES_PATH,
    }

    print(f"\n  [VERIFY] Checking required artifacts:")
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  MISSING: {path}\n"
                f"  Run the pipeline stages in order:\n"
                f"    01 → 04 → 05 → (06 → 07 optional) → 08"
            )
        size_kb = os.path.getsize(path) / 1024
        print(f"    ✓ {name:<20}  ({size_kb:.1f} KB)  → {path}")

    # Load and validate content
    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    with open(FEATURES_PATH) as f:
        feat_json = json.load(f)

    # Threshold sanity
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(
            f"threshold.pkl value {threshold} is outside (0, 1).\n"
            f"Re-run 05_threshold_validation.py."
        )
    print(f"    ✓ Threshold = {float(threshold):.4f} (valid range) ✓")

    # Feature list consistency
    json_feats   = feat_json.get('features', [])
    config_feats = FEATURE_COLS
    if json_feats != config_feats:
        raise ValueError(
            f"\n  FEATURE MISMATCH between features.json and config.FEATURE_COLS!\n"
            f"  features.json : {json_feats}\n"
            f"  config.py     : {config_feats}\n"
            f"  Re-run 05_threshold_validation.py to regenerate features.json."
        )
    print(f"    ✓ features.json matches config.FEATURE_COLS "
          f"({len(config_feats)} features) ✓")

    # Model feature count (XGBoost 3.x)
    try:
        n_model = model.n_features_in_
        if n_model != len(FEATURE_COLS):
            raise ValueError(
                f"Model expects {n_model} features but "
                f"config.FEATURE_COLS has {len(FEATURE_COLS)}.\n"
                f"Re-run 04_model_training.py."
            )
        print(f"    ✓ model.n_features_in_ = {n_model} ✓")
    except AttributeError:
        print(f"    ℹ  n_features_in_ not available (older XGBoost) — skipped")

    return {
        'model':     model,
        'threshold': float(threshold),
        'features':  json_feats,
    }
# SECTION 2 — COPY TO PROTOTYPE

def copy_to_prototype() -> None:
    """
    Copy verified artifacts into the prototype/ directory.

    The prototype loads from prototype/ — not from artifacts/ — so that
    the prototype folder can be distributed or deployed independently of
    the full training pipeline.
    """
    os.makedirs(PROTO_DIR, exist_ok=True)

    copies = [
        (MODEL_PATH,     os.path.join(PROTO_DIR, 'model.pkl')),
        (THRESHOLD_PATH, os.path.join(PROTO_DIR, 'threshold.pkl')),
        (FEATURES_PATH,  os.path.join(PROTO_DIR, 'features.json')),
    ]

    print(f"\n  [COPY] Deploying artifacts → {PROTO_DIR}/")
    for src, dst in copies:
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"    Copied: {os.path.basename(src):<20}  ({size_kb:.1f} KB)")

# SECTION 3 — LOAD PERFORMANCE METADATA

def load_performance_metadata() -> dict:
    """
    Load key metrics from the evaluation CSV for the manifest.
    Uses exact Partition names as keys to prevent silent lookup failures.
    """
    perf_path = os.path.join(OUTPUTS_DIR, 'xgb_performance_summary.csv')
    meta = {}
    if not os.path.exists(perf_path):
        print(f"  [META] xgb_performance_summary.csv not found — metrics omitted.")
        return meta

    df = pd.read_csv(perf_path)

    for _, row in df.iterrows():
        key = str(row.get('Partition', 'Unknown'))
        
        meta[key] = {
            'ROC_AUC':        round(float(row.get('ROC_AUC', 0)), 4),
            'PR_AUC':         round(float(row.get('PR_AUC', 0)), 4),
            'Recall':         round(float(row.get('Recall_(TPR)', 0)), 4),
            'Precision':      round(float(row.get('Precision_(PPV)', 0)), 4),
            'F1':             round(float(row.get('F1_Score', 0)), 4),
            'Referral_Rate_%':round(float(row.get('Referral_Rate_%', 0)), 1),
            'TP':             int(row.get('TP', 0)),
            'FN':             int(row.get('FN_(Minimize!)', 0)),
            'FP':             int(row.get('FP', 0)),
            'TN':             int(row.get('TN', 0)),
        }

    print(f"\n  [META] Performance metadata loaded from xgb_performance_summary.csv")
    for part, m in meta.items():
        print(f"    {part}: ROC-AUC={m['ROC_AUC']} | PR-AUC={m['PR_AUC']} | Recall={m['Recall']}")
    return meta

# SECTION 4 — ARTIFACT MANIFEST

def save_artifact_manifest(verified: dict, meta: dict) -> None:
    """
    Save a JSON manifest documenting the complete artifact bundle.

    This manifest serves as the traceability record for:
      - Thesis appendix (proving reproducibility)
      - Panel defense (confirming which model version was evaluated)
      - Any future replication of the prototype deployment
    """
    manifest = {
        'project':           'AIKONIC — LBW Risk Prediction',
        'model_version':     'v1.0',
        'created_at':        datetime.now().isoformat(),
        'pipeline_stages':   ['01_preprocessing', '04_model_training',
                              '05_threshold_validation', '06_evaluation',
                              '07_shap_analysis', '08_export_artifacts'],
        'random_seed':       42,
        'threshold':         verified['threshold'],
        'n_features':        len(verified['features']),
        'feature_list':      verified['features'],
        'artifacts_bundled': ['model.pkl', 'threshold.pkl', 'features.json'],
        'prototype_dir':     PROTO_DIR,
        'performance':       meta,
        'architecture_notes': {
            'layer1': 'XGBClassifier (no Pipeline) with NaN handled natively',
            'layer2': 'Dual-purpose: escalation (clinical danger → referral) '
                      'AND de-escalation (ML-flagged + no danger → monitoring)',
            'threshold_selection': (
                'Recall-prioritized on OOF probabilities only. '
                f'Value: {verified["threshold"]:.4f}. '
                'Addresses: FN (missed LBW) costs more than FP (extra referral).'
            ),
            'shap': 'TreeExplainer with tree_path_dependent (NaN-safe)',
            'leakage_prevention': (
                'GroupShuffleSplit + StratifiedGroupKFold; '
                '89 mothers with multiple records; zero overlap verified by assertion.'
            ),
        },
        'limitations': [
            'AUC ceiling ~0.58-0.65 for NDHS sociodemographic features (no hemoglobin/BMI)',
            'OOF referral rate ~39-43% — mitigated by dual-purpose Layer 2 de-escalation',
            'Layer 2 RQ3 evidence is simulation-based (no real BP/MUAC in NDHS)',
            'Small training set: ~1,267 rows, ~155 LBW cases',
            'Prototype: proof-of-concept, not field-tested',
        ],
    }

    manifest_path = os.path.join(OUTPUTS_DIR, 'artifact_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  [SAVED] artifact_manifest.json")
    print(f"    Threshold  : {verified['threshold']:.4f}")
    print(f"    Features   : {len(verified['features'])}")
    print(f"    Created    : {manifest['created_at'][:19]}")
    if meta:
        print(f"    Performance summary:")
        for part, m in meta.items():
            ref = m.get('Referral_Rate_%', '?')
            print(f"      {part:<35} AUC={m['ROC_AUC']:.4f} | "
                  f"Recall={m['Recall']:.4f} | Referral={ref}%")

# SECTION 5 — REGISTER IN DATABASE

def register_model_in_database(verified: dict, meta: dict) -> None:
    """
    Insert a record into model_registry in the SQLite database.
    Deactivates any previously active model version (is_active = 0).
    """
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'database', 'aikonic.db'
    )
    if not os.path.exists(db_path):
        print(f"\n  [DB] Database not initialized yet — skipping registration.")
        print(f"       Run 09_database_seed.py to initialize the database.")
        return

    import sqlite3

    perf_oof    = meta.get('10-Fold CV (OOF)', {})
    perf_test   = meta.get('Test Set', {})
    perf_unseen = meta.get('Unseen Holdout', {})

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE model_registry SET is_active = 0")
        conn.execute("""
            INSERT INTO model_registry
                (model_version, model_path, threshold, oof_auc,
                 test_auc, unseen_recall, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            'v1.0',
            os.path.join(PROTO_DIR, 'model.pkl'),
            verified['threshold'],
            perf_oof.get('ROC_AUC'),
            perf_test.get('ROC_AUC'),
            perf_unseen.get('Recall'),
        ))
        conn.commit()
    print(f"\n  [DB] Model v1.0 registered in model_registry (is_active=1)")

# MAIN

def run_export() -> None:
    print("=" * 70)
    print("STAGE 8: ARTIFACT EXPORT AND PROTOTYPE PACKAGING")
    print("=" * 70)
    print(f"\n  Bundling: model.pkl + threshold.pkl + features.json")
    print(f"  Why separate: threshold recalibratable without retraining")
    print(f"  Why features.json: schema validation for BHW form inputs")

    # 1. Verify
    verified = verify_artifacts()

    # 2. Copy to prototype/
    copy_to_prototype()

    # 3. Load performance metadata
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    meta = load_performance_metadata()

    # 4. Save manifest
    save_artifact_manifest(verified, meta)

    # 5. Register in DB (if initialized)
    register_model_in_database(verified, meta)

    print(f"\n  {'═'*60}")
    print(f"  STAGE 8 COMPLETE")
    print(f"  Prototype artifacts ready in: {PROTO_DIR}/")
    print(f"  Next: python src/09_database_seed.py")
    print(f"  Then: python prototype/app.py  → localhost:5000")
    print(f"  {'═'*60}")


if __name__ == "__main__":
    run_export()