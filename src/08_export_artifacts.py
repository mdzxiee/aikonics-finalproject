# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/08_export_artifacts.py
#
# Purpose : Collect and verify all three prototype artifacts (model.pkl,
#           threshold.pkl, features.json), copy them to prototype/ directory,
#           register the model version in the database, and produce an
#           artifact manifest for reproducibility documentation.
#
# Input   : artifacts/model.pkl
#           artifacts/threshold.pkl
#           artifacts/features.json
#           outputs/xgb_performance_summary.csv   (for metadata)
# Output  : prototype/model.pkl        (copy — used by predictor.py)
#           prototype/threshold.pkl    (copy — used by predictor.py)
#           prototype/features.json    (copy — schema validation)
#           outputs/artifact_manifest.json
#
# WHY THREE SEPARATE ARTIFACTS (not one bundle):
#   model.pkl      — The trained XGBClassifier. Large; version-controlled.
#   threshold.pkl  — A single float. Can be recalibrated by a health authority
#                    without retraining the model (e.g., if a region wants
#                    higher sensitivity for high-risk barangays).
#   features.json  — The feature list with order. Validates that any new
#                    data fed to the model uses the SAME columns in the
#                    SAME order as the training set. Prevents silent errors
#                    in the prototype if the form schema changes.
#
# Connects to: prototype/app.py, prototype/predictor.py, 09_database_seed.py
# ----------------------------------------------------------------------------
import os, sys, json, shutil, warnings
from datetime import datetime
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
   MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH,
   ARTIFACTS_DIR, OUTPUTS_DIR, PROTO_DIR, FEATURE_COLS
)

#  Artifact Verification
def verify_artifacts() -> dict:
   """
   Verify all three required artifacts exist and are internally consistent.
   Raises FileNotFoundError if any are missing.
   Raises ValueError if feature list in features.json does not match config.
   """
   required = {
       'model.pkl':     MODEL_PATH,
       'threshold.pkl': THRESHOLD_PATH,
       'features.json': FEATURES_PATH,
   }
   verified = {}

   print(f"\n  [VERIFY] Checking required artifacts:")
   for name, path in required.items():
       if not os.path.exists(path):
           raise FileNotFoundError(
               f"Missing artifact: {path}\n"
               f"Run the pipeline stages in order before exporting."
           )
       size_kb = os.path.getsize(path) / 1024
       print(f"    ✓ {name:<20} ({size_kb:.1f} KB)")
       verified[name] = path

   # Load and validate
   model     = joblib.load(MODEL_PATH)
   threshold = joblib.load(THRESHOLD_PATH)

   with open(FEATURES_PATH) as f:
       feat_json = json.load(f)

   # Feature consistency check
   json_feats   = feat_json.get('features', [])
   config_feats = FEATURE_COLS
   if json_feats != config_feats:
       raise ValueError(
           f"FEATURE MISMATCH between features.json and config.py!\n"
           f"  features.json: {json_feats}\n"
           f"  config.FEATURE_COLS: {config_feats}\n"
           "Regenerate features.json by rerunning 05_threshold_validation.py."
       )

   # Threshold sanity check
   if not (0.0 < float(threshold) < 1.0):
       raise ValueError(
           f"Threshold {threshold} is outside (0, 1). Check 05_threshold_validation.py."
       )

   # Model feature count check
   try:
       n_model_feats = model.n_features_in_
       if n_model_feats != len(FEATURE_COLS):
           raise ValueError(
               f"Model expects {n_model_feats} features but config has {len(FEATURE_COLS)}."
           )
       print(f"    ✓ Model feature count matches config ({n_model_feats})")
   except AttributeError:
       print(f"    ℹ  n_features_in_ not available on this XGBoost version — skip count check")


   print(f"    ✓ Threshold = {float(threshold):.4f} (valid range)")
   print(f"    ✓ features.json matches config.FEATURE_COLS ({len(FEATURE_COLS)} features)")


   return {'model': model, 'threshold': float(threshold), 'features': json_feats}

# Copy Artifacts to Prototype
def copy_to_prototype() -> None:
   """
   Copy the three validated artifacts into the prototype/ directory.
   The prototype loads from prototype/ — not from artifacts/ — so that
   the prototype can be distributed independently of the pipeline.
   """
   os.makedirs(PROTO_DIR, exist_ok=True)

   copies = [
       (MODEL_PATH,     os.path.join(PROTO_DIR, 'model.pkl')),
       (THRESHOLD_PATH, os.path.join(PROTO_DIR, 'threshold.pkl')),
       (FEATURES_PATH,  os.path.join(PROTO_DIR, 'features.json')),
   ]

   print(f"\n  [COPY] Deploying artifacts to {PROTO_DIR}/")
   for src, dst in copies:
       shutil.copy2(src, dst)
       print(f"    Copied: {os.path.basename(src)}")

#  Load Performance Metadata
def load_performance_metadata() -> dict:
   """Load key metrics from evaluation for the artifact manifest."""
   perf_path = os.path.join(OUTPUTS_DIR, 'xgb_performance_summary.csv')
   meta = {}
   if os.path.exists(perf_path):
       df = pd.read_csv(perf_path)
       for _, row in df.iterrows():
           # Matches the exact column name 'Partition' from 06_evaluation.py
           s = str(row.get('Partition', '')).replace(' ', '_').replace('(', '').replace(')', '')
           meta[s] = {
               # Matches the updated clinical column headers
               'ROC_AUC': round(float(row.get('ROC_AUC', 0)), 4),
               'PR_AUC':  round(float(row.get('PR_AUC', 0)), 4),
               'Recall':  round(float(row.get('Recall_(TPR)', 0)), 4),
               'F1':      round(float(row.get('F1_Score', 0)), 4),
           }
   else:
       print(f"  [META] xgb_performance_summary.csv not found — metrics omitted from manifest.")
   return meta

# Artifact Manifest
def save_artifact_manifest(verified: dict, meta: dict) -> None:
   """
   Save a JSON manifest documenting the full artifact bundle.
   This is the traceability record for the thesis and for any future
   replication of the prototype.
   """
   manifest = {
       'project':         'AIKONIC — LBW Risk Prediction',
       'model_version':   'v1.0',
       'created_at':      datetime.now().isoformat(),
       'threshold':       verified['threshold'],
       'n_features':      len(verified['features']),
       'feature_list':    verified['features'],
       'artifacts_bundled': ['model.pkl', 'threshold.pkl', 'features.json'],
       'prototype_dir':   PROTO_DIR,
       'performance':     meta,
       'notes': (
           'threshold.pkl is separate from model.pkl by design — allows '
           'recalibration without retraining. features.json enforces '
           'schema validation in the prototype input form.'
       ),
   }
   manifest_path = os.path.join(OUTPUTS_DIR, 'artifact_manifest.json')
   with open(manifest_path, 'w') as f:
       json.dump(manifest, f, indent=2)
   print(f"\n  [SAVED] artifact_manifest.json")
   print(f"    Threshold   : {verified['threshold']:.4f}")
   print(f"    Features    : {len(verified['features'])}")
   if meta:
       for name, m in meta.items():
           print(f"    {name:<25} ROC-AUC={m.get('ROC_AUC', 0):.4f}  PR-AUC={m.get('PR_AUC', 0):.4f}  Recall={m.get('Recall', 0):.4f}")

#  Register in Database
def register_model_in_database(verified: dict, meta: dict) -> None:
   """
   Insert a record into model_registry in the SQLite database.
   Deactivates any previously active model version.
   """
   db_path = os.path.join(
       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
       'database', 'aikonic.db'
   )
   if not os.path.exists(db_path):
       print(f"\n  [DB] Database not initialized yet — skipping model registration.")
       print(f"       Run 09_database_seed.py to initialize the database.")
       return

   import sqlite3
   with sqlite3.connect(db_path) as conn:
       conn.execute("UPDATE model_registry SET is_active = 0")
       perf_test   = meta.get('Test_Set', {})
       perf_unseen = meta.get('Unseen_Holdout', {})
       conn.execute("""
           INSERT INTO model_registry
               (model_version, model_path, threshold, test_auc, unseen_recall, is_active)
           VALUES (?, ?, ?, ?, ?, 1)
       """, (
           'v1.0',
           os.path.join(PROTO_DIR, 'model.pkl'),
           verified['threshold'],
           perf_test.get('ROC_AUC'),
           perf_unseen.get('Recall'),
       ))
       conn.commit()
   print(f"\n  [DB] Model registered in model_registry (is_active=1)")

#  Main
def run_export() -> None:
   print("=" * 65)
   print("STAGE 8: ARTIFACT EXPORT AND PROTOTYPE PACKAGING")
   print("=" * 65)

   verified = verify_artifacts()
   copy_to_prototype()

   os.makedirs(OUTPUTS_DIR, exist_ok=True)
   meta = load_performance_metadata()
   save_artifact_manifest(verified, meta)
   register_model_in_database(verified, meta)

   print(f"\n  [EXPORT COMPLETE]")
   print(f"  Prototype is ready. Start the API with:")
   print(f"    cd {PROTO_DIR}")
   print(f"    python app.py")

if __name__ == "__main__":
   run_export()
