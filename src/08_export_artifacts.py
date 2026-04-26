# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/08_export_artifacts.py
#
# Purpose : Collect and verify all three prototype artifacts (model.pkl,
#           threshold.pkl, features.json), copy them to prototype/ directory,
#           register the model version in the database, and produce an
#           artifact manifest for reproducibility documentation.
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
    """Verify all three required artifacts exist and are internally consistent."""
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

    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    with open(FEATURES_PATH) as f:
        feat_json = json.load(f)

    json_feats   = feat_json.get('features', [])
    config_feats = FEATURE_COLS
    if json_feats != config_feats:
        raise ValueError(
            f"FEATURE MISMATCH between features.json and config.py!\n"
            f"  features.json: {json_feats}\n"
            f"  config.FEATURE_COLS: {config_feats}\n"
            "Regenerate features.json by rerunning 05_threshold_validation.py."
        )

    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(
            f"Threshold {threshold} is outside (0, 1). Check 05_threshold_validation.py."
        )

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
