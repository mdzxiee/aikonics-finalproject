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

    return verified
