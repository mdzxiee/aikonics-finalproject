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

