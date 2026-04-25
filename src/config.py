# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/config.py
# Purpose: Single source of truth for all pipeline constants.
#          Every other module imports from here. Never hardcode values.
# -------------------------------------------------------------------------------------

import os

# Paths 
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
DB_DIR        = os.path.join(BASE_DIR, "database")
PROTO_DIR     = os.path.join(BASE_DIR, "prototype")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "PHKR82FL.csv")  # also accepts .xlsx

# Split CSV exports (produced by 04_model_training.py)
X_TRAIN_PATH  = os.path.join(ARTIFACTS_DIR, "X_train.csv")
Y_TRAIN_PATH  = os.path.join(ARTIFACTS_DIR, "y_train.csv")
X_TEST_PATH   = os.path.join(ARTIFACTS_DIR, "X_test.csv")
Y_TEST_PATH   = os.path.join(ARTIFACTS_DIR, "y_test.csv")
X_UNSEEN_PATH = os.path.join(ARTIFACTS_DIR, "X_unseen.csv")
Y_UNSEEN_PATH = os.path.join(ARTIFACTS_DIR, "y_unseen.csv")

# Model artifacts
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.pkl")
FEATURES_PATH  = os.path.join(ARTIFACTS_DIR, "features.json")
OOF_PATH       = os.path.join(ARTIFACTS_DIR, "oof_probabilities.pkl")
PREPROCESSED   = os.path.join(ARTIFACTS_DIR, "preprocessed.pkl")
DB_PATH        = os.path.join(DB_DIR, "aikonic.db")