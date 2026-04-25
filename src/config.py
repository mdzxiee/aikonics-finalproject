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

# Column Loading 
# v001, v002, v003 are loaded for GroupShuffleSplit only — dropped before modeling
LOAD_COLS = [
    'v001', 'v002', 'v003',     # Mother identifier (group key, NOT features)
    'm19', 'm19a', 'b20',       # Target + validity filters (dropped after use)
    'v012',   # maternal age
    'v133',   # education years
    'v191',   # wealth index score (continuous)
    'bord',   # birth order
    'b11',    # preceding birth interval (months)
    'v025',   # residence type: 1=urban, 2=rural
    'v024',   # region code
    'm14',    # month of FIRST ANC visit (0=no ANC, 1-9=month)
    'm45',    # received iron supplementation: 1=yes, 0=no
    'm46',    # days took iron supplements
    'm1',     # tetanus toxoid injections received
    'v501',   # marital status
    'v136',   # number of household members
]

# Feature Columns (model inputs) 
FEATURE_COLS = [
    'maternal_age',
    'education_yrs',
    'wealth_score',
    'birth_order',
    'birth_interval',
    'residence_type',
    'region',
    'anc_first_timing',   # m14 — month of first ANC visit
    'iron_supplement',
    'iron_days',
    'tetanus_shots',
    'marital_status',
    'household_size',
]