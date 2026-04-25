# ===========================================================================
# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/config.py
# Purpose: Single source of truth for all pipeline constants.
#          Every other module imports from here. Never hardcode values.
# ===========================================================================

import os

# Paths 
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
DB_DIR        = os.path.join(BASE_DIR, "database")
PROTO_DIR     = os.path.join(BASE_DIR, "prototype")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "PHKR82FL.csv")  # also accepts .xlsx