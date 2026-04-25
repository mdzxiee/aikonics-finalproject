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

COL_RENAME = {
    'v012': 'maternal_age',
    'v133': 'education_yrs',
    'v191': 'wealth_score',
    'bord': 'birth_order',
    'b11':  'birth_interval',
    'v025': 'residence_type',
    'v024': 'region',
    'm14':  'anc_first_timing',  
    'm45':  'iron_supplement',
    'm46':  'iron_days',
    'm1':   'tetanus_shots',
    'v501': 'marital_status',
    'v136': 'household_size',
}

# Validity Filter Thresholds 
GESTATIONAL_AGE_MIN  = 9       # b20 >= 9 (full-term only)
WEIGHT_MAX_GRAMS     = 9000    # m19 < 9000 (removes DHS coded 9996/9998)
LBW_THRESHOLD_GRAMS  = 2500   # WHO/DOH LBW definition

# Splitting  
RANDOM_STATE         = 42
UNSEEN_FRAC          = 0.10    # sealed immediately
TEST_FRAC            = 0.20    # of remaining 90%
CV_FOLDS             = 10

# For early stopping validation (held out from training only)
ES_VALIDATION_FRAC   = 0.10    # 10% of training data
EARLY_STOPPING_ROUNDS = 30

# XGBoost Hyperparameters 
# eval_metric = "aucpr":
#   PR-AUC is used as the MONITORING metric for early stopping.
#   IMPORTANT: This does NOT change the training loss (binary cross-entropy).
#   scale_pos_weight handles the loss imbalance. aucpr is the correct
#   stopping criterion for imbalanced data — logloss can decrease while
#   minority-class recall degrades (it is majority-dominated even with
#   scale_pos_weight in the monitoring context).
#
# No imputer in pipeline:
#   XGBoost handles NaN natively via "sparsity-aware split finding."
#   Imputing before XGBoost removes XGBoost's ability to learn the optimal
#   direction for missing values from the data.
#   SHAP uses tree_path_dependent mode which also handles NaN correctly.
#
# scale_pos_weight: set DYNAMICALLY from n_neg / n_pos in training script.

XGB_PARAMS = {
    'n_estimators':      500,      # upper bound; early stopping finds optimal
    'max_depth':         3,        # shallow — overlapping classes (Cohen's d < 0.20)
    'learning_rate':     0.05,     # slow, stable convergence
    'subsample':         0.8,      # row subsampling per tree
    'colsample_bytree':  0.8,      # feature subsampling per tree
    'min_child_weight':  5,        # prevents splits on < 5 samples (minority safety)
    'reg_alpha':         0.1,      # L1 regularization
    'reg_lambda':        2.0,      # L2 regularization
    'eval_metric':       'aucpr',  # PR-AUC monitoring — correct for imbalanced data
    'random_state':      RANDOM_STATE,
    'verbosity':         0,
    'n_jobs':            -1,
    # scale_pos_weight: set dynamically (n_neg / n_pos) in 04_model_training.py
}