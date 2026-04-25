# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/config.py
# Purpose: Single source of truth for all pipeline constants.
#          Every other module imports from here. Never hardcode values.
# -------------------------------------------------------------------------

import os

# Paths 
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
DB_DIR        = os.path.join(BASE_DIR, "database")
PROTO_DIR     = os.path.join(BASE_DIR, "prototype")

RAW_DATA_PATH = os.path.join(DATA_DIR, "PHKR82FL.csv")   # also accepts .xlsx

# Split CSV exports — produced by 04_model_training.py, consumed by 02, 03, 06, 07
X_TRAIN_PATH  = os.path.join(ARTIFACTS_DIR, "X_train.csv")
Y_TRAIN_PATH  = os.path.join(ARTIFACTS_DIR, "y_train.csv")
X_TEST_PATH   = os.path.join(ARTIFACTS_DIR, "X_test.csv")
Y_TEST_PATH   = os.path.join(ARTIFACTS_DIR, "y_test.csv")
X_UNSEEN_PATH = os.path.join(ARTIFACTS_DIR, "X_unseen.csv")
Y_UNSEEN_PATH = os.path.join(ARTIFACTS_DIR, "y_unseen.csv")

# Model artifacts — THREE SEPARATE FILES by design:
#   model.pkl     : trained XGBClassifier (large, versioned)
#   threshold.pkl : single float (recalibratable without retraining)
#   features.json : ordered feature list (schema validation for prototype form)
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.pkl")
FEATURES_PATH  = os.path.join(ARTIFACTS_DIR, "features.json")
OOF_PATH       = os.path.join(ARTIFACTS_DIR, "oof_probabilities.pkl")
PREPROCESSED   = os.path.join(ARTIFACTS_DIR, "preprocessed.pkl")
DB_PATH        = os.path.join(DB_DIR, "aikonic.db")

# Column Loading 
# v001/v002/v003 : group key for GroupShuffleSplit — NOT features, dropped before training
# m19/m19a/b20   : used for filters + target creation — dropped after use

LOAD_COLS = [
    'v001', 'v002', 'v003',   # Mother group key
    'm19', 'm19a', 'b20',     # Filter + target source
    'v012',    # maternal age
    'v133',    # education years (valid 0–20; 98=DK → NaN in preprocessing)
    'v191',    # wealth factor score (raw DHS large integer: ~-280K to +280K)
    'bord',    # birth order (valid 1–16)
    'b11',     # preceding birth interval in months (0 = first-born structural zero)
    'v025',    # residence type: 1=urban, 2=rural
    'v024',    # region code (1–17, all Philippine regions)
    'm14',     # month of first ANC visit: 0=no ANC, 1–20=weeks (98/99=DK → filter)
    'm45',     # iron supplementation: 1=yes, 0=no (8=DK → filter out)
    'm46',     # days took iron supplements (0–270; >270=DHS code → NaN)
    'm1',      # tetanus toxoid injections (0–7; 8/9=DK → NaN)
    'v501',    # marital status (0–5; 9=missing → NaN)
    'v136',    # household members (1–20; 99=missing → NaN)
]

# Feature Columns (13 model inputs — order is fixed and must match training) ─
FEATURE_COLS = [
    'maternal_age',
    'education_yrs',
    'wealth_score',       # v191 raw factor score — XGBoost needs no normalization
    'birth_order',
    'birth_interval',     # 0 for first-borns (structural zero, not missing)
    'residence_type',
    'region',
    'anc_first_timing',   # m14: 0=no ANC, 1–9=month of first visit
    'iron_supplement',
    'iron_days',          # 0 if iron_supplement != 1
    'tetanus_shots',
    'marital_status',
    'household_size',
]

COL_RENAME = {
    'v012': 'maternal_age',
    'v133': 'education_yrs',
    'v191': 'wealth_score',    # Raw DHS factor score — NOT a 0–5 normalized index
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
# b20 range in PHKR82FL: 5–10. b20=9 = full-term (9 calendar months).
# m19 DHS codes: 9996 = "size very large/very small", 9998 = missing → excluded by <9000.
GESTATIONAL_AGE_MIN = 9       # b20 >= 9 (full-term only — excludes preterm LBW confounding)
WEIGHT_MAX_GRAMS    = 9000    # m19 < 9000g (excludes DHS coded non-responses)
LBW_THRESHOLD_GRAMS = 2500    # WHO/DOH definition of Low Birth Weight

# Verified Dataset Statistics (from full 8,478-row scan of PHKR82FL.csv) ─
# Informational — used in documentation and thesis defense, not in pipeline logic.
DATASET_STATS = {
    'raw_rows':            8478,
    'rows_after_filters':  1760,
    'lbw_cases':           214,
    'lbw_pct':             12.16,
    'imbalance_ratio':     7.22,    # Normal:LBW
    'unique_mothers':      1669,
    'multi_birth_mothers': 89,      # mothers with >1 birth record → requires GroupShuffleSplit
}

# Splitting 
RANDOM_STATE          = 42
UNSEEN_FRAC           = 0.10    # sealed holdout (never touched until 06_evaluation.py)
TEST_FRAC             = 0.20    # of remaining 90% after unseen removal
CV_FOLDS              = 10      # StratifiedGroupKFold inside training

# Early stopping: 10% of TRAINING data only — test/unseen NEVER used for ES
ES_VALIDATION_FRAC    = 0.10
EARLY_STOPPING_ROUNDS = 30

# XGBoost Hyperparameters 
# IMPORTANT NOTES:
#
# (A) NO SimpleImputer in pipeline.
#     XGBoost handles NaN via sparsity-aware split finding (Chen & Guestrin, 2016).
#     Adding an imputer removes XGBoost's ability to learn optimal NaN directions.
#     SHAP uses tree_path_dependent — also NaN-safe (verified).
#
# (B) eval_metric = 'aucpr'
#     Controls ONLY the early stopping monitor — does NOT change training loss.
#     Training loss is always binary cross-entropy (logloss gradient/hessian).
#     PR-AUC monitoring is correct for imbalanced data: logloss monitoring
#     can show improvement while minority-class recall silently degrades.
#
# (C) scale_pos_weight is NOT set here — set DYNAMICALLY in 04_model_training.py.
#     Formula: n_neg / n_pos ≈ 7.22 (verified from full dataset).
#     Never hardcode — ratio changes with filter settings or dataset versions.
XGB_PARAMS = {
    'n_estimators':     500,       # Upper bound; early stopping selects optimal count
    'max_depth':        3,         # Shallow trees: Cohen's d < 0.20 for most features
                                   # (classes heavily overlap → deep trees overfit)
    'learning_rate':    0.05,      # Conservative; compensated by up to 500 rounds
    'subsample':        0.8,       # Row subsampling per tree (variance reduction)
    'colsample_bytree': 0.8,       # Feature subsampling per tree (regularization)
    'min_child_weight': 5,         # No split on < 5 samples (minority-class guard)
    'reg_alpha':        0.1,       # L1 regularization (sparsity)
    'reg_lambda':       2.0,       # L2 regularization (large-weight prevention)
    'eval_metric':      'aucpr',   # PR-AUC monitoring for early stopping
    'random_state':     RANDOM_STATE,
    'verbosity':        0,
    'n_jobs':           -1,
    # scale_pos_weight → set dynamically in 04_model_training.py
}

# Threshold Selection 
# Selected on OOF probabilities (training-derived) ONLY.
# Saved as threshold.pkl — separate from model.pkl.
# Clinical rationale: FN (missed LBW) = preventable harm; FP = extra referral only.
MIN_RECALL_FLOOR = 0.50   # System below this provides no value over equal monitoring

# Layer 2 Clinical Thresholds (WHO/DOH) 
# BP  sources: WHO Guidelines Hypertension in Pregnancy (2023) + DOH AO 2022-0012
# MUAC source: WHO adult MUAC cutoff + DOH National Nutrition Program
# These are BINARY flags — NOT probability weights. ML probability is NEVER modified.
CLINICAL_THRESHOLDS = {
    'bp_systolic_critical':  140,    # mmHg — hypertension in pregnancy
    'bp_diastolic_critical':  90,    # mmHg
    'bp_systolic_warning':   130,    # mmHg — Stage 1 elevated
    'bp_diastolic_warning':   80,    # mmHg
    'muac_critical_cm':      23.5,   # cm — maternal undernutrition
    'muac_warning_cm':       25.0,   # cm — borderline nutritional status
}


#  NDHS Realistic Ranges (used by 09_database_seed.py for input validation) ─
#   1. wealth_score: WAS (-2.5, 4.0) → FIXED (-300000, 300000)            
#      v191 is the RAW DHS wealth factor score (large integers).           
#      It is NOT normalized. Actual range: -281,446 to +281,643.           
#      Realistic values (verified percentiles from full dataset):           
#        Poor:         ≈ -70,000  (25th percentile)                        
#        Lower-mid:    ≈  -3,000  (50th percentile / median)              
#        Upper-mid:    ≈  65,000  (75th percentile)                        
#        Wealthy:      ≈ 127,000  (90th percentile)                        
#      USE THESE VALUES in seed data — not -0.5, 3.2, etc.                

NDHS_RANGES = {
    'maternal_age':     (15, 49),           # DHS eligible respondent age range
    'education_yrs':    (0, 20),            # Confirmed max in dataset
    'wealth_score':     (-300000, 300000),  # Raw factor score (e.g., -70000 = poor)
    'birth_order':      (1, 16),            # Confirmed max in dataset
    'birth_interval':   (0, 120),           # months; 0 = first-born (structural)
    'residence_type':   (1, 2),             # 1=urban, 2=rural
    'region':           (1, 17),            # All 17 Philippine regions
    'anc_first_timing': (0, 9),             # LOCKED TO 9. 
    'iron_supplement':  (0, 1),             # 0=no, 1=yes
    'iron_days':        (0, 270),           # 0 if no iron given; max 270
    'tetanus_shots':    (0, 7),             # 8/9=DK already replaced with NaN
    'marital_status':   (0, 5),             # 0=never married, 1=married, etc.
    'household_size':   (1, 20),            # 99=missing already NaN'd
}

#  Valid Philippine Region Codes 
VALID_REGIONS = list(range(1, 18))   # 1–17 inclusive (all confirmed in 2022 NDHS)

REGION_LABELS = {
    1:  'Ilocos Region (I)',
    2:  'Cagayan Valley (II)',
    3:  'Central Luzon (III)',
    4:  'CALABARZON (IV-A)',
    5:  'MIMAROPA (IV-B)',
    6:  'Bicol Region (V)',
    7:  'Western Visayas (VI)',
    8:  'Central Visayas (VII)',
    9:  'Eastern Visayas (VIII)',
    10: 'Zamboanga Peninsula (IX)',
    11: 'Northern Mindanao (X)',
    12: 'Davao Region (XI)',
    13: 'SOCCSKSARGEN (XII)',
    14: 'NCR (National Capital Region)',
    15: 'CAR (Cordillera Administrative Region)',
    16: 'BARMM (Bangsamoro Autonomous Region)',
    17: 'Caraga (XIII)',
}

RESIDENCE_LABELS = {
    1: 'Urban',
    2: 'Rural'
}

IRON_LABELS = {
    0: 'No (Did not receive)',
    1: 'Yes (Received)'
}

# Based on standard DHS v501 coding
MARITAL_LABELS = {
    0: 'Never married',
    1: 'Married',
    2: 'Living together',
    3: 'Widowed',
    4: 'Divorced',
    5: 'Separated'
}