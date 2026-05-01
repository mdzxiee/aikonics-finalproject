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
    'm13',     # month of first ANC visit: 0=no ANC, 1–20=weeks (98/99=DK → filter)
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
    'wealth_score',       
    'birth_order',
    'birth_interval',     
    'residence_type',
    'region',
    'anc_first_timing',   
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
    'm13':  'anc_first_timing',
    'm45':  'iron_supplement',
    'm46':  'iron_days',
    'm1':   'tetanus_shots',
    'v501': 'marital_status',
    'v136': 'household_size',
}

# Validity Filter Thresholds 
GESTATIONAL_AGE_MIN = 9       
WEIGHT_MAX_GRAMS    = 9000    
LBW_THRESHOLD_GRAMS = 2500    

# Verified Dataset Statistics
DATASET_STATS = {
    'raw_rows':            8478,
    'rows_after_filters':  1760,
    'lbw_cases':           214,
    'lbw_pct':             12.16,
    'imbalance_ratio':     7.22,    
    'unique_mothers':      1669,
    'multi_birth_mothers': 89,      
}

# Splitting 
RANDOM_STATE          = 42
UNSEEN_FRAC           = 0.10    
TEST_FRAC             = 0.20    
CV_FOLDS              = 10      

# Early stopping
ES_VALIDATION_FRAC    = 0.10
EARLY_STOPPING_ROUNDS = 30

# XGBoost Hyperparameters 
"""
NOTE: XGBoost parameters have been moved to 04_model_training.py 
inside the `baseline` dictionary to allow for dynamic Systematic 
Parameter Sensitivity Analysis without conflicting single-source-of-truth.
"""

# Threshold Selection 
MIN_RECALL_FLOOR = 0.50   

# Layer 2 Clinical Thresholds (WHO/DOH) 
CLINICAL_THRESHOLDS = {
    'bp_systolic_critical':  140,    
    'bp_diastolic_critical':  90,    
    'bp_systolic_warning':   130,    
    'bp_diastolic_warning':   80,    
    'muac_critical_cm':      23.5,   
    'muac_warning_cm':       25.0,   
}

#  NDHS Realistic Ranges 
NDHS_RANGES = {
    'maternal_age':     (15, 49),           
    'education_yrs':    (0, 20),            
    'wealth_score':     (-300000, 300000),  
    'birth_order':      (1, 16),            
    'birth_interval':   (0, 120),           
    'residence_type':   (1, 2),             
    'region':           (1, 17),            
    'anc_first_timing': (0, 9),             
    'iron_supplement':  (0, 1),             
    'iron_days':        (0, 270),           
    'tetanus_shots':    (0, 7),             
    'marital_status':   (0, 5),             
    'household_size':   (1, 20),            
}

#  Valid Philippine Region Codes 
VALID_REGIONS = list(range(1, 18))   

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

MARITAL_LABELS = {
    0: 'Never married',
    1: 'Married',
    2: 'Living together',
    3: 'Widowed',
    4: 'Divorced',
    5: 'Separated'
}