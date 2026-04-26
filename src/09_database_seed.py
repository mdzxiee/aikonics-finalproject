# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/09_database_seed.py
#
# Purpose : Initialize the SQLite database schema and insert seed records
#           for prototype testing. All seed data uses NDHS-realistic ranges
#           verified against PHKR82FL.csv. No out-of-distribution values.
#
# Connects to: prototype/app.py (live usage), prototype/predictor.py
# --------------------------------------------------------------------

import os, sys, json, sqlite3, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prototype'))

from config import (
    DB_PATH, NDHS_RANGES, PROTO_DIR, FEATURE_COLS
)

# Database Initialization 

def initialize_db() -> sqlite3.Connection:
    """Create all tables from schema.sql. Safe to run multiple times (IF NOT EXISTS)."""
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'database', 'schema.sql'
    )
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql)
    conn.commit()
    print(f"  [DB] Schema initialized: {DB_PATH}")
    return conn

# Validate Seed Data 

def validate_seed_record(record: dict) -> None:
    """Validate a seed record against NDHS_RANGES before insertion."""
    layer1 = record['layer1']
    for feat, val in layer1.items():
        if feat not in NDHS_RANGES or val is None:
            continue
        lo, hi = NDHS_RANGES[feat]
        if not (lo <= val <= hi):
            raise ValueError(
                f"Seed data OUT OF NDHS RANGE: {feat}={val} "
                f"(expected {lo}–{hi}). Fix the seed record."
            )

    if layer1.get('birth_order', 2) == 1 and layer1.get('birth_interval', 0) != 0:
        raise ValueError("First-born (birth_order=1) must have birth_interval=0.")

    if layer1.get('iron_supplement', 0) != 1 and layer1.get('iron_days', 0) > 0:
        raise ValueError("iron_days > 0 requires iron_supplement = 1.")

    anc = layer1.get('anc_first_timing', 0)
    if not (0 <= anc <= 9):
        raise ValueError(
            f"anc_first_timing={anc} is out of valid range (0–9). "
            f"Values 98/99 are DHS codes and must NOT be used in seed data."
        )

# Seed Records 

def get_seed_records() -> list:
    """Three clinically representative seed records."""
    return [
        {
            'patient': {
                'full_name':    'Maria Santos',
                'barangay':     'Brgy. San Pedro',
                'municipality': 'Iriga City',
                'region':       5,   
            },
            'layer1': {
                'maternal_age':     28,
                'education_yrs':    4,     
                'wealth_score':   -120000,   
                'birth_order':      5,
                'birth_interval':  18,     
                'residence_type':   2,     
                'region':           5,     
                'anc_first_timing': 7,     
                'iron_supplement':  1,
                'iron_days':       30,
                'tetanus_shots':    1,
                'marital_status':   1,
                'household_size':   8,
            },
            'layer2': {
                'bp_systolic':    118,    
                'bp_diastolic':    76,
                'muac_cm':        24.5,   
                'weight_kg':      50,
                'height_cm':     152,
                'gestational_weeks': 28,
            },
            'expected_tier': 'HIGH',
            'expected_escalated': False,
        },
        {
            'patient': {
                'full_name':    'Rosario Dela Cruz',
                'barangay':     'Brgy. Sta. Cruz',
                'municipality': 'Ligao City',
                'region':       5,   
            },
            'layer1': {
                'maternal_age':     32,
                'education_yrs':    6,     
                'wealth_score':    -35000,
                'birth_order':      3,
                'birth_interval':  18,
                'residence_type':   2,     
                'region':           5,
                'anc_first_timing': 5,     
                'iron_supplement':  1,
                'iron_days':       30,
                'tetanus_shots':    1,
                'marital_status':   1,
                'household_size':   7,
            },
            'layer2': {
                'bp_systolic':    138,    
                'bp_diastolic':    88,    
                'muac_cm':        22.0,   
                'weight_kg':      52,
                'height_cm':     155,
                'gestational_weeks': 24,
            },
            'expected_tier': 'MEDIUM',
            'expected_escalated': True,
            'expected_flag': 'MATERNAL_UNDERNUTRITION',
        },
        {
            'patient': {
                'full_name':    'Angela Reyes',
                'barangay':     'Brgy. Addition Hills',
                'municipality': 'Mandaluyong City',
                'region':       13,  
            },
            'layer1': {
                'maternal_age':     25,
                'education_yrs':   16,     
                'wealth_score':     145000,   
                'birth_order':      1,     
                'birth_interval':   0,     
                'residence_type':   1,     
                'region':          13,
                'anc_first_timing': 2,     
                'iron_supplement':  1,
                'iron_days':      180,
                'tetanus_shots':    2,
                'marital_status':   1,
                'household_size':   3,
            },
            'layer2': {
                'bp_systolic':    110,
                'bp_diastolic':    70,
                'muac_cm':        27.0,   
                'weight_kg':      58,
                'height_cm':     160,
                'gestational_weeks': 20,
            },
            'expected_tier': 'LOW',
            'expected_escalated': False,
        },
    ]

#  Seed BHW User 

def seed_bhw_user(conn: sqlite3.Connection) -> int:
    cur = conn.execute("""
        INSERT OR IGNORE INTO bhw_users
            (bhw_id, full_name, barangay, municipality, region, contact_no)
        VALUES (1, 'BHW Mildred Santos', 'Brgy. Sta. Cruz',
                'Ligao City', 'Bicol', '0917-555-0001')
    """)
    conn.commit()
    print(f"  [SEED] BHW user: Mildred Santos (bhw_id=1)")
    return 1

# Run Predictions and Insert Assessments 

def seed_assessments(conn: sqlite3.Connection, bhw_id: int) -> None:
    """Run each seed record through the predictor and clinical flags."""
    proto_dir = PROTO_DIR
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)

    try:
        from predictor      import get_predictor
        from clinical_flags import apply_decision_fusion
    except ImportError as e:
        print(f"  [SKIP] Could not import prototype modules: {e}")
        return

    predictor = get_predictor()
    records   = get_seed_records()

    for i, rec in enumerate(records, start=1):
        try:
            validate_seed_record(rec)
        except ValueError as e:
            print(f"  [SEED] Case {i} REJECTED: {e}")
            continue

        ml_result = predictor.predict(rec['layer1'])

        l2 = rec['layer2']
        fusion = apply_decision_fusion(
            ml_probability = ml_result['ml_probability'],
            bp_systolic    = l2.get('bp_systolic'),
            bp_diastolic   = l2.get('bp_diastolic'),
            muac_cm        = l2.get('muac_cm'),
        )
        result = {**fusion, 'shap_top_features': ml_result['shap_top_features']}

        expected_tier = rec.get('expected_tier')
        actual_tier   = ml_result['ml_risk_tier']
        expected_esc  = rec.get('expected_escalated')
        actual_esc    = result['escalated']

        if not ((expected_tier is None) or (actual_tier == expected_tier)):
            print(f"  [SEED] Case {i} TIER MISMATCH: expected {expected_tier}, got {actual_tier}")
        if not ((expected_esc is None) or (actual_esc == expected_esc)):
            print(f"  [SEED] Case {i} ESCALATION MISMATCH: expected {expected_esc}, got {actual_esc}")
