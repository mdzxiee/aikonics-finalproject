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
        }
    ]
