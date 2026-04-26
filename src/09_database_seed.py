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
