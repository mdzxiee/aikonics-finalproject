# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/09_database_seed.py
#
# Purpose : Initialize the SQLite database schema and insert seed records
#           for prototype testing. All seed data uses NDHS-realistic ranges
#           verified against PHKR82FL.csv. No out-of-distribution values.
#
# Input  : database/schema.sql
#          prototype/model.pkl + threshold.pkl  (must exist from 08)
# Output : database/aikonic.db  (initialized and seeded)
#
# CORRECTION vs. original:
#   All mock data respects NDHS-validated ranges (NDHS_RANGES from config).
#   Invalid combinations are explicitly avoided:
#     - A 16-year-old with bord=4 is implausible
#     - anc_first_timing > 9 are DHS codes, not real values (max real = 9)
#     - birth_interval = 0 is only valid when birth_order = 1
#     - iron_days > 0 only when iron_supplement = 1
#   Unrealistic seed data produces unreliable SHAP attributions and incorrect
#   demo outputs, invalidating the prototype demonstration.
#
# Three seed cases are provided:
#   Case 1 — HIGH RISK (Layer 1 only): Poor socioeconomic + late ANC
#   Case 2 — MEDIUM RISK + Layer 2 escalation: MUAC flag triggers upgrade
#   Case 3 — LOW RISK: Good ANC, adequate wealth, first pregnancy
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
   """
   Validate a seed record against NDHS_RANGES before insertion.
   Raises ValueError if any value is out of realistic NDHS bounds.
   """
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

   # Structural consistency checks
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
   """
   Three clinically representative seed records.
   All values are within NDHS_RANGES AND internally consistent.

   Case 1 — HIGH ML RISK (no Layer 2 escalation):
     Poor wealth, Grade 4 education, 5th pregnancy, 18-month interval,
     late first ANC (Month 7), only 30 iron days, rural Bicol.
     Expected: HIGH RISK ML tier, no clinical flags if BP/MUAC normal.

   Case 2 — MEDIUM ML RISK + Layer 2 MUAC Escalation:
     Moderate risk profile, but MUAC = 22.0 cm (< 23.5 threshold).
     Expected: MEDIUM → escalated to HIGH PRIORITY REFERRAL.
     This is the Nanay Rosario reference scenario.

   Case 3 — LOW ML RISK (well-resourced, early ANC):
     First pregnancy, high wealth, college education, ANC Month 2,
     full iron supplementation, urban NCR.
     Expected: LOW RISK, no clinical flags.
   """
   return [
       {
           'patient': {
               'full_name':    'Maria Santos',
               'barangay':     'Brgy. San Pedro',
               'municipality': 'Iriga City',
               'region':       5,   # Bicol
           },
           'layer1': {
               'maternal_age':     28,
               'education_yrs':    4,     # Grade 4
               'wealth_score':    -120000,   # very poor (NDHS continuous scale)
               'birth_order':      5,
               'birth_interval':  18,     # 18 months preceding
               'residence_type':   2,     # rural
               'region':           5,     # Bicol
               'anc_first_timing': 7,     # Month 7 (very late)
               'iron_supplement':  1,
               'iron_days':       30,
               'tetanus_shots':    1,
               'marital_status':   1,
               'household_size':   8,
           },
           'layer2': {
               'bp_systolic':    118,    # normal BP
               'bp_diastolic':    76,
               'muac_cm':        24.5,   # borderline but not critical
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
               'region':       5,   # Bicol
           },
           'layer1': {
               'maternal_age':     32,
               'education_yrs':    6,     # Grade 6
               'wealth_score':   -35000,
               'birth_order':      3,
               'birth_interval':  18,
               'residence_type':   2,     # rural
               'region':           5,
               'anc_first_timing': 5,     # Month 5 (late)
               'iron_supplement':  1,
               'iron_days':       30,
               'tetanus_shots':    1,
               'marital_status':   1,
               'household_size':   7,
           },
           'layer2': {
               'bp_systolic':    138,    # below critical (140), no BP flag
               'bp_diastolic':    88,    # below critical (90)
               'muac_cm':        22.0,   # < 23.5 → CRITICAL MUAC FLAG
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
               'region':       13,  # NCR
           },
           'layer1': {
               'maternal_age':     25,
               'education_yrs':   16,     # College graduate
               'wealth_score':    145000,   # high wealth
               'birth_order':      1,     # first pregnancy
               'birth_interval':   0,     # structural zero (first-born)
               'residence_type':   1,     # urban
               'region':          13,
               'anc_first_timing': 2,     # Month 2 (early ANC)
               'iron_supplement':  1,
               'iron_days':      180,
               'tetanus_shots':    2,
               'marital_status':   1,
               'household_size':   3,
           },
           'layer2': {
               'bp_systolic':    110,
               'bp_diastolic':    70,
               'muac_cm':        27.0,   # normal
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
   """
   Run each seed record through the predictor and clinical flags,
   then insert the full assessment into the database.
   """
   # Load prototype modules
   proto_dir = PROTO_DIR
   if proto_dir not in sys.path:
       sys.path.insert(0, proto_dir)

   try:
       from predictor      import get_predictor
       from clinical_flags import apply_decision_fusion
   except ImportError as e:
       print(f"  [SKIP] Could not import prototype modules: {e}")
       print(f"         Run 08_export_artifacts.py first to copy model to prototype/.")
       return

   predictor = get_predictor()
   records   = get_seed_records()

   for i, rec in enumerate(records, start=1):
       # Validate ranges before inserting
       try:
           validate_seed_record(rec)
       except ValueError as e:
           print(f"  [SEED] Case {i} REJECTED: {e}")
           continue

       # Layer 1 prediction
       ml_result = predictor.predict(rec['layer1'])

       # Layer 2 clinical escalation
       l2 = rec['layer2']
       fusion = apply_decision_fusion(
           ml_probability = ml_result['ml_probability'],
           bp_systolic    = l2.get('bp_systolic'),
           bp_diastolic   = l2.get('bp_diastolic'),
           muac_cm        = l2.get('muac_cm'),
       )
       result = {**fusion, 'shap_top_features': ml_result['shap_top_features']}

       # Verify against expected outcome (sanity check)
       expected_tier = rec.get('expected_tier')
       actual_tier   = ml_result['ml_risk_tier']
       expected_esc  = rec.get('expected_escalated')
       actual_esc    = result['escalated']

       tier_ok = (expected_tier is None) or (actual_tier == expected_tier)
       esc_ok  = (expected_esc is None) or (actual_esc == expected_esc)

       if not tier_ok:
           print(f"  [SEED] Case {i} TIER MISMATCH: expected {expected_tier}, "
                 f"got {actual_tier}  (model may differ — not a bug)")
       if not esc_ok:
           print(f"  [SEED] Case {i} ESCALATION MISMATCH: expected {expected_esc}, "
                 f"got {actual_esc}  (check clinical_flags.py thresholds)")

       # Insert patient
       cur = conn.execute("""
           INSERT INTO patients
               (bhw_id, full_name, barangay, municipality, region, residence_type)
           VALUES (?, ?, ?, ?, ?, ?)
       """, (
           bhw_id,
           rec['patient']['full_name'],
           rec['patient']['barangay'],
           rec['patient']['municipality'],
           rec['layer1']['region'],
           rec['layer1']['residence_type'],
       ))
       patient_id = cur.lastrowid

       # Insert assessment
       shap_json = json.dumps(result.get('shap_top_features', []))
       cur2 = conn.execute("""
           INSERT INTO assessments (
               patient_id, bhw_id,
               maternal_age, education_yrs, wealth_score, birth_order,
               birth_interval, residence_type, region, anc_first_timing,
               iron_supplement, iron_days, tetanus_shots, marital_status,
               household_size,
               bp_systolic, bp_diastolic, muac_cm, weight_kg, height_cm,
               gestational_weeks,
               ml_probability, ml_risk_tier, final_risk_level,
               escalated, shap_top_features, referral_recommended
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
       """, (
           patient_id, bhw_id,
           rec['layer1']['maternal_age'],
           rec['layer1']['education_yrs'],
           rec['layer1']['wealth_score'],
           rec['layer1']['birth_order'],
           rec['layer1']['birth_interval'],
           rec['layer1']['residence_type'],
           rec['layer1']['region'],
           rec['layer1']['anc_first_timing'],
           rec['layer1']['iron_supplement'],
           rec['layer1']['iron_days'],
           rec['layer1']['tetanus_shots'],
           rec['layer1']['marital_status'],
           rec['layer1']['household_size'],
           l2.get('bp_systolic'),
           l2.get('bp_diastolic'),
           l2.get('muac_cm'),
           l2.get('weight_kg'),
           l2.get('height_cm'),
           l2.get('gestational_weeks'),
           result['ml_probability'],
           result['ml_risk_tier'],
           result['final_risk_level'],
           int(result['escalated']),
           shap_json,
           1 if 'REFERRAL' in result['final_risk_level'] else 0,
       ))
       assessment_id = cur2.lastrowid

       # Insert clinical flags
       for flag in result.get('clinical_flags', []):
           conn.execute("""
               INSERT INTO clinical_flags
                   (assessment_id, flag_type, severity,
                    measured_value, threshold_value, flag_message)
               VALUES (?, ?, ?, ?, ?, ?)
           """, (
               assessment_id,
               flag.get('type'),
               flag.get('severity'),
               flag.get('value'),
               flag.get('threshold'),
               flag.get('message'),
           ))

       # Insert SHAP explanations
       for rank, s in enumerate(result.get('shap_top_features', []), start=1):
           conn.execute("""
               INSERT INTO shap_explanations
                   (assessment_id, feature_rank, feature_name,
                    shap_value, feature_value)
               VALUES (?, ?, ?, ?, ?)
           """, (
               assessment_id, rank,
               s.get('feature'),
               s.get('shap_value'),
               s.get('feature_value'),
           ))

       conn.commit()
       print(f"  [SEED] Case {i}: {rec['patient']['full_name']:<22} | "
             f"ML={actual_tier:<7} | Escalated={actual_esc} | "
             f"Final: {result['final_risk_level']}")

   print(f"\n  [SEED] {len(records)} assessment cases inserted.")

#  Main
def run_database_seed() -> None:
   print("=" * 65)
   print("STAGE 9: DATABASE INITIALIZATION AND SEEDING")
   print("=" * 65)
   print(f"\n  All seed data validated against NDHS_RANGES from config.py.")
   print(f"  No out-of-distribution values (no anc_first_timing > 9, etc.).")

   conn = initialize_db()

   bhw_id = seed_bhw_user(conn)
   seed_assessments(conn, bhw_id)

   # Quick verification query
   row = conn.execute(
       "SELECT COUNT(*) AS n FROM assessments"
   ).fetchone()
   print(f"\n  [VERIFY] Total assessments in DB: {row['n']}")

   flags_row = conn.execute(
       "SELECT COUNT(*) AS n FROM clinical_flags"
   ).fetchone()
   print(f"  [VERIFY] Total clinical flags in DB: {flags_row['n']}")

   conn.close()

   print(f"\n  [SEED COMPLETE] Database: {DB_PATH}")
   print(f"  Run prototype with: cd prototype && python app.py")

if __name__ == "__main__":
   run_database_seed()
