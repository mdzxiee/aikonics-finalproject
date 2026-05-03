# Purpose : Initialize the SQLite database schema and insert 3 seed
#           assessment cases for prototype testing and demonstration.
#
# Input  : database/schema.sql
#          prototype/model.pkl + threshold.pkl  (from 08_export_artifacts.py)
#
# Output : database/aikonic.db  (initialized and seeded)
#
# SEED DATA DESIGN RULES
#   All seed data is validated against NDHS_RANGES from config.py before
#   insertion. The following rules are strictly enforced:
#
#   1. wealth_score must be a RAW v191 factor score (large integers):
#      Poor      ≈ -70,000  (25th percentile of PHKR82FL.csv)
#      Median    ≈  -3,000  (50th percentile)
#      Comfortable ≈ 65,000 (75th percentile)
#      NOT -0.5, 3.2, or any normalized 0-5 value.
#
#   2. anc_first_timing range: 0–20 (0=no ANC, verified from PHKR82FL.csv)
#      NOT 0-9 as previously documented.
#
#   3. birth_interval = 0 ONLY when birth_order = 1 (first-born structural zero)
#
#   4. iron_days > 0 ONLY when iron_supplement = 1
#
# THREE SEED CASES 
#   Case 1 — HIGH ML risk, clinically stable → DE-ESCALATION demo
#     ML flags HIGH but BP and MUAC are normal
#     Expected: ELEVATED MONITORING (not RHU referral)
#     Demonstrates the dual-purpose Layer 2 de-escalation pathway
#
#   Case 2 — MEDIUM ML risk, MUAC critical → ESCALATION demo (Nanay Rosario)
#     ML below threshold (0.38) but MUAC confirmed < 23.5 cm
#     Expected: HIGH PRIORITY REFERRAL (Layer 2 escalated)
#     Demonstrates the escalation pathway overriding a below-threshold ML score
#
#   Case 3 — LOW ML risk, clinically normal → ROUTINE MONITORING demo
#     Good socioeconomic profile, early ANC, normal BP and MUAC
#     Expected: ROUTINE MONITORING
#
# Connects to: prototype/app.py (live usage after seeding)
# ===========================================================================

import os, sys, json, sqlite3, warnings
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')

# Path setup
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SRC_DIR)
_PROTO_DIR = os.path.join(_ROOT_DIR, 'prototype')

sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _PROTO_DIR)

from config import DB_PATH, NDHS_RANGES, PROTO_DIR

# SECTION 1 — DATABASE INITIALIZATION

def get_connection() -> sqlite3.Connection:
    """Open SQLite connection with FK enforcement and WAL mode."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_db() -> None:
    """Execute schema.sql to create all tables. Safe to call multiple times."""
    schema_path = os.path.join(_ROOT_DIR, 'database', 'schema.sql')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"schema.sql not found at {schema_path}.\n"
            "Ensure the database/ folder exists with schema.sql."
        )
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    with get_connection() as conn:
        conn.executescript(schema_sql)
    print(f"  [DB] Schema initialized: {DB_PATH}")

# SECTION 2 — SEED DATA VALIDATION

def validate_seed_record(record: dict) -> None:
    """
    Validate a seed record against NDHS_RANGES before insertion.
    Raises ValueError if any value is outside documented NDHS bounds.

    This prevents out-of-distribution seed data that would produce
    unreliable SHAP attributions and incorrect demo outputs.
    """
    layer1 = record['layer1']

    # Range checks
    for feat, val in layer1.items():
        if feat not in NDHS_RANGES or val is None:
            continue
        lo, hi = NDHS_RANGES[feat]
        if not (lo <= val <= hi):
            raise ValueError(
                f"Seed data OUT OF NDHS RANGE: {feat}={val} "
                f"(valid range: {lo}–{hi})\n"
                f"Fix the seed record to use realistic NDHS values."
            )

    # Structural consistency checks
    bo = layer1.get('birth_order', 2)
    bi = layer1.get('birth_interval', 0)
    if bo == 1 and bi != 0:
        raise ValueError(
            f"First-born (birth_order=1) must have birth_interval=0. "
            f"Got birth_interval={bi}."
        )
    if bo > 1 and bi == 0:
        raise ValueError(
            f"Non-first-born (birth_order={bo}) cannot have birth_interval=0. "
            f"That is reserved for first-borns only."
        )

    iron_s = layer1.get('iron_supplement', 0)
    iron_d = layer1.get('iron_days', 0)
    if iron_s != 1 and iron_d > 0:
        raise ValueError(
            f"iron_days={iron_d} > 0 requires iron_supplement=1."
        )

    anc = layer1.get('anc_first_timing', 0)
    if not (0 <= anc <= 20):
        raise ValueError(
            f"anc_first_timing={anc} is outside valid range (0–20).\n"
            f"0=no ANC; 1-9=typical; 10-20=very late. Values 98/99 are DHS codes."
        )

    ws = layer1.get('wealth_score', 0)
    if -10 <= ws <= 10 and ws != 0:
        raise ValueError(
            f"wealth_score={ws} looks like a normalized 0-5 value.\n"
            f"Use raw v191 factor scores: poor≈-70000, median≈-3000, rich≈65000."
        )

# SECTION 3 — SEED RECORDS

def get_seed_records() -> list:
    """
    Three clinically representative seed records.

    All wealth_score values are raw v191 DHS factor scores (large integers)
    verified against PHKR82FL.csv percentiles:
      Poor:         ≈ -70,000  (p25)
      Median:       ≈  -3,000  (p50)
      Comfortable:  ≈  65,000  (p75)

    All anc_first_timing values are in range 0–20 (verified from dataset).
    All birth_interval = 0 only when birth_order = 1.
    """
    return [
        # ──────────────────────────────────────────────────────────────────
        # CASE 1 — HIGH ML RISK, CLINICALLY STABLE → DE-ESCALATION
        # Demonstrates: ML flags as HIGH, but Layer 2 finds normal BP + MUAC
        # Final expected: ELEVATED MONITORING (de-escalated from ML referral)
        # ──────────────────────────────────────────────────────────────────
        {
            'patient': {
                'full_name':    'Maria Santos',
                'barangay':     'Brgy. San Pedro',
                'municipality': 'Iriga City',
            },
            'layer1': {
                'maternal_age':     28,
                'education_yrs':    4,       # Grade 4
                'wealth_score':     -70000,  # Poor (p25 of PHKR82FL)
                'birth_order':      5,
                'birth_interval':   18,      # 18 months since last birth
                'residence_type':   2,       # Rural
                'region':           6,       # Bicol (Region V)
                'anc_first_timing': 7,       # Month 7 — very late ANC
                'iron_supplement':  1,
                'iron_days':        30,
                'tetanus_shots':    1,
                'marital_status':   1,
                'household_size':   8,
            },
            'layer2': {
                'bp_systolic_r1':  118,   # Normal — no Rule of 3 needed
                'bp_diastolic_r1':  76,
                'muac_r1':         24.5,  # Borderline but above 23.5 — no flag
                'muac_r2':         None,  # Not re-measured (not triggered)
                'weight_kg':        50,
                'height_cm':       152,
                'gestational_weeks': 28,
            },
            'expected': {
                'ml_tier':    'HIGH',     # Poor wealth + late ANC + high parity
                'escalated':  False,      # No clinical danger signs
                'de_escalated': True,     # ML-flagged but clinically stable
                'final_contains': 'ELEVATED MONITORING',
            },
        },

        # ──────────────────────────────────────────────────────────────────
        # CASE 2 — MEDIUM ML RISK, MUAC CRITICAL → ESCALATION
        # Reference scenario: Nanay Rosario from project analogy
        # ML probability = ~0.38 (MEDIUM, below threshold)
        # Layer 2 MUAC confirmed critical → escalates to HIGH PRIORITY REFERRAL
        # ──────────────────────────────────────────────────────────────────
        {
            'patient': {
                'full_name':    'Rosario Dela Cruz',
                'barangay':     'Brgy. Sta. Cruz',
                'municipality': 'Ligao City',
            },
            'layer1': {
                'maternal_age':     32,
                'education_yrs':    6,      # Grade 6
                'wealth_score':     -3000,  # Median (p50 of PHKR82FL)
                'birth_order':      3,
                'birth_interval':   18,
                'residence_type':   2,      # Rural
                'region':           6,      # Bicol
                'anc_first_timing': 5,      # Month 5 — late ANC
                'iron_supplement':  1,
                'iron_days':        30,
                'tetanus_shots':    1,
                'marital_status':   1,
                'household_size':   7,
            },
            'layer2': {
                'bp_systolic_r1':  138,   # Below 140/90 — no BP flag
                'bp_diastolic_r1':  88,
                'muac_r1':         22.0,  # < 23.5 → re-measure triggered
                'muac_r2':         22.1,  # Also < 23.5 → CONFIRMED CRITICAL
                'weight_kg':        52,
                'height_cm':       155,
                'gestational_weeks': 24,
            },
            'expected': {
                'ml_tier':       'MEDIUM',
                'above_threshold': False,  # 0.38 < threshold
                'escalated':     True,     # Layer 2 MUAC critical
                'final_contains': 'HIGH PRIORITY REFERRAL',
            },
        },

        # ──────────────────────────────────────────────────────────────────
        # CASE 3 — LOW ML RISK, CLINICALLY NORMAL → ROUTINE MONITORING
        # Well-resourced mother, early ANC, first pregnancy, normal vitals
        # Final expected: ROUTINE MONITORING
        # ──────────────────────────────────────────────────────────────────
        {
            'patient': {
                'full_name':    'Angela Reyes',
                'barangay':     'Brgy. Addition Hills',
                'municipality': 'Mandaluyong City',
            },
            'layer1': {
                'maternal_age':     25,
                'education_yrs':    16,     # College graduate
                'wealth_score':     65000,  # Comfortable (p75 of PHKR82FL)
                'birth_order':      1,      # First pregnancy
                'birth_interval':   0,      # Structural zero — first-born
                'residence_type':   1,      # Urban
                'region':           14,     # NCR
                'anc_first_timing': 2,      # Month 2 — early ANC 
                'iron_supplement':  1,
                'iron_days':        180,
                'tetanus_shots':    2,
                'marital_status':   1,
                'household_size':   3,
            },
            'layer2': {
                'bp_systolic_r1':  110,   # Normal
                'bp_diastolic_r1':  70,
                'muac_r1':         27.0,  # Normal (above 25.0)
                'muac_r2':         None,
                'weight_kg':        58,
                'height_cm':       160,
                'gestational_weeks': 20,
            },
            'expected': {
                'ml_tier':    'LOW',
                'escalated':  False,
                'final_contains': 'ROUTINE MONITORING',
            },
        },
    ]

# SECTION 4 — BHW USER SEED

def seed_bhw_user(conn: sqlite3.Connection) -> int:
    """Insert the demo BHW user if not already present."""
    existing = conn.execute(
        "SELECT bhw_id FROM bhw_users WHERE bhw_id = 1"
    ).fetchone()
    if existing:
        print(f"  [SEED] BHW user already exists (bhw_id=1) — skipping.")
        return 1

    conn.execute("""
        INSERT INTO bhw_users
            (bhw_id, full_name, barangay, municipality, region, contact_no)
        VALUES (1, 'BHW Mildred Santos', 'Brgy. Sta. Cruz',
                'Ligao City', 'Bicol', '0917-555-0001')
    """)
    conn.commit()
    print(f"  [SEED] BHW user created: Mildred Santos (bhw_id=1)")
    return 1

# SECTION 5 — PREDICTION + INSERTION

def seed_assessments(conn: sqlite3.Connection, bhw_id: int) -> None:
    """
    Run each seed record through the predictor and clinical flags,
    then insert the full assessment into the database.

    Uses the production Layer 2 (clinical_flags.py) — not the academic
    simulation from 06_evaluation.py.
    """
    try:
        from predictor      import get_predictor
        from clinical_flags import run_full_clinical_assessment
    except ImportError as e:
        print(f"  [SKIP] Could not import prototype modules: {e}")
        print(f"         Ensure 08_export_artifacts.py has been run.")
        return

    predictor = get_predictor()
    records   = get_seed_records()

    for i, rec in enumerate(records, start=1):
        print(f"\n  {'─'*60}")
        print(f"  CASE {i}: {rec['patient']['full_name']}")
        print(f"  {'─'*60}")

        # Validate before anything else
        try:
            validate_seed_record(rec)
            print(f"  [VALIDATE] NDHS ranges ✓ | Structural consistency ✓")
        except ValueError as e:
            print(f"  [REJECTED] Case {i}: {e}")
            continue

        # Layer 1: ML prediction
        ml_result = predictor.predict(rec['layer1'])

        # Layer 2: Production clinical assessment (dual-purpose)
        l2 = rec['layer2']
        l2_result = run_full_clinical_assessment(
            ml_probability   = ml_result['ml_probability'],
            bp_systolic_r1   = l2.get('bp_systolic_r1'),
            bp_diastolic_r1  = l2.get('bp_diastolic_r1'),
            bp_systolic_r2   = l2.get('bp_systolic_r2'),
            bp_diastolic_r2  = l2.get('bp_diastolic_r2'),
            bp_systolic_r3   = l2.get('bp_systolic_r3'),
            bp_diastolic_r3  = l2.get('bp_diastolic_r3'),
            muac_r1          = l2.get('muac_r1'),
            muac_r2          = l2.get('muac_r2'),
        )

        full_result = {
            **l2_result,
            'shap_top_features': ml_result['shap_top_features'],
            'above_threshold':   ml_result['above_threshold'],
        }

        # Report
        print(f"  ML probability : {ml_result['ml_probability']:.4f} "
              f"({ml_result['ml_risk_tier']})")
        print(f"  Above threshold: {ml_result['above_threshold']}")
        print(f"  Clinical flags : "
              f"{[f['type'] for f in l2_result.get('clinical_flags', [])]}")
        print(f"  Escalated      : {l2_result['escalated']}")
        print(f"  Final level    : {l2_result['final_risk_level']}")

        # Verify against expected (soft check — model output may vary)
        exp = rec.get('expected', {})
        if 'final_contains' in exp:
            if exp['final_contains'] in l2_result['final_risk_level']:
                print(f"  Expected check : ✓ PASS ({exp['final_contains']})")
            else:
                print(f"  Expected check : ⚠ MISMATCH")
                print(f"    Expected to contain: {exp['final_contains']}")
                print(f"    Got: {l2_result['final_risk_level']}")
                print(f"    (Model parameters affect probability — not a code error)")

        # Insert patient
        cur = conn.execute("""
            INSERT INTO patients
                (bhw_id, full_name, barangay, municipality,
                 region, residence_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bhw_id,
            rec['patient']['full_name'],
            rec['patient']['barangay'],
            rec['patient']['municipality'],
            rec['layer1']['region'],
            rec['layer1']['residence_type'],
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))
        patient_id = cur.lastrowid

        # Build notes string (BP readings for audit trail)
        notes_parts = []
        for r_num in [1, 2, 3]:
            s = l2.get(f'bp_systolic_r{r_num}')
            d = l2.get(f'bp_diastolic_r{r_num}')
            if s is not None and d is not None:
                notes_parts.append(f"BP R{r_num}: {s:.0f}/{d:.0f} mmHg")
        if l2.get('muac_r2') is not None:
            notes_parts.append(f"MUAC R2: {l2['muac_r2']:.1f} cm")
        notes = ' | '.join(notes_parts) if notes_parts else None

        # Insert assessment
        shap_json = json.dumps(full_result.get('shap_top_features', []))
        cur2 = conn.execute("""
            INSERT INTO assessments (
                patient_id, bhw_id, assessment_date,
                maternal_age, education_yrs, wealth_score,
                birth_order, birth_interval, residence_type,
                region, anc_first_timing, iron_supplement,
                iron_days, tetanus_shots, marital_status, household_size,
                bp_systolic, bp_diastolic, muac_cm,
                weight_kg, height_cm, gestational_weeks,
                ml_probability, ml_risk_tier, above_threshold, final_risk_level,
                escalated, de_escalated, shap_top_features,
                referral_recommended, notes
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            patient_id, bhw_id, datetime.now().isoformat(),
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
            l2.get('bp_systolic_r1'),
            l2.get('bp_diastolic_r1'),
            l2.get('muac_r1'),
            l2.get('weight_kg'),
            l2.get('height_cm'),
            l2.get('gestational_weeks'),
            full_result['ml_probability'],
            full_result['ml_risk_tier'],
            int(full_result['above_threshold']),          
            full_result['final_risk_level'],
            int(full_result['escalated']),
            int(full_result.get('de_escalated', 0)),      
            shap_json,
            1 if 'REFERRAL' in full_result['final_risk_level'] else 0,
            notes,
        ))
        assessment_id = cur2.lastrowid

        # Insert clinical flags
        for flag in full_result.get('clinical_flags', []):
            if flag.get('severity', 'none') == 'none':
                continue
            conn.execute("""
                INSERT INTO clinical_flags
                    (assessment_id, flag_type, severity,
                     measured_value, threshold_value, flag_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                assessment_id,
                flag.get('type'),
                flag.get('severity'),
                flag.get('measured_value'),
                flag.get('threshold_value'),
                flag.get('message'),
            ))

        # Insert SHAP explanations
        for rank, sf in enumerate(
                full_result.get('shap_top_features', []), start=1):
            conn.execute("""
                INSERT INTO shap_explanations
                    (assessment_id, feature_rank, feature_name,
                     shap_value, feature_value)
                VALUES (?, ?, ?, ?, ?)
            """, (
                assessment_id, rank,
                sf.get('feature'),
                sf.get('shap_value'),
                sf.get('feature_value'),
            ))

        conn.commit()
        print(f"  Saved as assessment_id={assessment_id}")

# MAIN

def run_database_seed() -> None:
    print("=" * 70)
    print("STAGE 9: DATABASE INITIALIZATION AND SEEDING")
    print("=" * 70)
    print(f"\n  All seed wealth_score values use raw v191 DHS factor scores.")
    print(f"  All anc_first_timing values in range 0–20 (verified).")
    print(f"  All NDHS_RANGES validated before insertion.")
    print(f"  Production Layer 2 (clinical_flags.py) used — not academic sim.")

    # Initialize database
    initialize_db()

    with get_connection() as conn:
        # Seed BHW user
        bhw_id = seed_bhw_user(conn)

        # Seed 3 assessment cases
        seed_assessments(conn, bhw_id)

        # Verification query
        n_assessments = conn.execute(
            "SELECT COUNT(*) AS n FROM assessments"
        ).fetchone()['n']
        n_flags = conn.execute(
            "SELECT COUNT(*) AS n FROM clinical_flags"
        ).fetchone()['n']
        n_shap = conn.execute(
            "SELECT COUNT(*) AS n FROM shap_explanations"
        ).fetchone()['n']

    print(f"\n  {'═'*60}")
    print(f"  STAGE 9 COMPLETE")
    print(f"  Database     : {DB_PATH}")
    print(f"  Assessments  : {n_assessments}")
    print(f"  Clinical flags: {n_flags}")
    print(f"  SHAP entries : {n_shap}")
    print(f"\n  Prototype ready. Start with:")
    print(f"    cd prototype && python app.py")
    print(f"    → http://localhost:5000")
    print(f"  {'═'*60}")


if __name__ == "__main__":
    run_database_seed()