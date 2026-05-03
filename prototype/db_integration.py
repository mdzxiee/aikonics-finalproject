# Purpose : All SQLite read/write operations for the prototype.
#           Column names and table structure match schema.sql exactly.
#
# SCHEMA COLUMNS REFERENCE (assessments table) 
#   Layer 1 inputs   : maternal_age, education_yrs, wealth_score, birth_order,
#                      birth_interval, residence_type, region, anc_first_timing,
#                      iron_supplement, iron_days, tetanus_shots,
#                      marital_status, household_size
#   Layer 2 inputs   : bp_systolic, bp_diastolic, muac_cm,
#                      weight_kg, height_cm, gestational_weeks
#   System outputs   : ml_probability, ml_risk_tier,
#                      above_threshold  ← NEW: 1 if ML prob >= threshold
#                      final_risk_level,
#                      escalated        ← 1 if L2 upgraded recommendation
#                      de_escalated     ← NEW: 1 if L2 downgraded ML referral
#                      shap_top_features (JSON)
#                      referral_recommended, referral_completed,
#                      referral_outcome, notes
#
#  WHY above_threshold AND de_escalated ARE NEW 
#   These two columns make the dual-purpose fusion auditable:
#   above_threshold=1, escalated=0, de_escalated=1 → ML wanted to refer
#     but Layer 2 found no clinical danger → converted to monitoring.
#     This documents EVERY case where the system prevented an unnecessary
#     RHU referral — answering the operational question: "How many
#     unnecessary visits did the de-escalation pathway prevent?"
#
# Connects to: app.py (called after every prediction)
#              09_database_seed.py (seeding demo records)
# ===========================================================================

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Optional

_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_PROTO_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, 'src'))

from config import DB_PATH

_SCHEMA_PATH = os.path.join(_ROOT_DIR, 'database', 'schema.sql')

# CONNECTION
def get_connection() -> sqlite3.Connection:
    """
    Open SQLite connection with:
      Row factory → dict-like access by column name
      Foreign key enforcement
      WAL journal mode (safer for concurrent reads)
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# INITIALIZATION

def initialize_db() -> None:
    """
    Execute schema.sql to create all tables, views, and indexes.
    Uses CREATE TABLE IF NOT EXISTS — safe to call multiple times.
    """
    if not os.path.exists(_SCHEMA_PATH):
        raise FileNotFoundError(
            f"schema.sql not found at {_SCHEMA_PATH}.\n"
            "Ensure the database/ directory exists."
        )
    with open(_SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    with get_connection() as conn:
        conn.executescript(schema_sql)
    print(f"[DB] Initialized: {DB_PATH}")

# PATIENT — GET OR CREATE

def get_or_create_patient(conn: sqlite3.Connection,
                           bhw_id: int,
                           patient_data: dict,
                           region: int,
                           residence_type: int) -> int:
    """
    Find existing patient by full_name + barangay + municipality + bhw_id,
    or insert a new one. Returns patient_id.
    """
    full_name    = patient_data.get('full_name', 'Unknown')
    barangay     = patient_data.get('barangay', '')
    municipality = patient_data.get('municipality', '')

    existing = conn.execute(
        """
        SELECT patient_id FROM patients
        WHERE full_name = ? AND barangay = ?
          AND municipality = ? AND bhw_id = ?
        LIMIT 1
        """,
        (full_name, barangay, municipality, bhw_id)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE patients SET updated_at = ? WHERE patient_id = ?",
            (datetime.now().isoformat(), existing['patient_id'])
        )
        return existing['patient_id']

    cur = conn.execute(
        """
        INSERT INTO patients
            (bhw_id, full_name, barangay, municipality,
             region, residence_type, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bhw_id, full_name, barangay, municipality,
            region, residence_type,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        )
    )
    return cur.lastrowid

# SAVE ASSESSMENT

def save_assessment(bhw_id: int,
                    patient_data: dict,
                    layer1: dict,
                    layer2: dict,
                    result: dict) -> int:
    """
    Save a complete assessment session to the database.

    Inserts into: patients (get_or_create), assessments, clinical_flags,
                  shap_explanations.

    result dict is the merged output from predictor.py + clinical_flags.py:
      result['ml_probability']    → assessments.ml_probability
      result['ml_risk_tier']      → assessments.ml_risk_tier
      result['above_threshold']   → assessments.above_threshold   ← NEW
      result['final_risk_level']  → assessments.final_risk_level
      result['escalated']         → assessments.escalated
      result['de_escalated']      → assessments.de_escalated       ← NEW
      result['clinical_flags']    → clinical_flags table
      result['shap_top_features'] → shap_explanations table

    Returns the new assessment_id.
    """
    with get_connection() as conn:

        # Patient (get or create)
        patient_id = get_or_create_patient(
            conn,
            bhw_id       = bhw_id,
            patient_data = patient_data,
            region       = layer1.get('region'),
            residence_type = layer1.get('residence_type'),
        )

        # referral_recommended: 1 if final level contains 'REFERRAL'
        referral_recommended = int(
            'REFERRAL' in result.get('final_risk_level', '')
        )

        shap_json = json.dumps(result.get('shap_top_features', []))

        # Build notes string for BP Rule of 3 audit trail
        notes_parts = []
        for r_num in [1, 2, 3]:
            s = layer2.get(f'bp_systolic_r{r_num}')
            d = layer2.get(f'bp_diastolic_r{r_num}')
            if s is not None and d is not None:
                notes_parts.append(f"BP R{r_num}: {s:.0f}/{d:.0f} mmHg")
        muac_r2 = layer2.get('muac_r2')
        if muac_r2 is not None:
            notes_parts.append(f"MUAC R2: {muac_r2:.1f} cm")
        extra_notes = layer2.get('notes', '')
        if extra_notes:
            notes_parts.append(extra_notes)
        notes_str = ' | '.join(notes_parts) if notes_parts else None

        # Insert assessment — column order matches schema.sql exactly
        cur = conn.execute(
            """
            INSERT INTO assessments (
                patient_id, bhw_id, assessment_date,
                maternal_age, education_yrs, wealth_score,
                birth_order, birth_interval, residence_type,
                region, anc_first_timing, iron_supplement,
                iron_days, tetanus_shots, marital_status, household_size,
                bp_systolic, bp_diastolic, muac_cm,
                weight_kg, height_cm, gestational_weeks,
                ml_probability, ml_risk_tier, above_threshold,
                final_risk_level, escalated, de_escalated,
                shap_top_features, referral_recommended, notes
            ) VALUES (
                ?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?
            )
            """,
            (
                patient_id,
                bhw_id,
                datetime.now().isoformat(),
                # Layer 1 (13 features — schema order)
                layer1.get('maternal_age'),
                layer1.get('education_yrs'),
                layer1.get('wealth_score'),
                layer1.get('birth_order'),
                layer1.get('birth_interval'),
                layer1.get('residence_type'),
                layer1.get('region'),
                layer1.get('anc_first_timing'),
                layer1.get('iron_supplement'),
                layer1.get('iron_days'),
                layer1.get('tetanus_shots'),
                layer1.get('marital_status'),
                layer1.get('household_size'),
                # Layer 2 (primary readings stored — full audit in notes)
                layer2.get('bp_systolic_r1'),
                layer2.get('bp_diastolic_r1'),
                layer2.get('muac_r1'),
                layer2.get('weight_kg'),
                layer2.get('height_cm'),
                layer2.get('gestational_weeks'),
                # System outputs
                result.get('ml_probability'),
                result.get('ml_risk_tier'),
                int(result.get('above_threshold', False)),  # ← NEW column
                result.get('final_risk_level'),
                int(result.get('escalated', False)),
                int(result.get('de_escalated', False)),     # ← NEW column
                shap_json,
                referral_recommended,
                notes_str,
            )
        )
        assessment_id = cur.lastrowid

        # Clinical flags (one row per triggered flag)
        for flag in result.get('clinical_flags', []):
            if flag.get('severity', 'none') == 'none':
                continue
            conn.execute(
                """
                INSERT INTO clinical_flags
                    (assessment_id, flag_type, severity,
                     measured_value, threshold_value, flag_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    flag.get('type'),
                    flag.get('severity'),
                    flag.get('measured_value'),
                    flag.get('threshold_value'),
                    flag.get('message'),
                )
            )

        # SHAP explanations (one row per top feature)
        for rank, sf in enumerate(
                result.get('shap_top_features', []), start=1):
            conn.execute(
                """
                INSERT INTO shap_explanations
                    (assessment_id, feature_rank, feature_name,
                     shap_value, feature_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    rank,
                    sf.get('feature'),
                    sf.get('shap_value'),
                    sf.get('feature_value'),
                )
            )

        conn.commit()
        return assessment_id


# QUERIES
def get_patient_history(patient_id: int) -> list:
    """
    All past assessments for one patient, newest first.
    Includes clinical flag summary and dual-purpose fusion columns.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.assessment_id,
                a.assessment_date,
                a.ml_probability,
                a.ml_risk_tier,
                a.above_threshold,
                a.final_risk_level,
                a.escalated,
                a.de_escalated,
                a.referral_recommended,
                a.bp_systolic,
                a.bp_diastolic,
                a.muac_cm,
                a.shap_top_features,
                GROUP_CONCAT(cf.flag_type, ', ') AS clinical_flags_summary
            FROM assessments a
            LEFT JOIN clinical_flags cf
                ON a.assessment_id = cf.assessment_id
            WHERE a.patient_id = ?
            GROUP BY a.assessment_id
            ORDER BY a.assessment_date DESC
            """,
            (patient_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_assessment_detail(assessment_id: int) -> Optional[dict]:
    """Full detail for one assessment including flags and SHAP rows."""
    with get_connection() as conn:
        assessment = conn.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?",
            (assessment_id,)
        ).fetchone()
        if not assessment:
            return None

        flags = conn.execute(
            "SELECT * FROM clinical_flags WHERE assessment_id = ?",
            (assessment_id,)
        ).fetchall()

        shap_rows = conn.execute(
            """
            SELECT * FROM shap_explanations
            WHERE assessment_id = ?
            ORDER BY feature_rank ASC
            """,
            (assessment_id,)
        ).fetchall()

    return {
        'assessment':        dict(assessment),
        'clinical_flags':    [dict(f) for f in flags],
        'shap_explanations': [dict(s) for s in shap_rows],
    }


def get_active_model_version() -> Optional[dict]:
    """Return the currently active model version from model_registry."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE is_active = 1 LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_bhw(bhw_id: int) -> Optional[dict]:
    """Return one BHW record by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM bhw_users WHERE bhw_id = ?",
            (bhw_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_bhws() -> list:
    """All BHW users for login/selection screen."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT bhw_id, full_name, barangay, municipality "
            "FROM bhw_users ORDER BY full_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_fusion_stats() -> dict:
    """
    Aggregate dual-purpose fusion statistics from the assessments table.
    Used by GET /api/health and the supervisor dashboard.
    Returns counts of: total, escalated, de_escalated, above_threshold,
    referral_recommended.
    """
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                           AS total,
                SUM(escalated)                     AS total_escalated,
                SUM(de_escalated)                  AS total_de_escalated,
                SUM(above_threshold)               AS total_above_threshold,
                SUM(referral_recommended)          AS total_referrals,
                ROUND(AVG(ml_probability), 4)      AS avg_ml_probability
            FROM assessments
            """
        ).fetchone()
    return dict(row) if row else {}

# REFINED DASHBOARD & SYSTEM QUERIES
def get_dashboard_summary(bhw_id: int = None) -> list:
    """
    Fetches the main list of assessments. 
    Added: bhw_id filtering and COALESCE to prevent NULL errors.
    """
    with get_connection() as conn:
        query = """
            SELECT 
                a.assessment_id,
                a.assessment_date,
                p.full_name AS patient_name,
                p.barangay,
                a.ml_probability,
                a.ml_risk_tier,
                a.final_risk_level,
                a.escalated,
                a.de_escalated,
                COALESCE((SELECT COUNT(*) FROM clinical_flags WHERE assessment_id = a.assessment_id), 0) AS flag_count
            FROM assessments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE (? IS NULL OR a.bhw_id = ?)
            ORDER BY a.assessment_date DESC
        """
        rows = conn.execute(query, (bhw_id, bhw_id)).fetchall()
    return [dict(r) for r in rows]

def get_active_threshold() -> float:
    """
    Fetches the threshold of the currently active model.
    Ensures the Web App matches your 0.5054 mathematical proof.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT threshold FROM model_registry WHERE is_active = 1 LIMIT 1"
        ).fetchone()
    return row['threshold'] if row else 0.5054 # Fallback to your proven value

# INITIALIZE ON IMPORT
try:
    initialize_db()
except Exception as e:
    print(f"[DB] Warning: Could not initialize on import: {e}")
