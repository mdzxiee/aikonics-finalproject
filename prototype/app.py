# Purpose : Flask REST API — main entry point for the AIKONIC prototype.
#           Orchestrates Layer 1 (predictor.py) + Layer 2 (clinical_flags.py)
#           + database (db_integration.py) for every assessment request.
#
# ENDPOINTS 
#
#  API ROUTES (The Data Engine - returns JSON) 
#   POST /api/assess          — main assessment (Layer 1 + Layer 2 + DB save)
#   POST /api/bp-recheck      — submit additional BP reading (Rule of 3 flow)
#   GET  /api/history/<id>    — patient assessment history
#   GET  /api/assessment/<id> — full assessment detail with flags + SHAP
#   GET  /api/bhws            — all BHW users (for login/selection screen)
#   GET  /api/stats           — dual-purpose fusion aggregate statistics
#   GET  /api/health          — system health check
#
#  VISUAL ROUTES (The UI - returns HTML) 
#   GET  /                    — serve DASHBOARD (dashboard.html) -> Shows the 33 patients!
#   GET  /assess              — serve main BHW form (assess.html) -> To add a 34th patient
#   GET  /patient/<id>        — serve detailed patient view (detail.html) -> Replaces /result and /history
#
# ─── REQUEST / RESPONSE FORMAT ───────────────────────────────────────────────
#   All API endpoints: JSON (Content-Type: application/json)
#   HTML routes: Jinja2 templates from prototype/templates/
#
# ─── DUAL-PURPOSE LAYER 2 INTEGRATION ────────────────────────────────────────
#   The /api/assess endpoint:
#     1. Calls predictor.py → gets ml_probability, above_threshold, shap
#     2. Calls clinical_flags.py → applies Rule of 3 / MUAC verification
#     3. Fusion result includes: escalated, de_escalated, final_risk_level
#     4. All three new schema columns (above_threshold, de_escalated) are
#        saved to the database via db_integration.save_assessment()
#
#   The /api/bp-recheck endpoint handles the Rule of 3 multi-step flow:
#     BHW submits R1 → API returns recheck instruction
#     BHW submits R2 → API returns recheck instruction if still high
#     BHW submits R3 → API returns final confirmed assessment
#
# IMPORTANT 
#   This is the PRODUCTION system using real BHW-entered measurements.
#   The ACADEMIC Layer 2 simulation for RQ3 is in src/06_evaluation.py.
#   Do NOT conflate the two implementations.
#
# Run: python prototype/app.py  →  http://localhost:5000
# ===========================================================================

import os
import sys
import json
import traceback
from flask import (
    Flask, request, jsonify,
    render_template, send_from_directory
)

# Path setup 
_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_PROTO_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, 'src'))
sys.path.insert(0, _PROTO_DIR)

from predictor      import get_predictor
from clinical_flags import run_full_clinical_assessment
from db_integration import (
    save_assessment,
    get_patient_history,
    get_assessment_detail,
    get_all_bhws,
    get_bhw,
    get_active_model_version,
    get_fusion_stats,
    get_dashboard_summary  
)

# Flask app 
app = Flask(
    __name__,
    template_folder = os.path.join(_PROTO_DIR, 'templates'),
    static_folder   = os.path.join(_PROTO_DIR, 'static'),
)
app.config['JSON_SORT_KEYS'] = False


# Startup: warm up predictor 
try:
    _predictor = get_predictor()
    print(f"[APP] Predictor ready  | threshold={_predictor.threshold:.4f} "
          f"| {len(_predictor.features)} features")
except Exception as e:
    print(f"[APP] ⚠ Predictor failed to load: {e}")
    print(f"[APP]   Run src/08_export_artifacts.py first.")
    _predictor = None


# HTML ROUTES

@app.route('/')
def dashboard():
    """THE DASHBOARD: The Main BHW landing page showing all patients."""
    assessments = get_dashboard_summary()
    stats = get_fusion_stats()
    return render_template('dashboard.html', assessments=assessments, stats=stats)

@app.route('/assess')
def assess_page():
    """THE FORM: The blank page where the BHW inputs a new mother."""
    bhws = get_all_bhws()
    return render_template('assess.html', bhws=bhws)

@app.route('/patient/<int:assessment_id>')
def detail_page(assessment_id):
    """THE RESULT & HISTORY: The page showing SHAP charts for one specific mother."""
    detail = get_assessment_detail(assessment_id)
    return render_template('detail.html', detail=detail)

# POST /api/assess  — MAIN ASSESSMENT ENDPOINT
@app.route('/api/assess', methods=['POST'])
def assess():
    """
    Core assessment endpoint. Runs Layer 1 + Layer 2 + saves to DB.

    REQUEST BODY (JSON):
    {
        "bhw_id": 1,
        "patient": {
            "full_name":    "Rosario Dela Cruz",
            "barangay":     "Brgy. Sta. Cruz",
            "municipality": "Ligao City"
        },
        "layer1": {
            "maternal_age":     32,
            "education_yrs":    6,
            "wealth_score":     -3000,
            "birth_order":      3,
            "birth_interval":   18,
            "residence_type":   2,
            "region":           6,
            "anc_first_timing": 5,
            "iron_supplement":  1,
            "iron_days":        30,
            "tetanus_shots":    1,
            "marital_status":   1,
            "household_size":   7
        },
        "layer2": {
            "bp_systolic_r1":  138,
            "bp_diastolic_r1": 88,
            "bp_systolic_r2":  null,
            "bp_diastolic_r2": null,
            "bp_systolic_r3":  null,
            "bp_diastolic_r3": null,
            "muac_r1":         22.0,
            "muac_r2":         22.1,
            "weight_kg":       52,
            "height_cm":       155,
            "gestational_weeks": 24
        }
    }

    RESPONSE (JSON):
    {
        "assessment_id":    38,
        "ml_probability":   0.4819,
        "ml_risk_tier":     "MEDIUM",
        "above_threshold":  false,
        "clinical_flags":   [
            {"type": "ELEVATED_BLOOD_PRESSURE", "severity": "warning"},
            {"type": "MATERNAL_UNDERNUTRITION", "severity": "critical"}
        ],
        "final_risk_level": "HIGH PRIORITY REFERRAL",
        "escalated":        true,
        "de_escalated":     false,
        "requires_recheck": false,
        "recommendations":  [...],
        "shap_top_features":[...]
    }
    """
    if _predictor is None:
        return jsonify({
            'error': 'Model not loaded. Run src/08_export_artifacts.py first.'
        }), 503

    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({'error': 'Invalid or missing JSON body.'}), 400

        bhw_id  = data.get('bhw_id', 1)
        patient = data.get('patient', {})
        layer1  = data.get('layer1', {})
        layer2  = data.get('layer2', {})

        # Validate required Layer 1 fields
        required_l1 = ['maternal_age', 'birth_order', 'residence_type', 'region']
        missing = [f for f in required_l1 if layer1.get(f) is None]
        if missing:
            return jsonify({'error': f'Missing required Layer 1 fields: {missing}'}), 400

        # STEP 1: Layer 1 — ML prediction 
        ml_result = _predictor.predict(layer1)

        # STEP 2: Layer 2 — Clinical assessment
        l2_result = run_full_clinical_assessment(
            ml_probability   = ml_result['ml_probability'],
            bp_systolic_r1   = _f(layer2.get('bp_systolic_r1')),
            bp_diastolic_r1  = _f(layer2.get('bp_diastolic_r1')),
            bp_systolic_r2   = _f(layer2.get('bp_systolic_r2')),
            bp_diastolic_r2  = _f(layer2.get('bp_diastolic_r2')),
            bp_systolic_r3   = _f(layer2.get('bp_systolic_r3')),
            bp_diastolic_r3  = _f(layer2.get('bp_diastolic_r3')),
            muac_r1          = _f(layer2.get('muac_r1')),
            muac_r2          = _f(layer2.get('muac_r2')),
        )

        # STEP 3: Merge result 
        full_result = {
            **l2_result,
            'shap_top_features': ml_result['shap_top_features'],
            'above_threshold':   ml_result['above_threshold'] 
        }

        # STEP 4: Translate API data to Database format
        layer2_db = {
            'bp_systolic':       layer2.get('bp_systolic_r1'),
            'bp_diastolic':      layer2.get('bp_diastolic_r1'),
            'muac_cm':           layer2.get('muac_r1'),
            'weight_kg':         layer2.get('weight_kg'),
            'height_cm':         layer2.get('height_cm'),
            'gestational_weeks': layer2.get('gestational_weeks'),
            'notes':             _build_bp_notes(layer2)
        }

        # Save to database using the translated layer2_db
        assessment_id = save_assessment(
            bhw_id       = bhw_id,
            patient_data = patient,
            layer1       = layer1,
            layer2       = layer2_db,
            result       = full_result,
        )
        full_result['assessment_id'] = assessment_id

        return jsonify(full_result), 200

    except Exception as e:
        import traceback 
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

# POST /api/bp-recheck  — RULE OF 3 ADDITIONAL READING

@app.route('/api/bp-recheck', methods=['POST'])
def bp_recheck():
    """
    Submit an additional BP reading during the Rule of 3 flow.

    The frontend calls this after the BHW performs a rest (Reading 2)
    or posture-correction re-measurement (Reading 3).
    The endpoint re-evaluates Layer 2 with the new reading and returns
    an updated fusion result.

    REQUEST BODY (JSON):
    {
        "assessment_id":  42,
        "ml_probability": 0.60,
        "above_threshold": true,
        "ml_risk_tier":   "HIGH",
        "reading_number": 2,
        "bp_systolic":    142,
        "bp_diastolic":   91,
        "muac_result":    { ... }   -- carry forward from original (optional)
    }
    """
    if _predictor is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({'error': 'Invalid or missing JSON body.'}), 400

        ml_probability  = float(data.get('ml_probability', 0.0))
        above_threshold = bool(data.get('above_threshold', False))
        ml_risk_tier    = data.get('ml_risk_tier', 'HIGH')
        reading_number  = int(data.get('reading_number', 2))

        sbp             = _f(data.get('bp_systolic'))
        dbp             = _f(data.get('bp_diastolic'))

        if sbp is None or dbp is None:
            return jsonify({
                'error': 'bp_systolic and bp_diastolic are required.'
            }), 400

        from clinical_flags import evaluate_blood_pressure, apply_decision_fusion

        bp_flag   = evaluate_blood_pressure(sbp, dbp, reading_number=reading_number)
        muac_flag = data.get('muac_result')  # carry-forward dict or None

        result = apply_decision_fusion(
            ml_probability  = ml_probability,
            above_threshold = above_threshold,
            ml_risk_tier    = ml_risk_tier,
            bp_flag         = bp_flag,
            muac_flag       = muac_flag,
        )
        result['assessment_id'] = data.get('assessment_id')
        
        return jsonify(result), 200

    except Exception as e:
        import traceback 
        return jsonify({
            'error': str(e), 
            'trace': traceback.format_exc()
        }), 500

# ===========================================================================
# GET /api/history/<patient_id>
# ===========================================================================

@app.route('/api/history/<int:patient_id>', methods=['GET'])
def patient_history(patient_id: int):
    """
    All past assessments for one patient, newest first.
    Includes above_threshold and de_escalated for longitudinal fusion tracking.
    """
    try:
        records = get_patient_history(patient_id)
        return jsonify({
            'patient_id':  patient_id,
            'total':       len(records),
            'assessments': records,
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
# GET /api/assessment/<assessment_id>
# ===========================================================================

@app.route('/api/assessment/<int:assessment_id>', methods=['GET'])
def assessment_detail(assessment_id: int):
    """
    Full detail for one assessment:
    assessment record + clinical flags + SHAP explanations.
    """
    try:
        detail = get_assessment_detail(assessment_id)
        if not detail:
            return jsonify({
                'error': f'Assessment {assessment_id} not found.'
            }), 404
        return jsonify(detail), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
# GET /api/bhws
# ===========================================================================

@app.route('/api/bhws', methods=['GET'])
def list_bhws():
    """All BHW users — for login/selection screen on the frontend."""
    try:
        return jsonify({'bhws': get_all_bhws()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
# GET /api/stats
# ===========================================================================

@app.route('/api/stats', methods=['GET'])
def fusion_stats():
    """
    Aggregate dual-purpose fusion statistics.
    Shows how often the system escalated vs de-escalated.
    Useful for RHU coordinator dashboard and thesis defense data.

    RESPONSE:
    {
        "total": 150,
        "total_escalated": 22,
        "total_de_escalated": 41,
        "total_above_threshold": 63,
        "total_referrals": 22,
        "avg_ml_probability": 0.3821
    }
    """
    try:
        stats = get_fusion_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
# GET /api/health
# ===========================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    System health check.
    Confirms model, threshold, and database are all operational.
    """
    model_info = get_active_model_version()
    return jsonify({
        'status':         'ok' if _predictor else 'model_not_loaded',
        'model_loaded':   _predictor is not None,
        'threshold':      _predictor.threshold if _predictor else None,
        'n_features':     len(_predictor.features) if _predictor else None,
        'model_version':  model_info.get('model_version') if model_info else None,
        'db_path':        os.path.abspath(
                              os.path.join(_ROOT_DIR, 'database', 'aikonic.db')
                          ),
        'dual_purpose_l2': True,
        'fusion_stats':   get_fusion_stats(),
    }), 200


# HELPER

def _f(value) -> float | None:
    """Convert value to float, return None if missing/null/empty string."""
    if value is None or value == '' or value == 'null':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  AIKONIC LBW Risk Assessment — Prototype API")
    print("=" * 65)
    print(f"  Root      : {_ROOT_DIR}")
    print(f"  Templates : {os.path.join(_PROTO_DIR, 'templates')}")
    print(f"  Database  : {os.path.join(_ROOT_DIR, 'database', 'aikonic.db')}")
    print(f"  URL       : http://localhost:5000")
    print(f"")
    print(f"  Endpoints:")
    print(f"    POST /api/assess          — main assessment")
    print(f"    POST /api/bp-recheck      — Rule of 3 additional reading")
    print(f"    GET  /api/history/<id>    — patient history")
    print(f"    GET  /api/assessment/<id> — full assessment detail")
    print(f"    GET  /api/bhws            — BHW list")
    print(f"    GET  /api/stats           — fusion statistics")
    print(f"    GET  /api/health          — health check")
    print("=" * 65)

    app.run(
        debug    = False,
        host     = '0.0.0.0',
        port     = 5000,
        threaded = False,   # SHAP is not thread-safe — single-threaded
    )