# Purpose : PRODUCTION Layer 2 — Clinical Safety and De-escalation.
#           Evaluates real BHW-entered measurements against WHO/DOH thresholds
#           using multi-reading verification protocols, then applies dual-purpose
#           Decision-Level Fusion with the ML output from predictor.py.
#
#  PRODUCTION vs ACADEMIC LAYER 2 
#   PRODUCTION (this file):
#     Handles REAL measurements from the BHW form via app.py.
#     Enforces Rule of 3 (BP) and MUAC re-measurement through the UI flow.
#     Produces results stored in the assessments table (schema.sql).
#
#   ACADEMIC (src/06_evaluation.py → simulate_layer2_academic_rq3):
#     Statistical simulation using DOH/FNRI prevalence-based random data.
#     Strictly for quantifying RQ3 in the thesis manuscript.
#     Never used in the deployed prototype.
#
#  DUAL-PURPOSE FUSION LOGIC 
#   ESCALATION PATH:
#     Any confirmed CRITICAL Layer 2 flag → HIGH PRIORITY REFERRAL
#     regardless of ML tier or whether ML was above threshold.
#     This catches clinical danger the ML model cannot see.
#
#   DE-ESCALATION PATH (addresses the 39%+ false positive referral rate):
#     ML above_threshold (True) + NO confirmed Layer 2 flags
#     → ELEVATED MONITORING (not RHU referral)
#     The ML model "wanted" to refer, but Layer 2 found no clinical danger
#     → Enhanced home monitoring instead of an unnecessary RHU visit.
#
#  SCHEMA COLUMN MAPPING 
#   assessments.escalated       = 1 if Layer 2 upgraded the recommendation
#   assessments.de_escalated    = 1 if Layer 2 downgraded an ML referral
#   assessments.above_threshold = 1 if ML probability >= threshold
#   assessments.referral_recommended = 1 if final level contains 'REFERRAL'
#
# Connects to: app.py (called inside POST /api/assess)
#              db_integration.py (result dict stored in assessments table)
# ===========================================================================

import os
from typing import Optional

# HARDCODED CLINICAL THRESHOLDS (Self-Contained for Production) 
CLINICAL_THRESHOLDS = {
    'bp_systolic_critical': 140.0,
    'bp_diastolic_critical': 90.0,
    'bp_systolic_warning': 130.0,
    'bp_diastolic_warning': 80.0,
    'muac_critical_cm': 23.5,
    'muac_warning_cm': 25.0
}

# SECTION 1 — BLOOD PRESSURE EVALUATION (RULE OF 3)

def evaluate_blood_pressure(sbp: float,
                              dbp: float,
                              reading_number: int = 1) -> dict:
    """
    Evaluate one BP reading against WHO/DOH thresholds.

    READING NUMBER determines clinical meaning:
      1 → Initial reading. If critical: app tells BHW to seat patient
          for 15 minutes then submit a second reading. No escalation yet.
      2 → Post-rest reading. If still critical: app tells BHW to correct
          posture and submit a third reading. Still no escalation.
      3 → Final confirmatory. ONLY Reading 3 can confirm hypertension
          and trigger a CRITICAL flag.

    SEVERITY LEVELS:
      'critical' → Reading 3 confirmed ≥ 140/90. Triggers escalation.
      'recheck'  → Readings 1 or 2 are ≥ 140/90. BHW must re-measure.
      'warning'  → Between 130/80 and 140/90. Monitor closely.
      'none'     → Below 130/80. No action.

    WHY RULE OF 3:
      Philippine Heart Association 2022 and DOH AO 2022-0012 require
      confirmation on ≥ 2 readings (with at least one post-rest) before
      diagnosing hypertension in pregnancy — to exclude isolated
      white-coat effect which is common in home visit settings.
    """
    crit_sys = CLINICAL_THRESHOLDS['bp_systolic_critical']    # 140
    crit_dia = CLINICAL_THRESHOLDS['bp_diastolic_critical']   # 90
    warn_sys = CLINICAL_THRESHOLDS['bp_systolic_warning']     # 130
    warn_dia = CLINICAL_THRESHOLDS['bp_diastolic_warning']    # 80

    is_critical = (sbp >= crit_sys) or (dbp >= crit_dia)
    is_warning  = (not is_critical) and ((sbp >= warn_sys) or (dbp >= warn_dia))

    flag = {
        'type':               None,
        'severity':           'none',
        'measured_value':     f"{sbp:.0f}/{dbp:.0f} mmHg",
        'threshold_value':    None,
        'message':            None,
        'reading_number':     reading_number,
        'requires_recheck':   False,
        'recheck_instruction':None,
    }

    if is_critical:
        if reading_number < 3:
            # Not yet confirmed — request next reading from BHW
            if reading_number == 1:
                instruction = (
                    "BP elevated. Seat the patient comfortably for 15 minutes "
                    "then take a second reading. "
                    "(I-upo ang pasyente ng 15 minuto at kumuha ng pangalawang BP.)"
                )
            else:  # reading_number == 2
                instruction = (
                    "BP still elevated after rest. Ask patient to sit upright "
                    "with both feet flat and arm at heart level, then take a "
                    "third reading. "
                    "(Ituwid ang pasyente at kumuha ng ikatlong BP.)"
                )
            flag.update({
                'type':               'BP_RECHECK_REQUIRED',
                'severity':           'recheck',
                'threshold_value':    f"≥{crit_sys}/{crit_dia} mmHg",
                'message':            instruction,
                'requires_recheck':   True,
                'recheck_instruction':instruction,
            })
        else:
            # Reading 3 — confirmed → escalate
            flag.update({
                'type':            'HYPERTENSION_IN_PREGNANCY',
                'severity':        'critical',
                'threshold_value': f"≥{crit_sys}/{crit_dia} mmHg",
                'message': (
                    f"CONFIRMED: BP {sbp:.0f}/{dbp:.0f} mmHg meets "
                    f"hypertension-in-pregnancy threshold (≥{crit_sys}/{crit_dia} mmHg) "
                    f"on 3rd reading. IMMEDIATE referral required. "
                    f"Risk of preeclampsia. Do NOT give antihypertensives "
                    f"without physician order."
                ),
            })

    elif is_warning:
        flag.update({
            'type':            'ELEVATED_BLOOD_PRESSURE',
            'severity':        'warning',
            'threshold_value': f"≥{warn_sys}/{warn_dia} mmHg",
            'message': (
                f"BP {sbp:.0f}/{dbp:.0f} mmHg is elevated "
                f"(≥{warn_sys}/{warn_dia} mmHg). Monitor closely. "
                f"Recheck at next visit. Educate on warning signs."
            ),
        })

    return flag

# SECTION 2 — MUAC EVALUATION (DOUBLE VERIFICATION)

def evaluate_muac(muac_r1: float,
                   muac_r2: Optional[float] = None) -> dict:
    """
    Evaluate MUAC against DOH/WHO thresholds with re-measurement verification.

    PROTOCOL:
      muac_r1 only → If < 23.5 cm: app prompts BHW to verify tape tension
                     on non-dominant arm and re-measure. Returns recheck.
      Both provided → Flag CONFIRMED only if BOTH readings are < 23.5 cm.
                      If only r1 is low: borderline/technique error → warning.

    WHY RE-MEASUREMENT:
      MUAC measurement error from incorrect tape tension or dominant-arm
      placement can be ± 0.3–0.7 cm. DOH MUAC guidelines require confirmed
      measurement on the non-dominant arm with correct tension before
      declaring maternal undernutrition.
    """
    crit_muac = CLINICAL_THRESHOLDS['muac_critical_cm']   # 23.5
    warn_muac = CLINICAL_THRESHOLDS['muac_warning_cm']    # 25.0

    measured_str = (f"{muac_r1:.1f} cm"
                    if muac_r2 is None else
                    f"R1={muac_r1:.1f} cm / R2={muac_r2:.1f} cm")

    flag = {
        'type':               None,
        'severity':           'none',
        'measured_value':     measured_str,
        'threshold_value':    None,
        'message':            None,
        'requires_recheck':   False,
        'recheck_instruction':None,
    }

    r1_critical = muac_r1 < crit_muac
    r1_warning  = (not r1_critical) and (muac_r1 < warn_muac)

    # Only Reading 1 provided 
    if muac_r2 is None:
        if r1_critical:
            instruction = (
                f"MUAC {muac_r1:.1f} cm is below {crit_muac} cm. "
                f"Re-check: place tape on non-dominant arm at midpoint "
                f"between shoulder and elbow with correct tension "
                f"(tape flat, not compressing skin), then enter 2nd reading. "
                f"(I-re-measure ang MUAC sa non-dominant arm.)"
            )
            flag.update({
                'type':               'MUAC_RECHECK_REQUIRED',
                'severity':           'recheck',
                'threshold_value':    f"<{crit_muac} cm",
                'message':            instruction,
                'requires_recheck':   True,
                'recheck_instruction':instruction,
            })
        elif r1_warning:
            flag.update({
                'type':            'LOW_MUAC_BORDERLINE',
                'severity':        'warning',
                'threshold_value': f"<{warn_muac} cm",
                'message': (
                    f"MUAC {muac_r1:.1f} cm is borderline (<{warn_muac} cm). "
                    f"Monitor nutrition and weight gain every visit."
                ),
            })
        return flag

    # Both readings provided 
    r2_critical = muac_r2 < crit_muac

    if r1_critical and r2_critical:
        # Both below threshold → CONFIRMED
        flag.update({
            'type':            'MATERNAL_UNDERNUTRITION',
            'severity':        'critical',
            'threshold_value': f"<{crit_muac} cm",
            'message': (
                f"CONFIRMED: MUAC {muac_r1:.1f}/{muac_r2:.1f} cm — both readings "
                f"below {crit_muac} cm. Maternal undernutrition confirmed. "
                f"Immediate nutritional counseling + RHU referral required. "
                f"Coordinate with MSWDO for supplementary feeding program."
            ),
        })
    elif r1_critical and not r2_critical:
        # First flagged, second cleared → technique issue → borderline
        flag.update({
            'type':            'LOW_MUAC_BORDERLINE',
            'severity':        'warning',
            'threshold_value': f"<{warn_muac} cm",
            'message': (
                f"MUAC borderline: R1={muac_r1:.1f} cm (below threshold) but "
                f"R2={muac_r2:.1f} cm (above on re-measure). "
                f"Likely measurement technique variation. Monitor closely."
            ),
        })
    elif (muac_r1 < warn_muac) or (muac_r2 < warn_muac):
        flag.update({
            'type':            'LOW_MUAC_BORDERLINE',
            'severity':        'warning',
            'threshold_value': f"<{warn_muac} cm",
            'message': (
                f"MUAC borderline: R1={muac_r1:.1f} cm / R2={muac_r2:.1f} cm. "
                f"At least one reading below {warn_muac} cm. "
                f"Monitor nutrition closely."
            ),
        })

    return flag

# SECTION 3 — DECISION-LEVEL FUSION (DUAL-PURPOSE)

def apply_decision_fusion(ml_probability: float,
                           above_threshold: bool,
                           ml_risk_tier: str,
                           bp_flag: Optional[dict] = None,
                           muac_flag: Optional[dict] = None) -> dict:
    """
    Apply dual-purpose Decision-Level Fusion.

    Takes the ML output (probability, above_threshold, tier) and any
    Layer 2 flags, then produces the final recommendation.

    Parameters
    ----------
    ml_probability   : Raw XGBoost probability (0–1). NEVER modified.
    above_threshold  : bool from predictor.py. True if prob >= threshold.
                       Required for de-escalation logic.
    ml_risk_tier     : 'LOW' / 'MEDIUM' / 'HIGH' (for display + fusion rules)
    bp_flag          : dict from evaluate_blood_pressure() or None
    muac_flag        : dict from evaluate_muac() or None

    Returns
    -------
    dict containing all fields needed by db_integration.save_assessment():
      ml_probability, ml_risk_tier, above_threshold,
      clinical_flags, final_risk_level,
      escalated (bool), de_escalated (bool), requires_recheck (bool),
      recommendations

    SCHEMA ALIGNMENT:
      escalated    → assessments.escalated    (0/1)
      de_escalated → assessments.de_escalated (0/1)
      above_threshold is passed through from predictor.py
      referral_recommended → 1 if 'REFERRAL' in final_risk_level
    """
    # Collect active flags (non-none severity)
    active_flags = []
    for f in [bp_flag, muac_flag]:
        if f is not None and f.get('severity', 'none') != 'none':
            active_flags.append(f)

    has_critical  = any(f['severity'] == 'critical' for f in active_flags)
    has_warning   = any(f['severity'] == 'warning'  for f in active_flags)
    has_recheck   = any(f['severity'] == 'recheck'  for f in active_flags)

    escalated    = False
    de_escalated = False
    requires_recheck = False

    # FUSION DECISION TREE 
    if has_critical:
        # 1. ESCALATION: Clinical danger confirmed → refer regardless of ML or pending rechecks
        final_risk_level = 'HIGH PRIORITY REFERRAL'
        escalated        = True

    elif has_recheck:
        # 2. PENDING DATA: We cannot finalize a warning or de-escalation if a measurement is unfinished
        final_risk_level = f'{ml_risk_tier} RISK - RECHECK PENDING'
        requires_recheck = True

    elif has_warning and ml_risk_tier == 'HIGH':
        # 3. ESCALATION: Warning + already high ML tier → refer
        final_risk_level = 'HIGH PRIORITY REFERRAL'
        escalated        = True

    elif has_warning:
        # 4. PARTIAL ESCALATION: Warning but ML is LOW/MEDIUM → monitor
        final_risk_level = 'ELEVATED MONITORING'
        escalated        = True

    elif above_threshold:
        # 5. DE-ESCALATION: ML wanted to refer (above_threshold), but vitals are perfectly normal
        final_risk_level = 'ELEVATED MONITORING'
        de_escalated     = True

    elif ml_risk_tier == 'MEDIUM':
        # 6. BUFFER ZONE: ML is below threshold but inside the 0.40 warning buffer. No clinical flags.
        final_risk_level = 'MEDIUM RISK'
        
    else:
        # 7. ROUTINE: ML is strictly LOW RISK and there are absolutely no clinical flags.
        final_risk_level = 'ROUTINE MONITORING'

    # Build recommendations
    recommendations = _build_recommendations(
        final_risk_level = final_risk_level,
        active_flags     = active_flags,
        ml_probability   = ml_probability,
        above_threshold  = above_threshold,
        escalated        = escalated,
        de_escalated     = de_escalated,
        requires_recheck = requires_recheck,
    )

    return {
        'ml_probability':    round(float(ml_probability), 4),
        'ml_risk_tier':      ml_risk_tier,
        'above_threshold':   bool(above_threshold),
        'clinical_flags':    active_flags,
        'final_risk_level':  final_risk_level,
        'escalated':         escalated,
        'de_escalated':      de_escalated,
        'requires_recheck':  requires_recheck,
        'recommendations':   recommendations,
    }

# SECTION 4 — RECOMMENDATION BUILDER

def _build_recommendations(final_risk_level: str,
                           active_flags: list,
                           ml_probability: float,
                           above_threshold: bool,
                           escalated: bool,
                           de_escalated: bool,
                           requires_recheck: bool) -> list:
    
    recs = []

    # 1. PENDING RECHECK OVERRIDE 
    # If the assessment is paused, only output the immediate recheck instructions.
    if requires_recheck:
        recheck_msgs = [f['recheck_instruction'] for f in active_flags if f.get('requires_recheck')]
        for msg in recheck_msgs:
            recs.append(msg)
        return [f"{i}. {r}" for i, r in enumerate(recs, start=1)]

    # 2. URGENT ESCALATIONS
    if 'HIGH PRIORITY REFERRAL' in final_risk_level:
        recs.append(
            "URGENT: I-refer sa RHU o pinakamalapit na health center NGAYON MISMO. "
            "(Refer to RHU or nearest health center TODAY.)"
        )
        recs.append(
            "I-coordinate sa midwife para sa kaagad na prenatal evaluation. "
            "(Coordinate with midwife for immediate prenatal evaluation.)"
        )

    # 3. SPECIFIC CLINICAL WARNINGS
    if any(f.get('type') == 'HYPERTENSION_IN_PREGNANCY' for f in active_flags):
        recs.append(
            "Huwag bigyan ng antihypertensive nang walang utos ng doktor. "
            "(Do NOT administer antihypertensives without physician order.)"
        )
        recs.append(
            "Turuan ang ina ng mga babala ng preeclampsia: matinding sakit ng ulo, "
            "malabong paningin, kirot sa itaas ng tiyan, pamamaga. "
            "(Educate on preeclampsia warning signs.)"
        )

    if any(f.get('type') == 'ELEVATED_BLOOD_PRESSURE' for f in active_flags):
        recs.append(
            "Subaybayan ang BP sa bawat pagbisita. I-record sa MCH book. "
            "(Monitor BP every visit. Record in MCH book.)"
        )

    if any(f.get('type') == 'MATERNAL_UNDERNUTRITION' for f in active_flags):
        recs.append(
            "Magbigay ng nutritional counseling at tiyakin ang iron + folate. "
            "(Nutritional counseling + ensure consistent iron + folate.)"
        )
        recs.append(
            "I-coordinate sa MSWDO para sa supplementary feeding program. "
            "(Coordinate with MSWDO for supplementary feeding program.)"
        )

    if any(f.get('type') == 'LOW_MUAC_BORDERLINE' for f in active_flags):
        recs.append(
            "Subaybayan ang nutritional status at weight gain sa bawat bisita. "
            "(Monitor nutritional status and weight gain every visit.)"
        )

    # 4. DE-ESCALATION CONTEXT 
    if de_escalated:
        recs.append(
            f"NOTA: Mataas ang statistical risk ng ina (ML probability={ml_probability:.2f}). "
            "Walang natagpuang clinical danger signs ngayon. Intensified home monitoring. "
            "(NOTE: Elevated statistical risk detected. No clinical danger signs today. "
            "Intensified monitoring recommended.)"
        )

    # 5. FOLLOW-UP SCHEDULE (Only triggers if assessment is finished) 
    if escalated or de_escalated or above_threshold:
        recs.append(
            "Mag-schedule ng follow-up home visit sa loob ng 1 linggo. "
            "(Schedule follow-up home visit within 1 week.)"
        )
    else:
        recs.append(
            "Ituloy ang regular na prenatal monitoring. "
            "(Continue routine prenatal monitoring as scheduled.)"
        )

    # 6. STANDARD CLOSING 
    recs.append(
        "Tiyakin ang iron supplementation at tetanus toxoid status. "
        "(Verify iron supplementation and tetanus toxoid status.)"
    )
    recs.append(
        "I-record ang lahat ng findings sa MCH book. "
        "(Record all findings in the MCH book.)"
    )

    return [f"{i}. {r}" for i, r in enumerate(recs, start=1)]

# SECTION 5 — FULL ASSESSMENT ENTRY POINT

def run_full_clinical_assessment(ml_probability: float,
                                   above_threshold: bool,
                                   ml_risk_tier: str,
                                   bp_systolic_r1: Optional[float] = None,
                                   bp_diastolic_r1: Optional[float] = None,
                                   bp_systolic_r2: Optional[float] = None,
                                   bp_diastolic_r2: Optional[float] = None,
                                   bp_systolic_r3: Optional[float] = None,
                                   bp_diastolic_r3: Optional[float] = None,
                                   muac_r1: Optional[float] = None,
                                   muac_r2: Optional[float] = None) -> dict:
    """
    Full Layer 2 clinical assessment from all available readings.

    Called by app.py after collecting all readings from the BHW form.
    Determines which BP reading to evaluate based on which readings exist:
      Only R1          → evaluate R1 (may return recheck)
      R1 + R2          → evaluate R2 (may return recheck)
      R1 + R2 + R3     → evaluate R3 (final — can confirm hypertension)

    Returns the full fusion result dict matching schema column names.
    """
    # Determine highest available BP reading
    if bp_systolic_r3 is not None and bp_diastolic_r3 is not None:
        bp_flag = evaluate_blood_pressure(
            bp_systolic_r3, bp_diastolic_r3, reading_number=3
        )
    elif bp_systolic_r2 is not None and bp_diastolic_r2 is not None:
        bp_flag = evaluate_blood_pressure(
            bp_systolic_r2, bp_diastolic_r2, reading_number=2
        )
    elif bp_systolic_r1 is not None and bp_diastolic_r1 is not None:
        bp_flag = evaluate_blood_pressure(
            bp_systolic_r1, bp_diastolic_r1, reading_number=1
        )
    else:
        bp_flag = None

    # MUAC evaluation
    muac_flag = evaluate_muac(muac_r1, muac_r2) if muac_r1 is not None else None

    return apply_decision_fusion(
        ml_probability  = ml_probability,
        above_threshold = above_threshold,
        ml_risk_tier    = ml_risk_tier,
        bp_flag         = bp_flag,
        muac_flag       = muac_flag,
    )


# DEMO

if __name__ == '__main__':
    print("=== DEMO: Nanay Rosario ===")
    result = run_full_clinical_assessment(
        ml_probability   = 0.38,
        above_threshold  = False,   # 0.38 < threshold 0.4854
        ml_risk_tier     = 'MEDIUM',
        bp_systolic_r1   = 138,
        bp_diastolic_r1  = 88,
        muac_r1          = 22.0,
        muac_r2          = 22.1,    # Both < 23.5 → CONFIRMED
    )
    print(f"ML            : {result['ml_probability']} ({result['ml_risk_tier']})")
    print(f"Above threshold: {result['above_threshold']}")
    print(f"Flags          : {[f['type'] for f in result['clinical_flags']]}")
    print(f"Final level    : {result['final_risk_level']}")
    print(f"Escalated      : {result['escalated']}")
    print(f"De-escalated   : {result['de_escalated']}")
    print("\nRecommendations:")
    for r in result['recommendations']:
        print(f"  {r}")

    print("\n=== DEMO: De-escalation case ===")
    result2 = run_full_clinical_assessment(
        ml_probability   = 0.60,
        above_threshold  = True,    # 0.60 > threshold → ML wanted to refer
        ml_risk_tier     = 'HIGH',
        bp_systolic_r1   = 118,
        bp_diastolic_r1  = 76,
        muac_r1          = 26.0,      # Perfectly normal MUAC (> 25.0)
    )
    print(f"ML            : {result2['ml_probability']} ({result2['ml_risk_tier']})")
    print(f"Above threshold: {result2['above_threshold']}")
    print(f"Final level    : {result2['final_risk_level']}")
    print(f"Escalated      : {result2['escalated']}")
    print(f"De-escalated   : {result2['de_escalated']}")
    print("→ ML wanted HIGH PRIORITY REFERRAL, Layer 2 de-escalated to MONITORING")