# Purpose : Layer 1 — Production ML inference wrapper.
#           Loads the trained XGBClassifier, threshold, and feature list
#           from prototype/, runs prediction on a single patient's Layer 1
#           inputs, and computes per-patient SHAP explanations.
#
# Input  : prototype/model.pkl     — XGBClassifier (no Pipeline)
#          prototype/threshold.pkl — float (e.g., 0.4854)
#          prototype/features.json — ordered feature list
#          layer1_inputs dict      — from BHW form (app.py)
#
# Output : dict containing:
#          ml_probability   — raw XGBoost probability (0–1)
#          ml_risk_tier     — 'LOW' / 'MEDIUM' / 'HIGH'
#          above_threshold  — bool: prob >= threshold (ML wanted to flag)
#                             Stored as above_threshold in assessments table
#                             Used by clinical_flags.py for de-escalation logic
#          shap_top_features — top 5 features by |SHAP| with direction
#
#  KEY ARCHITECTURE 
#   NO SimpleImputer — model was trained without one.
#   XGBoost handles NaN natively via sparsity-aware split finding.
#   SHAP uses tree_path_dependent — required for NaN inputs.
#   threshold.pkl is loaded separately from model.pkl (by design).
#   above_threshold is explicitly returned because clinical_flags.py
#   needs it to implement the de-escalation pathway correctly.
#
#  ABOUT above_threshold 
#   above_threshold = True  → ML probability >= threshold
#     Meaning: statistically, this mother is in the flagged population.
#     clinical_flags.py uses this to determine if de-escalation applies:
#       "ML flagged (above_threshold=True) + no Layer 2 danger → monitoring"
#
#   above_threshold = False → ML probability < threshold
#     Meaning: ML did not flag this mother.
#     If Layer 2 still finds a danger sign → escalation applies regardless.
#
# Connects to: clinical_flags.py (apply_decision_fusion uses above_threshold)
#              app.py (called inside POST /api/assess)
#              db_integration.py (above_threshold stored in assessments table)
# ===========================================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from typing import Optional

warnings.filterwarnings('ignore')

# Allow imports from src/
_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_PROTO_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, 'src'))

# Artifact paths (loaded from prototype/ directory)
_MODEL_PATH     = os.path.join(_PROTO_DIR, 'model.pkl')
_THRESHOLD_PATH = os.path.join(_PROTO_DIR, 'threshold.pkl')
_FEATURES_PATH  = os.path.join(_PROTO_DIR, 'features.json')


# LBWPredictor CLASS

class LBWPredictor:
    """
    Production ML inference wrapper for AIKONIC.

    Instantiate once at app startup via get_predictor().
    Reuse the singleton across all requests — loading the model
    and building the SHAP explainer takes ~200-500ms, which is
    unacceptable per-request latency in a field deployment.
    """

    def __init__(self):
        # Validate artifacts exist before loading
        for path, label in [
            (_MODEL_PATH,     'model.pkl'),
            (_THRESHOLD_PATH, 'threshold.pkl'),
            (_FEATURES_PATH,  'features.json'),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing prototype artifact: {path}\n"
                    "Run src/08_export_artifacts.py first."
                )

        # Load artifacts
        self.model     = joblib.load(_MODEL_PATH)
        self.threshold = float(joblib.load(_THRESHOLD_PATH))

        with open(_FEATURES_PATH) as f:
            feat_data = json.load(f)
        self.features = feat_data['features']

        # Validate model feature count
        try:
            expected = self.model.n_features_in_
            if expected != len(self.features):
                raise ValueError(
                    f"Model expects {expected} features but features.json "
                    f"has {len(self.features)}. Re-run 08_export_artifacts.py."
                )
        except AttributeError:
            pass  # Older XGBoost — skip

        # SHAP explainer — lazy-loaded on first call
        self._explainer: Optional[shap.TreeExplainer] = None

        print(f"[PREDICTOR] Loaded | threshold={self.threshold:.4f} "
              f"| {len(self.features)} features")

    def _get_explainer(self) -> shap.TreeExplainer:
        """
        Lazy-load SHAP TreeExplainer with tree_path_dependent mode.

        WHY tree_path_dependent (not default 'interventional'):
          The model was trained WITHOUT a SimpleImputer. Patient inputs
          can contain NaN values (e.g., birth_interval for non-first-borns
          with unreported prior birth, iron_days for non-supplement mothers
          where the conditional fill did not apply, etc.).

          Default 'interventional' SHAP substitutes background dataset
          values at split points to compute counterfactual predictions.
          When the actual split direction for NaN is different from the
          substituted value's direction, the counterfactual is incorrect —
          producing wrong Shapley attributions.

          tree_path_dependent follows the EXACT PATH each observation takes
          through every tree, including the learned NaN direction.
          This is both methodologically correct and computationally faster
          for tree ensembles.

        Cached after first call — building takes ~100-200ms.
        """
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation='tree_path_dependent'
            )
        return self._explainer

    def predict(self, layer1_inputs: dict) -> dict:
        """
        Run Layer 1 ML prediction for one patient.

        Parameters
        ----------
        layer1_inputs : dict
            Keys must match self.features exactly.
            Missing keys → NaN (XGBoost handles natively).
            wealth_score must be raw v191 integer (e.g., -70000, -3000, 65000).
            NOT a normalized 0–5 value.

        Returns
        -------
        dict:
          ml_probability   : float  — raw XGBoost LBW probability (0–1)
          ml_risk_tier     : str    — 'LOW' / 'MEDIUM' / 'HIGH'
          above_threshold  : bool   — probability >= threshold
                                      Stored in assessments.above_threshold
                                      Used in clinical_flags.py fusion logic
          shap_top_features: list   — top 5 features by |SHAP| with metadata
        """
        # Build input row in fixed feature order
        row = []
        for feat in self.features:
            val = layer1_inputs.get(feat)
            row.append(float(val) if val is not None else np.nan)

        X = pd.DataFrame([row], columns=self.features)

        # ML probability (NaN handled natively by XGBoost)
        proba = float(self.model.predict_proba(X)[0, 1])

       # above_threshold — the key field for de-escalation logic
        above_threshold = proba >= self.threshold

        # Risk tier calculation
        if above_threshold:
            ml_tier = 'HIGH'
        elif proba >= 0.47:  
            ml_tier = 'MEDIUM'
        else:
            ml_tier = 'LOW'

        # SHAP explanation
        shap_top = self._explain(X)

        return {
            'ml_probability':    round(proba, 4),
            'ml_risk_tier':      ml_tier,
            'above_threshold':   bool(above_threshold),
            'shap_top_features': shap_top,
        }

    def _explain(self, X: pd.DataFrame, top_n: int = 5) -> list:
        """
        Compute SHAP values for one input row.
        Returns top_n features sorted by absolute SHAP value.
        NaN inputs handled correctly via tree_path_dependent.
        """
        explainer   = self._get_explainer()
        shap_values = explainer(X)
        values      = shap_values.values[0]     # shape: (n_features,)
        feat_vals   = X.iloc[0].values

        ranked = sorted(
            zip(self.features, values, feat_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        result = []
        for feat, sv, fv in ranked:
            result.append({
                'feature':       feat,
                'shap_value':    round(float(sv), 5),
                'feature_value': round(float(fv), 4) if not np.isnan(fv) else None,
                'direction':     'increases_risk' if sv > 0 else 'decreases_risk',
                'display_label': _make_display_label(feat, fv),
            })
        return result


# DISPLAY LABEL HELPER

def _make_display_label(feature: str, value: float) -> str:
    """
    Convert raw feature name + value to a BHW-readable string.
    Used in result.html and recommendation display.
    wealth_score uses raw v191 integers — labels reflect actual percentile ranges.
    """
    if np.isnan(value):
        return f"{feature}: not recorded"

    labels = {
        'maternal_age': lambda v:
            f"Age: {v:.0f} years",

        'education_yrs': lambda v:
            f"Education: {v:.0f} years",

        'wealth_score': lambda v: (
            "Wealth: Very poor"      if v < -100000 else
            "Wealth: Poor"           if v <  -30000 else
            "Wealth: Below average"  if v <       0 else
            "Wealth: Average"        if v <   30000 else
            "Wealth: Above average"  if v <   80000 else
            "Wealth: Comfortable"
        ),

        'birth_order': lambda v:
            (f"Birth order: 1st child (first pregnancy)"
             if v == 1 else f"Birth order: {v:.0f}th child"),

        'birth_interval': lambda v: (
            "First pregnancy — no preceding birth interval"
            if v == 0 else
            f"Birth interval: {v:.0f} months since last birth"
        ),

        'residence_type': lambda v:
            "Residence: Urban" if v == 1 else "Residence: Rural",

        'region': lambda v:
            f"Region: {v:.0f}",

        'anc_first_timing': lambda v: (
            "ANC: No antenatal care visits recorded"
            if v == 0 else
            f"First ANC visit: Month {v:.0f} of pregnancy"
        ),

        'iron_supplement': lambda v:
            "Iron supplement: Received" if v == 1 else
            "Iron supplement: Not received",

        'iron_days': lambda v:
            f"Iron taken: {v:.0f} days",

        'tetanus_shots': lambda v:
            f"Tetanus shots: {v:.0f}",

        'marital_status': lambda v: {
            0: "Never married",
            1: "Married",
            2: "Living together",
            3: "Widowed",
            4: "Divorced/separated",
            5: "Not living together",
        }.get(int(v), f"Marital status: {v:.0f}"),

        'household_size': lambda v:
            f"Household size: {v:.0f} members",
    }

    formatter = labels.get(feature)
    if formatter:
        try:
            return formatter(value)
        except Exception:
            return f"{feature}: {value}"
    return f"{feature}: {value}"

# SINGLETON ACCESSOR

_predictor_instance: Optional['LBWPredictor'] = None


def get_predictor() -> LBWPredictor:
    """
    Return the shared LBWPredictor instance.
    Initializes on first call, reuses on subsequent calls.
    app.py calls this once at startup to warm up the model.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LBWPredictor()
    return _predictor_instance

# DEMO

if __name__ == '__main__':
    pred = get_predictor()

    # Nanay Rosario — Case 2 from seed data
    inputs = {
        'maternal_age':     32,
        'education_yrs':    6,
        'wealth_score':     -3000,    # Median raw v191 score
        'birth_order':      3,
        'birth_interval':   18,
        'residence_type':   2,
        'region':           6,
        'anc_first_timing': 5,
        'iron_supplement':  1,
        'iron_days':        30,
        'tetanus_shots':    1,
        'marital_status':   1,
        'household_size':   7,
    }

    result = pred.predict(inputs)
    print(f"\nML Probability  : {result['ml_probability']}")
    print(f"Risk Tier       : {result['ml_risk_tier']}")
    print(f"Above threshold : {result['above_threshold']}  "
          f"(threshold={pred.threshold:.4f})")
    print(f"\nTop SHAP factors:")
    for f in result['shap_top_features']:
        arrow = '↑' if f['direction'] == 'increases_risk' else '↓'
        print(f"  {f['feature']:<24} {arrow} SHAP={f['shap_value']:+.5f} "
              f"| {f['display_label']}")