# Purpose : SHAP explainability for RQ1 (global feature ranking) and
#           RQ4 (individual decision pathways for BHWs).
#
# Connects to: 08_export_artifacts.py
# ------------------------------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OUTPUTS_DIR, FEATURE_COLS,
    X_TEST_PATH, Y_TEST_PATH, X_UNSEEN_PATH, Y_UNSEEN_PATH
)

# 1. TreeExplainer Setup 

def get_explainer(model) -> shap.TreeExplainer:
    """Build SHAP TreeExplainer with tree_path_dependent perturbation."""
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation='tree_path_dependent'
    )
    return explainer


