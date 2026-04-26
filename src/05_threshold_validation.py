# ===========================================================================
# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/05_threshold_validation.py
#
# Purpose : Select and save the optimal classification threshold using
#           exclusively out-of-fold (OOF) probabilities from 04.
#           Saves threshold.pkl and features.json as SEPARATE artifacts.
#
# Input  : artifacts/oof_probabilities.pkl
# Output : artifacts/threshold.pkl
#          artifacts/features.json
#          outputs/threshold_analysis.png
#          outputs/threshold_selection_table.csv
#
# CRITICAL DESIGN RULE:
#   The threshold is selected on OOF probabilities (training-derived).
#   It is NEVER selected using test or unseen data.
#   Using test/unseen probabilities for threshold selection is data leakage —
#   it optimizes the threshold on the very data used to evaluate it.
#
# WHY threshold.pkl IS SEPARATE FROM model.pkl:
#   1. The threshold can be recalibrated (e.g., for a new health district
#      with different risk tolerance) without retraining the model.
#   2. The prototype loads model.pkl and threshold.pkl independently —
#      this makes the system modular and auditable.
#   3. A panel can inspect the threshold decision independently from
#      the model's internal parameters.
#
# Connects to: 06_evaluation.py, prototype/predictor.py
# ===========================================================================

import os, sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OOF_PATH, ARTIFACTS_DIR, OUTPUTS_DIR, THRESHOLD_PATH,
    FEATURES_PATH, FEATURE_COLS, MIN_RECALL_FLOOR
)

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score
)