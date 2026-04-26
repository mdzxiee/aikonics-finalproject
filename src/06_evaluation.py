# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/06_evaluation.py
#
# Purpose : Full model evaluation across all three partitions (OOF / Test /
#           Unseen) using the SAME threshold from 05. Also simulates Layer 2
#           clinical escalation effects to answer RQ3 quantitatively.
#
# Connects to: 07_shap_analysis.py
# --------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OOF_PATH, OUTPUTS_DIR, ARTIFACTS_DIR
)

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)