# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/02_eda.py
#
# Purpose : Exploratory Data Analysis exclusively on the TRAINING set.
#           Any EDA on the full/test/unseen dataset constitutes data snooping
#           — analyst decisions informed by test-set patterns contaminate
#           the integrity of subsequent evaluations.
#
# Input  : artifacts/X_train.csv + artifacts/y_train.csv
#          (produced by 04_model_training.py — run 04 first, then return here
#           for documentation; or run 04 immediately after 01 then run 02)
#
# Output : outputs/eda_distributions.png
#          outputs/eda_class_comparison.png
#          outputs/feature_auc_train.csv     (RQ1 evidence)
#          outputs/feature_stats_train.csv   (Cohen's d + correlations)
#
# Connects to: 03_feature_selection.py (documentation chain)
# ---------------------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, OUTPUTS_DIR, FEATURE_COLS
)

from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr, mannwhitneyu


#  Helpers 

def cohens_d(group1: pd.Series, group0: pd.Series) -> float:
    """Pooled standard deviation Cohen's d effect size."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std() ** 2 + (n0 - 1) * group0.std() ** 2) /
        (n1 + n0 - 2)
    )
    return float((group1.mean() - group0.mean()) / pooled_std) if pooled_std > 0 else 0.0


def individual_auc(series: pd.Series, y: pd.Series) -> float:
    """Compute AUC of a single feature vs. binary target. Always ≥ 0.50."""
    s = series.fillna(series.median())
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)