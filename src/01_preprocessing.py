# AIKONIC — CS 322 | LBW Risk Prediction
# File: src/01_preprocessing.py
#
# Purpose : Load NDHS raw data → apply validity filters → create target
#           variable → create mother_id group key → apply structural
#           domain-aware imputation → save preprocessed.pkl
#
# Input   : data/PHKR82FL.csv (or .xlsx)
# Output  : artifacts/preprocessed.pkl  (dict: df, imbalance_ratio)
#
# DOES NOT split data — splitting is done in 04_model_training.py
# after GroupShuffleSplit to ensure mother-level separation.
# CSV exports (X_train, y_train, etc.) are produced by 04, not this file.
#
# Connects to: 04_model_training.py
# -------------------------------------------------------------------------

import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')