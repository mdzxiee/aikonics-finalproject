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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DATA_PATH, ARTIFACTS_DIR, PREPROCESSED,
    LOAD_COLS, COL_RENAME, FEATURE_COLS,
    GESTATIONAL_AGE_MIN, WEIGHT_MAX_GRAMS, LBW_THRESHOLD_GRAMS,
)

# 1. Load 
def load_raw(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path, usecols=LOAD_COLS, low_memory=False)
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path, usecols=LOAD_COLS)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    print(f"  [LOAD] Raw: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# 2. Validity Filters 
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    4 sequential validity filters — each justified independently.
 
    Filter 1 (b20 >= 9): Full-term births only.
      Preterm LBW is driven by gestational age, not maternal socioeconomic
      factors. Including preterm records teaches the model the wrong causal
      signal and would produce referrals for an outcome (gestational age)
      that BHWs cannot observe or influence in early prenatal visits.
 
    Filter 2 (m19a == 1): Measured birth weights only.
      Recalled weights (m19a=2,3) have systematic maternal recall bias,
      particularly for older births. Unreliable ground truth degrades model
      calibration and inflates false-negative rates in evaluation.
 
    Filter 3 (0 < m19 < 9000): Valid weight range.
      DHS codes 9,996 and 9,998 indicate non-response. Zero weights are
      recording errors. Values above 9,000g are biologically implausible.
 
    Filter 4 (Remove DHS special codes in m14, m45):
      m14: 98 = "don't know", 99 = missing
      m45:  8 = "don't know",  9 = missing
      These are survey non-response codes, not real clinical values.
      Treating them as numeric would introduce error into key features.
    """
    n0 = len(df)
    df = df[df['b20'] >= GESTATIONAL_AGE_MIN].copy()
    print(f"  [F1] Full-term (b20>={GESTATIONAL_AGE_MIN}): {len(df):,}  (-{n0-len(df):,})")

    n1 = len(df)
    df = df[df['m19a'] == 1].copy()
    print(f"  [F2] Measured weight (m19a=1):              {len(df):,}  (-{n1-len(df):,})")