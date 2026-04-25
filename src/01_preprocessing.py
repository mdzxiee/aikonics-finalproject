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

    n2 = len(df)
    df = df[df['m19'].notna() & (df['m19'] > 0) & (df['m19'] < WEIGHT_MAX_GRAMS)].copy()
    print(f"  [F3] Valid weight range:                    {len(df):,}  (-{n2-len(df):,})")

    n3 = len(df)
    df = df[~df['m13'].isin([98, 99])].copy()
    df = df[~df['m45'].isin([8, 9])].copy()
    print(f"  [F4] Remove DHS special codes:              {len(df):,}  (-{n3-len(df):,})")

    pct = len(df) / n0 * 100
    print(f"\n  [FILTER RESULT] {len(df):,} of {n0:,} retained ({pct:.1f}%)")

    if len(df) < 1000:
        raise ValueError(
            f"Only {len(df)} rows after filtering (minimum required: 1000). "
            "Verify data path and filter logic."
        )
    return df

def create_target(df: pd.DataFrame):
    df = df.copy()
    df['LBW_Risk'] = (df['m19'] < LBW_THRESHOLD_GRAMS).astype(int)
    n_lbw  = df['LBW_Risk'].sum()
    n_norm = (df['LBW_Risk'] == 0).sum()
    ratio  = n_norm / n_lbw
    print(f"\n  [TARGET] LBW (1): {n_lbw:,} ({n_lbw/len(df)*100:.1f}%)")
    print(f"  [TARGET] Normal (0): {n_norm:,} ({n_norm/len(df)*100:.1f}%)")
    print(f"  [TARGET] Imbalance: {ratio:.2f}:1 → scale_pos_weight = {ratio:.1f}")
    return df, ratio

def create_mother_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['mother_id'] = (
        df['v001'].astype(str) + '_' +
        df['v002'].astype(str) + '_' +
        df['v003'].astype(str)
    )
    return df

n_mothers = df['mother_id'].nunique()
    n_multi   = df[df.duplicated('mother_id', keep=False)]['mother_id'].nunique()
    print(f"\n  [GROUP] Unique mothers: {n_mothers:,}")
    print(f"  [GROUP] Mothers with multiple birth records: {n_multi:,}")
    print(f"  [GROUP] Avg records per mother: {len(df)/n_mothers:.2f}")
    if n_multi > 0:
        print(f"  [GROUP] ⚠  GroupShuffleSplit REQUIRED — "
              f"{n_multi} mothers would leak across splits without it.")
    return df

def clean_dhs_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['m46'] = pd.to_numeric(df['m46'], errors='coerce')
    df.loc[df['m46'] > 270, 'm46'] = np.nan
    df['m1']  = pd.to_numeric(df['m1'],  errors='coerce').replace([8, 9], np.nan)
    df['v501']= pd.to_numeric(df['v501'],errors='coerce').replace([9], np.nan)
    df['v136']= pd.to_numeric(df['v136'],errors='coerce').replace([99], np.nan)
    return df

def apply_structural_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

df.loc[(df['m45'] != 1) & df['m46'].isnull(), 'm46'] = 0.0
    n_iron_zeroed = ((df['m45'] != 1) & (df['m46'] == 0.0)).sum()
    print(f"\n  [IMPUTE] iron_days → 0 for {n_iron_zeroed} non-supplement mothers")
n_tet = df['m1'].isnull().sum()
    df['m1'] = df['m1'].fillna(0.0)
    print(f"  [IMPUTE] tetanus_shots → 0 for {n_tet} records with no documentation")

n_first = (df['bord'] == 1).sum()
    df.loc[df['bord'] == 1, 'b11'] = 0.0
    print(f"  [IMPUTE] birth_interval → 0 for {n_first} first-borns (structural zero)")
n_non_first_nan = df[(df['bord'] > 1) & df['b11'].isnull()].shape[0]
    if n_non_first_nan > 0:
        print(f"  [IMPUTE] birth_interval NaN for non-first-borns: {n_non_first_nan} → XGBoost handles")
    return df
def drop_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['m19', 'm19a', 'b20', 'm14']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.rename(columns=COL_RENAME)
    return df
def validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns after preprocessing: {missing}")
    print(f"\n  [VALIDATE] All {len(FEATURE_COLS)} feature columns present ✓")
    null_summary = df[FEATURE_COLS].isnull().mean() * 100
    remaining    = null_summary[null_summary > 0]
    if len(remaining) > 0:
        print(f"  [VALIDATE] Remaining NaN (handled by XGBoost natively):")
        for col, pct in remaining.items():
            print(f"               {col:<25} {pct:.2f}% NaN")

            def run_preprocessing() -> pd.DataFrame:
    print("=" * 65)
    print("STAGE 1: DATA LOADING AND PREPROCESSING")
    print("=" * 65)
    df = load_raw(RAW_DATA_PATH)
    df = apply_filters(df)