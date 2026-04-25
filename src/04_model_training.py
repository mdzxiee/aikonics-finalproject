import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PREPROCESSED, ARTIFACTS_DIR, FEATURE_COLS,
    X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH,
    X_UNSEEN_PATH, Y_UNSEEN_PATH, MODEL_PATH, OOF_PATH,
    RANDOM_STATE, UNSEEN_FRAC, TEST_FRAC, CV_FOLDS,
    ES_VALIDATION_FRAC, EARLY_STOPPING_ROUNDS, XGB_PARAMS,
)

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold
)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier