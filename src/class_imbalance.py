import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
import warnings

# Ignore warnings to keep your terminal output clean
warnings.filterwarnings("ignore")

print("--- Starting Class Imbalance Strategy Evaluation ---")

# 1. Load Data
print("Loading data from CSVs...")
X_train = pd.read_csv("artifacts/X_train.csv")
y_train_df = pd.read_csv("artifacts/y_train.csv")

# Ensure y_train is a 1D Series 
if 'LBW_Risk' in y_train_df.columns:
    y_train = y_train_df['LBW_Risk']
else:
    y_train = y_train_df.iloc[:, 0]

# 2. Dynamic Group Handling (The "mother_id" fix)
use_groups = False
groups_train = None

if 'mother_id' in X_train.columns:
    print("Found 'mother_id' in X_train.csv! Extracting for StratifiedGroupKFold.")
    groups_train = X_train['mother_id']
    X_train = X_train.drop(columns=['mother_id']) # Remove it so XGBoost doesn't train on it
    use_groups = True
    cv = StratifiedGroupKFold(n_splits=5)
else:
    print(" 'mother_id' not found in X_train.csv.")
    print(" Falling back to standard StratifiedKFold (this is fine for generating the baseline results table).")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Base XGBoost Parameters (Matching your methodology Table 9)
xgb_params = {
    'n_estimators': 100, # Kept at 100 for faster cross-validation
    'max_depth': 3,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'eval_metric': 'aucpr',
    'random_state': 42,
    'n_jobs': -1
}

# Calculate dynamic scale_pos_weight
n_normal = (y_train == 0).sum()
n_lbw = (y_train == 1).sum()
dynamic_spw = n_normal / n_lbw
print(f"Calculated scale_pos_weight: {dynamic_spw:.4f}")

# 4. Define the 5 Strategies
strategies = {
    # Strategy 1: Native XGBoost (NO imputer needed, respects NaNs)
    "XGBoost scale_pos_weight": xgb.XGBClassifier(**xgb_params, scale_pos_weight=dynamic_spw),
    
    # Strategy 2: SMOTE (Requires Imputation)
    "SMOTE": ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ]),
    
    # Strategy 3: ADASYN
    "ADASYN": ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', ADASYN(random_state=42)),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ]),
    
    # Strategy 4: BorderlineSMOTE
    "BorderlineSMOTE": ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', BorderlineSMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ]),
    
    # Strategy 5: SMOTEENN
    "SMOTEENN": ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', SMOTEENN(random_state=42)),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ])
}

# 5. Execute Cross-Validation Loop
results = []
print("\nRunning Cross-Validation... (this may take a minute or two)")

for name, model in strategies.items():
    oof_preds = np.zeros(len(y_train))
    
    # Split using the dynamically chosen CV method
    if use_groups:
        splits = cv.split(X_train, y_train, groups=groups_train)
    else:
        splits = cv.split(X_train, y_train)
        
    for train_idx, val_idx in splits:
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit the pipeline/model
        model.fit(X_tr, y_tr)
        
        # Predict OOF probabilities
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
    # Calculate Metrics
    pr_auc = average_precision_score(y_train, oof_preds)
    oof_class = (oof_preds >= 0.50).astype(int) # Default 0.5 for baseline comparison
    recall = recall_score(y_train, oof_class)
    
    results.append({
        "Strategy": name,
        "OOF PR-AUC": round(pr_auc, 4),
        "OOF Recall (at 0.5)": round(recall, 4)
    })
    print(f"Finished {name}: PR-AUC = {pr_auc:.4f}")

# 6. Display Final Results Table
results_df = pd.DataFrame(results).sort_values(by="OOF PR-AUC", ascending=False)
print("\n" + "="*50)
print("FINAL RESULTS TABLE FOR CHAPTER 4")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)