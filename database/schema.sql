-- AIKONIC — CS 322 | LBW Risk Prediction
-- File: database/schema.sql
-- Purpose: Relational database blueprint for the Prototype Web App.


-- 1. BHW USERS
CREATE TABLE IF NOT EXISTS bhw_users (
    bhw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    barangay TEXT NOT NULL,
    municipality TEXT NOT NULL,
    region TEXT NOT NULL,
    contact_no TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. PATIENTS
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bhw_id INTEGER NOT NULL,
    full_name TEXT NOT NULL,
    birth_date TEXT,
    barangay TEXT NOT NULL,
    municipality TEXT NOT NULL,
    region INTEGER NOT NULL,
    residence_type INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bhw_id) REFERENCES bhw_users(bhw_id)
);

-- 3. ASSESSMENTS
CREATE TABLE IF NOT EXISTS assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    bhw_id INTEGER NOT NULL,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Layer 1: ML Features
    maternal_age INTEGER,
    education_yrs INTEGER,
    wealth_score REAL,
    birth_order INTEGER,
    birth_interval INTEGER,
    residence_type INTEGER,
    region INTEGER,
    anc_first_timing INTEGER,
    iron_supplement INTEGER,
    iron_days INTEGER,
    tetanus_shots INTEGER,
    marital_status INTEGER,
    household_size INTEGER,

    -- Layer 2: Clinical Physical Measurements
    bp_systolic REAL,
    bp_diastolic REAL,
    muac_cm REAL,
    weight_kg REAL,
    height_cm REAL,
    gestational_weeks INTEGER,

    -- System Outputs & Workflow Tracking
    ml_probability REAL NOT NULL,
    ml_risk_tier TEXT NOT NULL,
    final_risk_level TEXT NOT NULL,
    escalated INTEGER DEFAULT 0,
    shap_top_features TEXT,
    referral_recommended INTEGER DEFAULT 0,
    
    -- Post-Assessment Workflow
    referral_completed TEXT,
    referral_outcome TEXT,
    notes TEXT,

    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (bhw_id) REFERENCES bhw_users(bhw_id)
);

-- 4. CLINICAL FLAGS
CREATE TABLE IF NOT EXISTS clinical_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    flag_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    measured_value TEXT,     -- Changed to TEXT to allow units (e.g., "22.0 cm")
    threshold_value TEXT,    -- Changed to TEXT to allow operators (e.g., "<23.5")
    flag_message TEXT,
    FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
);

-- 5. SHAP EXPLANATIONS
CREATE TABLE IF NOT EXISTS shap_explanations (
    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    feature_rank INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    shap_value REAL NOT NULL,
    feature_value REAL,
    FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
);

-- 6. MODEL REGISTRY
CREATE TABLE IF NOT EXISTS model_registry (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_path TEXT NOT NULL,
    threshold REAL NOT NULL,
    oof_auc REAL,
    test_auc REAL,
    unseen_recall REAL,
    is_active INTEGER DEFAULT 0,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- VIEWS & INDEXES

-- 7. DENORMALIZED VIEW FOR REPORTING DASHBOARDS
CREATE VIEW IF NOT EXISTS v_assessment_report AS
SELECT 
    a.assessment_id,
    a.assessment_date,
    p.full_name AS patient_name,
    p.barangay,
    b.full_name AS bhw_name,
    a.final_risk_level,
    a.referral_recommended,
    a.referral_completed,
    a.bp_systolic,
    a.bp_diastolic,
    a.muac_cm
FROM assessments a
JOIN patients p ON a.patient_id = p.patient_id
JOIN bhw_users b ON a.bhw_id = b.bhw_id;

-- 8. PERFORMANCE INDEXES
CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_bhw ON assessments(bhw_id);
CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date);
CREATE INDEX IF NOT EXISTS idx_clinical_flags_asmnt ON clinical_flags(assessment_id);
CREATE INDEX IF NOT EXISTS idx_shap_asmnt ON shap_explanations(assessment_id);