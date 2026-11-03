"""
04_predict.py
--------------
Loads the saved best model + scaler and predicts productivity
for new employee input. Can be run as a script or imported.
"""

import os, joblib
import numpy as np
import pandas as pd

MODELS_DIR = 'models'


def load_artifacts():
    best_model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    best_name     = joblib.load(os.path.join(MODELS_DIR, 'best_model_name.pkl'))
    return best_model, scaler, feature_names, best_name


def predict_productivity(employee_dict: dict) -> dict:
    """
    Parameters
    ----------
    employee_dict : dict
        Keys must match the training feature names.
        Categorical fields:
            department  → IT=0, HR=1, Sales=2, Finance=3, Operations=4
                          (use numeric codes after encoding)
            gender      → Female=0, Male=1

    Returns
    -------
    dict with predicted_label, confidence, and class_probabilities
    """
    best_model, scaler, feature_names, best_name = load_artifacts()

    df = pd.DataFrame([employee_dict])

    # Derived features must be re-created
    df['productivity_index'] = df['tasks_completed'] / (df['hours_worked_per_week'] + 1e-5)
    df['engagement_score']   = (df['satisfaction_score'] + df['communication_score']) / 2
    df['efficiency_ratio']   = (1 - df['error_rate']) * df['tasks_completed']

    df = df[feature_names]                       # enforce column order
    X_sc = scaler.transform(df)

    pred       = best_model.predict(X_sc)[0]
    proba      = best_model.predict_proba(X_sc)[0]

    label_map  = {0: 'Low', 1: 'Medium', 2: 'High'}
    confidence = round(float(proba[pred]) * 100, 2)

    return {
        'model_used':         best_name,
        'predicted_label':    label_map[pred],
        'confidence_pct':     confidence,
        'class_probabilities': {
            'Low':    round(float(proba[0]) * 100, 2),
            'Medium': round(float(proba[1]) * 100, 2),
            'High':   round(float(proba[2]) * 100, 2),
        }
    }


if __name__ == '__main__':
    # ── Example new employee ──────────────────────────────────────────────────
    sample_employee = {
        'department':            2,      # Sales
        'gender':                1,      # Male
        'age':                   34,
        'experience_years':      8,
        'hours_worked_per_week': 44.5,
        'tasks_completed':       72,
        'attendance_rate':       92.0,
        'training_hours':        35,
        'satisfaction_score':    7.8,
        'team_size':             10,
        'overtime_hours':        4.2,
        'error_rate':            0.08,
        'communication_score':   8.1,
        'manager_rating':        4.2,
        'projects_completed':    9,
    }

    result = predict_productivity(sample_employee)

    print("\n── Prediction Result ──────────────────────────────────")
    print(f"  Model Used       : {result['model_used']}")
    print(f"  Predicted Label  : {result['predicted_label']}")
    print(f"  Confidence       : {result['confidence_pct']}%")
    print(f"  Class Probabilities:")
    for cls, pct in result['class_probabilities'].items():
        bar = '█' * int(pct / 5)
        print(f"    {cls:<8} {pct:>6.2f}%  {bar}")
    print("───────────────────────────────────────────────────────")
