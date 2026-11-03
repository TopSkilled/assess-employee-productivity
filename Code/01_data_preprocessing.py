"""
01_data_preprocessing.py
-------------------------
Loads the raw dataset, cleans it, engineers features,
scales numerical columns, and returns train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os, joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH    = os.path.join('datasets', 'employee_data.csv')
MODELS_DIR  = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    # Numerical → median imputation
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical → mode imputation
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    before = len(df)
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"Outlier removal: {before} → {len(df)} rows")
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in ['department', 'gender']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid divide-by-zero
    df['productivity_index'] = df['tasks_completed'] / (df['hours_worked_per_week'] + 1e-5)
    df['engagement_score']   = (df['satisfaction_score'] + df['communication_score']) / 2
    df['efficiency_ratio']   = (1 - df['error_rate']) * df['tasks_completed']
    return df


def preprocess(path: str = RAW_PATH):
    df = load_data(path)
    df = handle_missing(df)

    # Drop non-feature columns
    df.drop(columns=['employee_id'], inplace=True)

    df = encode_categoricals(df)
    df = feature_engineering(df)

    outlier_cols = [
        'hours_worked_per_week', 'tasks_completed',
        'overtime_hours', 'projects_completed'
    ]
    df = remove_outliers_iqr(df, outlier_cols)

    # ── Split features / target ────────────────────────────────────────────────
    X = df.drop(columns=['productivity_label'])
    y = df['productivity_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler for inference
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, 'feature_names.pkl'))

    print(f"\nTrain size: {X_train_sc.shape}, Test size: {X_test_sc.shape}")
    print(f"Features: {X.shape[1]}")
    print("\nClass distribution (train):")
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    print(pd.Series(y_train).map(label_map).value_counts())

    return X_train_sc, X_test_sc, y_train.values, y_test.values, list(X.columns)


if __name__ == '__main__':
    preprocess()
