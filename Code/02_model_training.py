"""
02_model_training.py
---------------------
Trains 6 ML classifiers, tunes the best one with GridSearchCV,
saves all trained models, and returns results dict.
"""

import os, time, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost                 import XGBClassifier

from data_preprocessing import preprocess          # reuse preprocessing

warnings.filterwarnings('ignore')

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, multi_class='multinomial', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42
        ),
        'KNN': KNeighborsClassifier(n_neighbors=7)
    }


def train_all(X_train, y_train) -> dict:
    models    = get_models()
    results   = {}
    trained   = {}

    print("=" * 55)
    print(f"{'Model':<25} {'CV Accuracy':>12} {'Time (s)':>10}")
    print("=" * 55)

    for name, model in models.items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        results[name] = {
            'cv_mean':  cv_scores.mean(),
            'cv_std':   cv_scores.std(),
            'time_s':   round(elapsed, 2)
        }
        trained[name] = model
        print(f"{name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}   {elapsed:>8.2f}s")

    print("=" * 55)
    return trained, results


def tune_best_model(trained: dict, results: dict, X_train, y_train):
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    print(f"\nTuning best model: {best_name}")

    if best_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth':    [8, 10, 12],
            'min_samples_split': [2, 5]
        }
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif best_name == 'XGBoost':
        param_grid = {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                             random_state=42, verbosity=0)
    else:
        print("Skipping tuning for this model type (not RF or XGBoost).")
        tuned = trained[best_name]
        return tuned, best_name

    gs = GridSearchCV(base, param_grid, cv=5, scoring='accuracy',
                      n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV score: {gs.best_score_:.4f}")
    return gs.best_estimator_, best_name


def save_models(trained: dict):
    for name, model in trained.items():
        fname = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
    print(f"\nAll {len(trained)} models saved to '{MODELS_DIR}/'")


def run_training():
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    trained, results = train_all(X_train, y_train)
    tuned_model, best_name = tune_best_model(trained, results, X_train, y_train)

    # Replace with tuned version
    trained[best_name] = tuned_model
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(best_name,   os.path.join(MODELS_DIR, 'best_model_name.pkl'))

    save_models(trained)

    return trained, results, X_train, X_test, y_train, y_test, feature_names


if __name__ == '__main__':
    run_training()
