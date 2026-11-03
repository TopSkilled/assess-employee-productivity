"""
generate_dataset.py
-------------------
Generates a synthetic employee productivity dataset (1000 records)
and saves it to datasets/employee_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

departments = ['IT', 'HR', 'Sales', 'Finance', 'Operations']
genders = ['Male', 'Female']

# ── Base features ──────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'employee_id':            [f'EMP{str(i).zfill(4)}' for i in range(1, N + 1)],
    'department':             np.random.choice(departments, N),
    'gender':                 np.random.choice(genders, N),
    'age':                    np.random.randint(22, 61, N),
    'experience_years':       np.random.randint(1, 36, N),
    'hours_worked_per_week':  np.round(np.random.normal(42, 7, N).clip(20, 70), 1),
    'tasks_completed':        np.random.randint(10, 100, N),
    'attendance_rate':        np.round(np.random.normal(88, 10, N).clip(50, 100), 1),
    'training_hours':         np.random.randint(0, 80, N),
    'satisfaction_score':     np.round(np.random.uniform(1, 10, N), 1),
    'team_size':              np.random.randint(3, 20, N),
    'overtime_hours':         np.round(np.random.exponential(3, N).clip(0, 20), 1),
    'error_rate':             np.round(np.random.uniform(0, 0.4, N), 3),
    'communication_score':    np.round(np.random.uniform(1, 10, N), 1),
    'manager_rating':         np.round(np.random.uniform(1, 5, N), 1),
    'projects_completed':     np.random.randint(0, 15, N),
})

# ── Derived productivity score (used only to assign labels realistically) ───────
score = (
    0.25 * (df['tasks_completed'] / 100) +
    0.15 * (df['attendance_rate'] / 100) +
    0.15 * (df['manager_rating'] / 5) +
    0.10 * (df['satisfaction_score'] / 10) +
    0.10 * (df['communication_score'] / 10) +
    0.10 * (df['projects_completed'] / 15) +
    0.10 * (df['training_hours'] / 80) +
    0.05 * (df['experience_years'] / 35) -
    0.10 * df['error_rate'] -
    0.05 * (df['overtime_hours'] / 20)
)

# Add noise so labels aren't perfectly separable
score += np.random.normal(0, 0.05, N)
score = score.clip(0, 1)

# ── Assign labels ──────────────────────────────────────────────────────────────
conditions = [score < 0.38, score < 0.65]
choices = [0, 1]                            # 0=Low, 1=Medium, 2=High
df['productivity_label'] = np.select(conditions, choices, default=2)

# ── Introduce a small % of missing values for realism ─────────────────────────
for col in ['training_hours', 'satisfaction_score', 'communication_score']:
    mask = np.random.choice([True, False], size=N, p=[0.03, 0.97])
    df.loc[mask, col] = np.nan

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('datasets', exist_ok=True)
out_path = os.path.join('datasets', 'employee_data.csv')
df.to_csv(out_path, index=False)

print(f"Dataset saved → {out_path}")
print(f"Shape: {df.shape}")
print("\nClass distribution:")
label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
print(df['productivity_label'].map(label_map).value_counts())
print("\nSample rows:")
print(df.head(5).to_string())
