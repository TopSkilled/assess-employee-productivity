"""
03_model_evaluation.py
-----------------------
Evaluates all trained models on the test set.
Produces and saves:
  • Classification report
  • Confusion matrix heatmap
  • ROC-AUC curves (OvR)
  • Model comparison bar chart
  • Feature importance plot (best tree-based model)
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                     # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

from model_training import run_training   # runs the full pipeline

warnings.filterwarnings('ignore')

OUTPUTS_DIR  = 'outputs'
MODELS_DIR   = 'models'
CLASS_LABELS = ['Low', 'Medium', 'High']
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']


def evaluate_all(trained, X_test, y_test) -> pd.DataFrame:
    rows = []
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("=" * 70)

    for name, model in trained.items():
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = np.nan

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        rows.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc
        })
        print(f"{name:<25} {acc:.4f} {prec:.4f} {rec:.4f} {f1:.4f} {auc:.4f}")

    print("=" * 70)
    results_df = pd.DataFrame(rows).set_index('Model')
    return results_df


def plot_model_comparison(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(results_df))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width,
               label=metric, color=PALETTE[i], alpha=0.88)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df.index, rotation=20, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")

    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))


def plot_roc_auc(model, X_test, y_test, model_name: str):
    n_classes = 3
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = model.predict_proba(X_test)

    colors = ['#4C72B0', '#DD8452', '#55A868']
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, clr) in enumerate(zip(CLASS_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=clr, lw=2,
                label=f'{cls} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC-AUC Curves — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'roc_auc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"Feature importance not available for {model_name}")
        return

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                   color='#4C72B0', edgecolor='white', alpha=0.88)
    ax.set_xlabel('Feature Importance Score', fontsize=12)
    ax.set_title(f'Top 15 Feature Importances — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, fi_df['Importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def run_evaluation():
    trained, results, X_train, X_test, y_train, y_test, feature_names = run_training()

    results_df = evaluate_all(trained, X_test, y_test)
    print("\nFull Metrics Table:")
    print(results_df.round(4).to_string())

    # ── Best model ────────────────────────────────────────────────────────────
    best_name  = results_df['F1-Score'].idxmax()
    best_model = trained[best_name]
    print(f"\nBest Model by F1-Score: {best_name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_model_comparison(results_df)
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_roc_auc(best_model, X_test, y_test, best_name)
    plot_feature_importance(best_model, feature_names, best_name)

    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'model_results.csv'))
    print(f"\nResults CSV saved → {os.path.join(OUTPUTS_DIR, 'model_results.csv')}")

    return results_df, best_model, best_name


if __name__ == '__main__':
    run_evaluation()
