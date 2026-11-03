"""
app.py — Employee Productivity Assessment Web Application
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib, os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing    import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Productivity Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SEED         = 42
CLASS_LABELS = ['Low', 'Medium', 'High']
LABEL_MAP    = {0: 'Low', 1: 'Medium', 2: 'High'}
PALETTE      = ['#C44E52', '#DD8452', '#55A868']
MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH    = os.path.join(os.path.dirname(__file__), 'datasets', 'employee_data.csv')
os.makedirs(MODELS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_dataset(n=1000):
    np.random.seed(SEED)
    departments = ['IT', 'HR', 'Sales', 'Finance', 'Operations']
    df = pd.DataFrame({
        'employee_id':            [f'EMP{str(i).zfill(4)}' for i in range(1, n + 1)],
        'department':             np.random.choice(departments, n),
        'gender':                 np.random.choice(['Male', 'Female'], n),
        'age':                    np.random.randint(22, 61, n),
        'experience_years':       np.random.randint(1, 36, n),
        'hours_worked_per_week':  np.round(np.random.normal(42, 7, n).clip(20, 70), 1),
        'tasks_completed':        np.random.randint(10, 100, n),
        'attendance_rate':        np.round(np.random.normal(88, 10, n).clip(50, 100), 1),
        'training_hours':         np.random.randint(0, 80, n),
        'satisfaction_score':     np.round(np.random.uniform(1, 10, n), 1),
        'team_size':              np.random.randint(3, 20, n),
        'overtime_hours':         np.round(np.random.exponential(3, n).clip(0, 20), 1),
        'error_rate':             np.round(np.random.uniform(0, 0.4, n), 3),
        'communication_score':    np.round(np.random.uniform(1, 10, n), 1),
        'manager_rating':         np.round(np.random.uniform(1, 5, n), 1),
        'projects_completed':     np.random.randint(0, 15, n),
    })
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
    ) + np.random.normal(0, 0.05, n)
    score = score.clip(0, 1)
    df['productivity_label'] = np.select([score < 0.38, score < 0.65], [0, 1], default=2)
    for col in ['training_hours', 'satisfaction_score', 'communication_score']:
        mask = np.random.choice([True, False], size=n, p=[0.03, 0.97])
        df.loc[mask, col] = np.nan
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


@st.cache_data
def preprocess_data(df):
    data = df.copy().drop(columns=['employee_id'])
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].median(), inplace=True)
    le = LabelEncoder()
    for col in ['department', 'gender']:
        data[col] = le.fit_transform(data[col])
    data['productivity_index'] = data['tasks_completed'] / (data['hours_worked_per_week'] + 1e-5)
    data['engagement_score']   = (data['satisfaction_score'] + data['communication_score']) / 2
    data['efficiency_ratio']   = (1 - data['error_rate']) * data['tasks_completed']
    for col in ['hours_worked_per_week', 'tasks_completed', 'overtime_hours', 'projects_completed']:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]
    data.reset_index(drop=True, inplace=True)
    X = data.drop(columns=['productivity_label'])
    y = data['productivity_label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler,          os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns), scaler


def train_models(X_train, y_train, selected_models):
    configs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Decision Tree':        DecisionTreeClassifier(max_depth=8, random_state=SEED),
        'Random Forest':        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        'XGBoost':              XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                              eval_metric='mlogloss',
                                              random_state=SEED, verbosity=0),
        'SVM':                  SVC(kernel='rbf', probability=True, random_state=SEED),
        'KNN':                  KNeighborsClassifier(n_neighbors=7),
    }
    trained, cv_res = {}, {}
    progress = st.progress(0)
    status   = st.empty()
    for idx, name in enumerate(selected_models):
        model = configs[name]
        status.text(f"Training {name}...")
        t0 = time.time()
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
        cv_res[name] = {'cv_mean': cv.mean(), 'cv_std': cv.std(), 'time_s': round(time.time() - t0, 2)}
        trained[name] = model
        progress.progress((idx + 1) / len(selected_models))
    status.empty()
    progress.empty()
    return trained, cv_res


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='text-align:center; font-size:64px; line-height:1.1;'>📊</div>",
        unsafe_allow_html=True
    )
    st.title("Employee Productivity")
    st.caption("Performance Analytics ML App")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📂 Dataset", "📊 EDA", "🤖 Train Models", "📈 Evaluation", "🔮 Predict"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Built with Python · scikit-learn · XGBoost · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("📊 Employee Productivity Assessment")
    st.subheader("Using Performance Analytics Models")
    st.markdown("""
    This application predicts **employee productivity level** (Low / Medium / High) using
    machine learning models trained on behavioral, workload, and engagement metrics.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "1,000", "employees")
    col2.metric("Features", "19", "including engineered")
    col3.metric("ML Models", "6", "classifiers")
    col4.metric("Target Classes", "3", "Low · Medium · High")

    st.divider()

    st.markdown("### How to Use")
    steps = {
        "📂 Dataset":     "View and explore the raw employee dataset",
        "📊 EDA":         "Visualize distributions, correlations & class balance",
        "🤖 Train Models":"Select models, train them and see cross-validation scores",
        "📈 Evaluation":  "Compare models on test set — accuracy, F1, ROC-AUC",
        "🔮 Predict":     "Enter a new employee's details and get an instant prediction",
    }
    for step, desc in steps.items():
        st.markdown(f"**{step}** — {desc}")

    st.divider()
    st.markdown("### Productivity Classes")
    c1, c2, c3 = st.columns(3)
    c1.error("🔴 Low\nAttendance < 75%, high error rate, low task completion")
    c2.warning("🟡 Medium\nAverage performance, moderate engagement")
    c3.success("🟢 High\nConsistent delivery, high manager rating, low errors")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Dataset":
    st.title("📂 Dataset")

    if st.button("Generate / Reload Dataset", type="primary"):
        st.cache_data.clear()

    df = generate_dataset()

    st.success(f"Dataset loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

    with col2:
        st.subheader("Class Distribution")
        counts = df['productivity_label'].map(LABEL_MAP).value_counts()
        for label, cnt in counts.items():
            pct = cnt / len(df) * 100
            color = "🔴" if label == "Low" else ("🟡" if label == "Medium" else "🟢")
            st.metric(f"{color} {label}", f"{cnt}", f"{pct:.1f}%")

        st.subheader("Missing Values")
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if len(mv):
            st.dataframe(mv.rename("Count").to_frame(), use_container_width=True)
        else:
            st.info("No missing values")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    df = generate_dataset()
    df['label'] = df['productivity_label'].map(LABEL_MAP)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Class Balance", "Feature Distributions", "Box Plots", "Correlation Heatmap"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        counts = df['label'].value_counts()

        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
                   colors=PALETTE, startangle=140,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            ax.set_title('Productivity Label Distribution', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            dept_prod = df.groupby(['department', 'label']).size().unstack(fill_value=0)
            dept_prod.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='white', alpha=0.88)
            ax.set_title('Productivity by Department', fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=20)
            ax.legend(title='Productivity')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        num_features = ['hours_worked_per_week', 'tasks_completed',
                        'attendance_rate', 'manager_rating',
                        'error_rate', 'satisfaction_score',
                        'training_hours', 'experience_years']
        chosen = st.multiselect("Select features to plot", num_features, default=num_features[:6])
        if chosen:
            cols_per_row = 3
            rows = (len(chosen) + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
            axes = np.array(axes).flatten()
            clrs = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#4C72B0', '#DD8452']
            for ax, col, clr in zip(axes, chosen, clrs):
                ax.hist(df[col].dropna(), bins=25, color=clr, edgecolor='white', alpha=0.85)
                ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Frequency')
            for ax in axes[len(chosen):]:
                ax.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        feature_bp = st.selectbox("Select feature for box plot",
                                  ['hours_worked_per_week', 'tasks_completed',
                                   'attendance_rate', 'manager_rating',
                                   'error_rate', 'satisfaction_score',
                                   'training_hours', 'projects_completed'])
        fig, ax = plt.subplots(figsize=(7, 4))
        order = ['Low', 'Medium', 'High']
        pal = {'Low': '#C44E52', 'Medium': '#DD8452', 'High': '#55A868'}
        sns.boxplot(data=df, x='label', y=feature_bp, order=order, palette=pal, ax=ax)
        ax.set_title(f'{feature_bp.replace("_", " ").title()} by Productivity Level',
                     fontweight='bold')
        ax.set_xlabel('Productivity Level')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        num_df = df.select_dtypes(include=np.number).drop(columns=['productivity_label'])
        corr   = num_df.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Train Models":
    st.title("🤖 Model Training")

    df = generate_dataset()

    st.subheader("Configure Training")
    col1, col2 = st.columns(2)
    with col1:
        selected = st.multiselect(
            "Select models to train",
            ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM', 'KNN'],
            default=['Random Forest', 'XGBoost', 'Logistic Regression']
        )
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        st.caption(f"Train: {100 - test_size}% | Test: {test_size}%")

    if not selected:
        st.warning("Select at least one model.")
        st.stop()

    if st.button("🚀 Train Selected Models", type="primary"):
        with st.spinner("Preprocessing data..."):
            X_train_sc, X_test_sc, y_train, y_test, feature_names, scaler = preprocess_data(df)

        st.info(f"Train: {X_train_sc.shape[0]} samples | Test: {X_test_sc.shape[0]} samples | Features: {X_train_sc.shape[1]}")

        st.subheader("Training Progress")
        trained, cv_res = train_models(X_train_sc, y_train, selected)

        # Save to session state
        st.session_state['trained']       = trained
        st.session_state['X_test_sc']     = X_test_sc
        st.session_state['y_test']        = y_test
        st.session_state['feature_names'] = feature_names
        st.session_state['scaler']        = scaler

        # Save best model
        best_name = max(cv_res, key=lambda k: cv_res[k]['cv_mean'])
        joblib.dump(trained[best_name],  os.path.join(MODELS_DIR, 'best_model.pkl'))
        joblib.dump(best_name,           os.path.join(MODELS_DIR, 'best_model_name.pkl'))

        st.success(f"Training complete! Best model: **{best_name}** (CV: {cv_res[best_name]['cv_mean']:.4f})")

        st.subheader("Cross-Validation Results (5-Fold)")
        cv_df = pd.DataFrame({
            'Model':       list(cv_res.keys()),
            'CV Mean Acc': [v['cv_mean'] for v in cv_res.values()],
            'CV Std':      [v['cv_std']  for v in cv_res.values()],
            'Time (s)':    [v['time_s']  for v in cv_res.values()],
        }).sort_values('CV Mean Acc', ascending=False).reset_index(drop=True)
        st.dataframe(cv_df.style.highlight_max(subset=['CV Mean Acc'], color='#d4edda').format({
            'CV Mean Acc': '{:.4f}', 'CV Std': '{:.4f}', 'Time (s)': '{:.2f}'
        }), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        clrs = ['#55A868' if n == best_name else '#4C72B0' for n in cv_df['Model']]
        bars = ax.barh(cv_df['Model'], cv_df['CV Mean Acc'], color=clrs,
                       edgecolor='white', alpha=0.88)
        ax.set_xlabel('CV Accuracy')
        ax.set_title('Cross-Validation Accuracy by Model', fontweight='bold')
        ax.set_xlim(0, 1.1)
        for bar, val in zip(bars, cv_df['CV Mean Acc']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation")

    if 'trained' not in st.session_state:
        st.warning("No trained models found. Go to **Train Models** first.")
        st.stop()

    trained       = st.session_state['trained']
    X_test_sc     = st.session_state['X_test_sc']
    y_test        = st.session_state['y_test']
    feature_names = st.session_state['feature_names']

    # Compute metrics
    rows = []
    for name, model in trained.items():
        y_pred = model.predict(X_test_sc)
        try:
            y_prob = model.predict_proba(X_test_sc)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = np.nan
        rows.append({
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'ROC-AUC':   auc,
        })

    results_df = pd.DataFrame(rows).set_index('Model').round(4)
    best_name  = results_df['F1-Score'].idxmax()
    best_model = trained[best_name]

    st.subheader("Test-Set Metrics")
    st.dataframe(
        results_df.style
            .highlight_max(color='#d4edda')
            .highlight_min(color='#f8d7da')
            .format('{:.4f}'),
        use_container_width=True
    )
    st.success(f"Best Model: **{best_name}** (F1 = {results_df.loc[best_name, 'F1-Score']:.4f})")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model Comparison", "Confusion Matrix", "ROC-AUC", "Feature Importance"]
    )

    with tab1:
        metrics_plot = ['Accuracy', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(results_df))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        clrs_bar = ['#4C72B0', '#DD8452', '#55A868']
        for i, (metric, clr) in enumerate(zip(metrics_plot, clrs_bar)):
            ax.bar(x + i * width, results_df[metric], width,
                   label=metric, color=clr, alpha=0.88)
        ax.set_xticks(x + width)
        ax.set_xticklabels(results_df.index, rotation=15, ha='right')
        ax.set_ylim(0, 1.12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        chosen_model_name = st.selectbox("Select model", list(trained.keys()),
                                         index=list(trained.keys()).index(best_name))
        chosen_model = trained[chosen_model_name]
        y_pred = chosen_model.predict(X_test_sc)
        cm = confusion_matrix(y_test, y_pred)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                        linewidths=0.5, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix — {chosen_model_name}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            report = classification_report(y_test, y_pred,
                                           target_names=CLASS_LABELS, output_dict=True)
            report_df = pd.DataFrame(report).T.round(3)
            st.subheader("Classification Report")
            st.dataframe(report_df, use_container_width=True)

    with tab3:
        try:
            y_bin  = label_binarize(y_test, classes=[0, 1, 2])
            y_prob = best_model.predict_proba(X_test_sc)
            fig, ax = plt.subplots(figsize=(8, 6))
            clrs_roc = ['#4C72B0', '#DD8452', '#55A868']
            for i, (cls, clr) in enumerate(zip(CLASS_LABELS, clrs_roc)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=clr, lw=2, label=f'{cls} (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC-AUC — {best_name}', fontsize=13, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"ROC-AUC plot error: {e}")

    with tab4:
        try:
            importances = best_model.feature_importances_
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                      .sort_values('Importance', ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                           color='#4C72B0', edgecolor='white', alpha=0.88)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top Feature Importances — {best_name}', fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f'{val:.4f}', va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except AttributeError:
            st.info("Feature importance is only available for tree-based models.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict":
    st.title("🔮 Predict Employee Productivity")

    # Load model from file or session state
    model_source = "session"
    if 'trained' in st.session_state:
        trained       = st.session_state['trained']
        scaler        = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        best_name = max(trained, key=lambda k:
            f1_score(st.session_state['y_test'],
                     trained[k].predict(st.session_state['X_test_sc']),
                     average='weighted', zero_division=0))
        best_model = trained[best_name]
    elif os.path.exists(os.path.join(MODELS_DIR, 'best_model.pkl')):
        best_model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        best_name     = joblib.load(os.path.join(MODELS_DIR, 'best_model_name.pkl'))
        scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
        model_source  = "file"
    else:
        st.warning("No trained model found. Go to **Train Models** first.")
        st.stop()

    st.info(f"Using model: **{best_name}** (loaded from {model_source})")

    st.subheader("Enter Employee Details")

    dept_map   = {'IT': 2, 'HR': 1, 'Sales': 4, 'Finance': 0, 'Operations': 3}
    gender_map = {'Male': 1, 'Female': 0}

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Personal Info**")
            department  = st.selectbox("Department", list(dept_map.keys()))
            gender      = st.selectbox("Gender", list(gender_map.keys()))
            age         = st.slider("Age", 22, 60, 34)
            experience  = st.slider("Experience (years)", 1, 35, 8)

        with c2:
            st.markdown("**Work Metrics**")
            hours_worked  = st.slider("Hours Worked / Week", 20.0, 70.0, 44.0, 0.5)
            tasks         = st.slider("Tasks Completed / Month", 10, 100, 65)
            attendance    = st.slider("Attendance Rate (%)", 50.0, 100.0, 90.0, 0.5)
            overtime      = st.slider("Overtime Hours / Week", 0.0, 20.0, 3.0, 0.5)
            projects      = st.slider("Projects Completed / Year", 0, 15, 8)

        with c3:
            st.markdown("**Performance & Engagement**")
            training      = st.slider("Training Hours / Year", 0, 80, 30)
            satisfaction  = st.slider("Satisfaction Score (1–10)", 1.0, 10.0, 7.5, 0.1)
            team_size     = st.slider("Team Size", 3, 20, 10)
            error_rate    = st.slider("Error Rate (0–1)", 0.0, 0.4, 0.08, 0.01)
            comm_score    = st.slider("Communication Score (1–10)", 1.0, 10.0, 7.8, 0.1)
            mgr_rating    = st.slider("Manager Rating (1–5)", 1.0, 5.0, 4.0, 0.1)

        submitted = st.form_submit_button("🔮 Predict Productivity", type="primary", use_container_width=True)

    if submitted:
        emp = {
            'department':            dept_map[department],
            'gender':                gender_map[gender],
            'age':                   age,
            'experience_years':      experience,
            'hours_worked_per_week': hours_worked,
            'tasks_completed':       tasks,
            'attendance_rate':       attendance,
            'training_hours':        training,
            'satisfaction_score':    satisfaction,
            'team_size':             team_size,
            'overtime_hours':        overtime,
            'error_rate':            error_rate,
            'communication_score':   comm_score,
            'manager_rating':        mgr_rating,
            'projects_completed':    projects,
        }

        new_df = pd.DataFrame([emp])
        new_df['productivity_index'] = new_df['tasks_completed'] / (new_df['hours_worked_per_week'] + 1e-5)
        new_df['engagement_score']   = (new_df['satisfaction_score'] + new_df['communication_score']) / 2
        new_df['efficiency_ratio']   = (1 - new_df['error_rate']) * new_df['tasks_completed']
        new_df = new_df[feature_names]
        new_sc = scaler.transform(new_df)

        pred  = best_model.predict(new_sc)[0]
        proba = best_model.predict_proba(new_sc)[0]

        label_colors = {0: ('🔴', 'error'), 1: ('🟡', 'warning'), 2: ('🟢', 'success')}
        icon, color  = label_colors[pred]

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns([1, 1, 2])
        with r1:
            if color == 'error':
                st.error(f"{icon} **{LABEL_MAP[pred]}** Productivity")
            elif color == 'warning':
                st.warning(f"{icon} **{LABEL_MAP[pred]}** Productivity")
            else:
                st.success(f"{icon} **{LABEL_MAP[pred]}** Productivity")
            st.metric("Confidence", f"{proba[pred]*100:.1f}%")

        with r2:
            st.markdown("**Class Probabilities**")
            for i, cls in enumerate(CLASS_LABELS):
                st.metric(cls, f"{proba[i]*100:.1f}%")

        with r3:
            fig, ax = plt.subplots(figsize=(6, 3))
            bar_clrs = [PALETTE[i] for i in range(3)]
            bars = ax.bar(CLASS_LABELS, proba * 100, color=bar_clrs,
                          edgecolor='white', alpha=0.9)
            for bar, val in zip(bars, proba * 100):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5, f'{val:.1f}%',
                        ha='center', fontweight='bold', fontsize=10)
            ax.set_ylim(0, 115)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Predicted Class Probabilities', fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            # Highlight predicted class
            bars[pred].set_edgecolor('black')
            bars[pred].set_linewidth(2.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Insight
        st.divider()
        st.subheader("Key Insights")
        insights = []
        if attendance < 75:
            insights.append("⚠️ Attendance rate is below 75% — a strong predictor of low productivity.")
        if error_rate > 0.25:
            insights.append("⚠️ High error rate detected — consider targeted training.")
        if mgr_rating >= 4.0:
            insights.append("✅ Strong manager rating — positive performance indicator.")
        if tasks >= 70:
            insights.append("✅ High task completion rate — excellent output.")
        if training < 10:
            insights.append("💡 Low training hours — increasing L&D investment may help.")
        if satisfaction >= 8:
            insights.append("✅ High satisfaction score — employee is engaged.")
        if not insights:
            insights.append("ℹ️ Performance metrics are within normal range.")
        for i in insights:
            st.markdown(i)
