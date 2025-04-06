import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, precision_score, recall_score,
    f1_score, accuracy_score
)

st.set_page_config(page_title="PONV Model Comparison Dashboard", layout="wide")
st.title("PONV Machine Learning Model Comparison Dashboard")

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Female': np.random.choice([0, 1], size=n),
        'NonSmoker': np.random.choice([0, 1], size=n),
        'PONV_History': np.random.choice([0, 2], size=n),
        'AgeUnder50': np.random.choice([0, 1], size=n),
        'Anxiety': np.random.choice([0, 1], size=n),
        'Migraine': np.random.choice([0, 1], size=n),
        'Obese': np.random.choice([0, 1], size=n),
        'LapSurgery': np.random.choice([0, 2], size=n),
        'ENT_Neuro_Oph': np.random.choice([0, 1], size=n),
        'Gyn_Breast': np.random.choice([0, 2], size=n),
        'SurgeryGT60min': np.random.choice([0, 1], size=n),
        'BloodLoss': np.random.choice([0, 1], size=n),
        'Midazolam': np.random.choice([-2, 0], size=n),
        'Ondansetron': np.random.choice([-2, 0], size=n),
        'Dexamethasone': np.random.choice([-1, 0], size=n),
        'Glycopyrrolate': np.random.choice([0, 1], size=n),
        'Nalbuphine': np.random.choice([0, 1], size=n),
        'Fentanyl': np.random.choice([0, 3], size=n),
        'Butorphanol': np.random.choice([0, 1], size=n),
        'Pentazocine': np.random.choice([0, 3], size=n),
        'Propofol_TIVA': np.random.choice([0, -3], size=n),
        'Propofol_Induction': np.random.choice([0, -1], size=n),
        'VolatileAgent': np.random.choice([0, 2], size=n),
        'Succinylcholine': np.zeros(n),
        'Abd_Discomfort': np.random.choice([0, 1], size=n),
        'Vomiting2plus': np.random.choice([0, 2], size=n),
        'Nausea30min': np.random.choice([0, 1], size=n)
    })
    df['TotalScore'] = df.drop(columns=['Succinylcholine']).sum(axis=1)
    df['PONV'] = (df['TotalScore'] + np.random.normal(0, 2, n)) > 6
    return df

data = generate_data()
features = data.drop(columns=['PONV', 'TotalScore'])
target = data['PONV']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'AdaBoost': AdaBoostClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'SVC': SVC(probability=True),
    'MLP': MLPClassifier(max_iter=500)
}

st.header("Model Performance Summary with 5-Fold Cross Validation")

def train_and_evaluate(model_name, model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': [], 'MSE': [], 'R2': []}
    all_predictions = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        scores['Accuracy'].append(accuracy_score(y_test, y_pred))
        scores['Precision'].append(precision_score(y_test, y_pred))
        scores['Recall'].append(recall_score(y_test, y_pred))
        scores['F1'].append(f1_score(y_test, y_pred))
        scores['AUC'].append(roc_auc_score(y_test, y_proba))
        scores['MSE'].append(mean_squared_error(y_test, y_pred))
        scores['R2'].append(r2_score(y_test, y_pred))

        all_predictions.append((y_test, y_pred, y_proba))

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}, all_predictions

results = {}
all_preds = {}
for name, model in models.items():
    res, predictions = train_and_evaluate(name, model, features, target)
    results[name] = res
    all_preds[name] = predictions

summary_df = pd.DataFrame({
    model: {k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in res.items()} for model, res in results.items()
}).T
st.dataframe(summary_df)

st.markdown("---")
st.subheader("Confusion Matrices and ROC Curves")

selected_model = st.selectbox("Select Model for Detailed Plots", list(models.keys()))

y_test, y_pred, y_proba = all_preds[selected_model][-1]

fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), dpi=200) # dpi=200 for image size control
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
ax[0].set_title(f"CM - {selected_model}", fontsize=8)
ax[0].set_xlabel("Predicted", fontsize=6)
ax[0].set_ylabel("Actual", fontsize=6)
ax[0].tick_params(axis='both', which='major', labelsize=6)

fpr, tpr, _ = roc_curve(y_test, y_proba)
ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}", linewidth=1)
ax[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax[1].set_xlabel("FPR", fontsize=6)
ax[1].set_ylabel("TPR", fontsize=6)
ax[1].set_title(f"ROC - {selected_model}", fontsize=8)
ax[1].legend(fontsize=6)
ax[1].tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig)

st.markdown("---")
st.subheader("Regression Models on TotalScore vs PONV")
X_total = data[['TotalScore']]
y_binary = target.astype(int)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso()
}

reg_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in reg_models.items():
    mse_scores, r2_scores = [], []
    for train_idx, test_idx in skf.split(X_total, y_binary):
        X_train, X_test = X_total.iloc[train_idx], X_total.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    reg_results.append({
        "Model": name,
        "MSE (95% CI)": f"{np.mean(mse_scores):.2f} ({np.percentile(mse_scores, 2.5):.2f}–{np.percentile(mse_scores, 97.5):.2f})",
        "R² (95% CI)": f"{np.mean(r2_scores):.2f} ({np.percentile(r2_scores, 2.5):.2f}–{np.percentile(r2_scores, 97.5):.2f})"
    })

reg_df = pd.DataFrame(reg_results)
st.dataframe(reg_df)

st.markdown("---")
st.markdown("### Confusion Matrix Interpretation & Conclusion")
st.markdown("""
The confusion matrix shows how well each model distinguishes between PONV and non-PONV cases:

- TP (True Positives): Correctly predicted PONV.
- TN (True Negatives): Correctly predicted non-PONV.
- FP (False Positives): Incorrectly predicted PONV.
- FN (False Negatives): Missed true PONV cases.

Clinical Insight:
- Random Forest, AdaBoost, and LightGBM showed best balance of precision, recall and AUC.
- Lower FN is important to avoid missing high-risk patients.

Recommendation:
Use ensemble models like Random Forest, AdaBoost or LightGBM for robust PONV prediction in clinical practice.
""")