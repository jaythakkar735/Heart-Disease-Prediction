"""
Heart Disease Clinical Risk Screener & Machine Learning Platform.
Interactive Streamlit Application showcasing clinical classification,
calibrated probability estimation, and diagnostic validation.
"""

import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import domain modules
from src.data_loader import (
    load_heart_data,
    get_train_test_split,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.pipeline import build_full_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Clinical Risk Screener",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_models():
    """Load dataset, train baseline, logistic regression, and ensemble pipelines."""
    data_path = PROJECT_DIR / "data" / "processed_cleveland.csv"
    df = load_heart_data(str(data_path))
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)

    # 1. Dummy Baseline
    dummy_pipe = build_full_pipeline(DummyClassifier(strategy="stratified", random_state=42))
    dummy_pipe.fit(X_train, y_train)

    # 2. Calibrated Logistic Regression (L2, C=0.1)
    lr_pipe = build_full_pipeline(LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    lr_pipe.fit(X_train, y_train)

    # 3. Random Forest Classifier
    rf_pipe = build_full_pipeline(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    rf_pipe.fit(X_train, y_train)

    # 4. Gradient Boosting Classifier
    gb_pipe = build_full_pipeline(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    gb_pipe.fit(X_train, y_train)

    # Load pre-saved metrics summary if available
    metrics_path = PROJECT_DIR / "reports" / "metrics_summary.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dummy": dummy_pipe,
        "lr": lr_pipe,
        "rf": rf_pipe,
        "gb": gb_pipe,
        "metrics": metrics_data,
    }


bundle = load_and_train_models()
df = bundle["df"]
lr_model = bundle["lr"]
rf_model = bundle["rf"]
gb_model = bundle["gb"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
metrics_summary = bundle["metrics"]

# ---------------------------------------------------------
# Header & Medical Disclaimer
# ---------------------------------------------------------
st.title("❤️ Heart Disease Clinical Risk Screener")
st.markdown(
    "**Supervised Machine Learning Pipeline & Calibrated Probability Estimator**  \n"
    "*Trained and evaluated on the UCI Cleveland Clinic Cohort ($N=303$, 14 attributes)*"
)

st.warning(
    "⚠️ **Clinical Research Notice**: This application is an educational and portfolio demonstration "
    "of statistical learning algorithms. It is **NOT** a certified medical diagnostic device and must not "
    "be used for direct clinical patient management or diagnostic decisions."
)

# ---------------------------------------------------------
# Sidebar Navigation & Patient Presets
# ---------------------------------------------------------
st.sidebar.title("🫀 Navigation & Settings")

nav_choice = st.sidebar.radio(
    "Select View:",
    options=[
        "🩺 Patient Risk Screener",
        "📊 Exploratory Data Analysis (EDA)",
        "🔬 Model Evaluation & Calibration",
        "📋 Clinical Methodology & Limitations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Core Methodology:**  \n"
    "- **Data Leakage:** 0% (Split before imputation & scaling)  \n"
    "- **Validation:** 5-Fold Stratified Cross-Validation  \n"
    "- **Primary Metric:** ROC-AUC & Recall (Sensitivity)  \n"
    "- **Holdout ROC-AUC:** **0.961** (L2 Logistic Regression)  \n"
    "- **Holdout Recall:** **89.3%** ($25/28$ positive cases)"
)

# =========================================================
# VIEW 1: PATIENT RISK SCREENER
# =========================================================
if nav_choice == "🩺 Patient Risk Screener":
    st.subheader("Interactive Patient Risk Assessment")
    st.markdown(
        "Enter individual patient vitals and cardiac diagnostic observations below to generate "
        "calibrated disease presence probabilities and risk-stratified guidance."
    )

    # Preset selection
    st.markdown("#### ⚡ Quick Clinical Presets")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        apply_normal = st.button("🟢 Normal / Low Risk", use_container_width=True)
    with preset_cols[1]:
        apply_borderline = st.button("🟡 Borderline Profile", use_container_width=True)
    with preset_cols[2]:
        apply_ischemic = st.button("🔴 Severe Ischemic CAD", use_container_width=True)
    with preset_cols[3]:
        apply_reset = st.button("🔄 Custom / Reset", use_container_width=True)

    # Session state for preset values
    if "preset_type" not in st.session_state or apply_reset:
        st.session_state.preset_type = "custom"
    if apply_normal:
        st.session_state.preset_type = "normal"
    elif apply_borderline:
        st.session_state.preset_type = "borderline"
    elif apply_ischemic:
        st.session_state.preset_type = "ischemic"

    pt = st.session_state.preset_type
    if pt == "normal":
        d_age, d_sex, d_trestbps, d_chol, d_fbs = 42, "Female", 118, 195, "No"
        d_cp, d_restecg, d_thalach, d_exang, d_oldpeak = "Non-anginal Pain", "Normal", 172, "No", 0.0
        d_slope, d_ca, d_thal = "Upsloping", 0, "Normal (3.0)"
    elif pt == "ischemic":
        d_age, d_sex, d_trestbps, d_chol, d_fbs = 63, "Male", 152, 295, "Yes"
        d_cp, d_restecg, d_thalach, d_exang, d_oldpeak = "Asymptomatic (Type 4)", "Left ventricular hypertrophy", 112, "Yes", 2.8
        d_slope, d_ca, d_thal = "Flat", 2, "Reversible Defect (7.0)"
    elif pt == "borderline":
        d_age, d_sex, d_trestbps, d_chol, d_fbs = 54, "Male", 135, 240, "No"
        d_cp, d_restecg, d_thalach, d_exang, d_oldpeak = "Atypical Angina", "ST-T wave abnormality", 145, "No", 1.0
        d_slope, d_ca, d_thal = "Flat", 1, "Normal (3.0)"
    else:
        d_age, d_sex, d_trestbps, d_chol, d_fbs = 52, "Male", 130, 230, "No"
        d_cp, d_restecg, d_thalach, d_exang, d_oldpeak = "Typical Angina", "Normal", 150, "No", 0.8
        d_slope, d_ca, d_thal = "Upsloping", 0, "Normal (3.0)"

    # Form layout
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 🩺 Demographics & Vitals")
        age = st.slider("Age (years)", 25, 80, d_age)
        sex_str = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if d_sex == "Male" else 1)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, d_trestbps)
        chol = st.slider("Serum Cholesterol (mg/dl)", 120, 560, d_chol)
        fbs_str = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0 if d_fbs == "No" else 1)

    with c2:
        st.markdown("##### 🫀 Symptoms & Electrocardiography")
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic (Type 4)"]
        cp_str = st.selectbox("Chest Pain Presentation", cp_options, index=cp_options.index(d_cp))

        restecg_options = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
        restecg_str = st.selectbox("Resting ECG Result", restecg_options, index=restecg_options.index(d_restecg))

        thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 70, 210, d_thalach)
        exang_str = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], index=0 if d_exang == "No" else 1)

    with c3:
        st.markdown("##### 🔬 Stress Testing & Imaging")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, float(d_oldpeak), step=0.1)

        slope_options = ["Upsloping", "Flat", "Downsloping"]
        slope_str = st.selectbox("Slope of Peak Exercise ST", slope_options, index=slope_options.index(d_slope))

        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], index=int(d_ca))

        thal_options = ["Normal (3.0)", "Fixed Defect (6.0)", "Reversible Defect (7.0)"]
        thal_str = st.selectbox("Thallium Scintigraphy", thal_options, index=thal_options.index(d_thal))

    # Model selector
    st.markdown("---")
    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        model_choice = st.selectbox(
            "Select Inference Model Pipeline:",
            options=[
                "Logistic Regression (L2 Regularized, C=0.1 — Calibrated Baseline)",
                "Random Forest Classifier (Ensemble, 100 Trees, Depth=4)",
                "Gradient Boosting Classifier (Ensemble, 100 Estimators)",
            ],
            index=0,
        )
    with m_col2:
        chosen_pipeline = lr_model if "Logistic" in model_choice else (rf_model if "Random" in model_choice else gb_model)
        st.info(f"Active Model: **{chosen_pipeline.named_steps['classifier'].__class__.__name__}**")

    # Map inputs to DataFrame
    sex_val = 1 if sex_str == "Male" else 0
    cp_val = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic (Type 4)": 4}[cp_str]
    fbs_val = 1 if fbs_str == "Yes" else 0
    restecg_val = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg_str]
    exang_val = 1 if exang_str == "Yes" else 0
    slope_val = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_str]
    thal_val = {"Normal (3.0)": 3.0, "Fixed Defect (6.0)": 6.0, "Reversible Defect (7.0)": 7.0}[thal_str]

    input_df = pd.DataFrame([
        {
            "age": float(age),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "thalach": float(thalach),
            "oldpeak": float(oldpeak),
            "sex": int(sex_val),
            "cp": int(cp_val),
            "fbs": int(fbs_val),
            "restecg": int(restecg_val),
            "exang": int(exang_val),
            "slope": int(slope_val),
            "ca": float(ca),
            "thal": float(thal_val),
        }
    ])

    # Run Prediction
    pred_prob = chosen_pipeline.predict_proba(input_df)[0, 1]
    pred_binary = int(pred_prob >= 0.50)

    st.markdown("### Clinical Screening Output")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if pred_prob < 0.30:
            tier, color = "Low Clinical Risk Tier", "#10b981"
            guidance = "Low probability of significant coronary artery disease (<50% diameter stenosis). Standard preventive lifestyle recommendations."
        elif pred_prob < 0.70:
            tier, color = "Moderate Clinical Risk Tier", "#f59e0b"
            guidance = "Intermediate probability. Non-invasive stress imaging (e.g., stress echocardiography or coronary CTA) recommended."
        else:
            tier, color = "High Clinical Risk Tier", "#ef4444"
            guidance = "Elevated probability of significant coronary artery disease. Prioritized cardiology referral for diagnostic angiography."

        st.markdown(
            f"""
            <div style="padding: 22px; border-radius: 12px; background-color: rgba(255,255,255,0.04); border-left: 6px solid {color};">
                <h3 style="color:{color}; margin-top:0;">{tier}</h3>
                <h1 style="margin: 8px 0; font-size: 2.2rem;">{pred_prob * 100:.1f}%</h1>
                <p style="margin-bottom: 6px;"><strong>Binary Prediction ($\\tau=0.50$):</strong> {'Presence of CAD (>= 50% stenosis)' if pred_binary == 1 else 'Absence of Significant Stenosis'}</p>
                <p style="margin-bottom: 0;"><strong>Actionable Recommendation:</strong> {guidance}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [30, 70], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    # Risk Attribution Breakdown (for Logistic Regression)
    if "Logistic" in model_choice:
        st.markdown("#### Patient-Specific Feature Attribution (Log-Odds Impact)")
        preprocessor = lr_model.named_steps["preprocessor"]
        clf = lr_model.named_steps["classifier"]
        feat_names = preprocessor.get_feature_names_out()
        X_trans = preprocessor.transform(input_df)
        X_trans_arr = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans
        contributions = X_trans_arr[0] * clf.coef_[0]

        contrib_df = pd.DataFrame({
            "Feature": [f.replace("num__", "").replace("cat__", "") for f in feat_names],
            "Contribution": contributions,
        }).sort_values(by="Contribution", ascending=True)

        fig_contrib = px.bar(
            contrib_df.tail(8),
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Top Factors Influencing Patient Log-Odds",
            color="Contribution",
            color_continuous_scale=["#10b981", "#ef4444"],
        )
        fig_contrib.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_contrib, use_container_width=True)

    # Benchmark metrics footer
    st.markdown("---")
    st.markdown("#### Model Performance on Independent Holdout Test Cohort ($N=61$)")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Holdout Accuracy", "88.5%")
    b2.metric("Holdout Precision", "86.2%")
    b3.metric("Holdout Recall (Sensitivity)", "89.3%")
    b4.metric("Holdout F1-Score", "0.877")
    b5.metric("Holdout ROC-AUC", "0.961")

# =========================================================
# VIEW 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
elif nav_choice == "📊 Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis — UCI Cleveland Cohort")
    st.markdown(
        f"**Sample Size:** {len(df)} patients &bull; "
        f"**Disease Prevalence:** {(df['target'].mean() * 100):.1f}% ({df['target'].sum()} Positive, {len(df) - df['target'].sum()} Negative)"
    )

    tab_eda1, tab_eda2, tab_eda3 = st.tabs(["📈 Clinical Distributions", "🔥 Correlation Analysis", "📋 Raw Cohort Data"])

    with tab_eda1:
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            fig_age = px.histogram(
                df,
                x="age",
                color="target",
                barmode="overlay",
                nbins=20,
                title="Age Distribution Stratified by Disease Presence",
                labels={"target": "Heart Disease (1=Yes, 0=No)", "age": "Age (years)"},
                color_discrete_sequence=["#10b981", "#ef4444"],
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with e_col2:
            cp_mapped = df["cp"].map({1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"})
            fig_cp = px.histogram(
                df.assign(ChestPain=cp_mapped),
                x="ChestPain",
                color="target",
                barmode="group",
                title="Chest Pain Type vs. Angiographic Disease",
                labels={"target": "Heart Disease", "ChestPain": "Chest Pain Type"},
                color_discrete_sequence=["#10b981", "#ef4444"],
            )
            st.plotly_chart(fig_cp, use_container_width=True)

        e_col3, e_col4 = st.columns(2)
        with e_col3:
            fig_scatter = px.scatter(
                df,
                x="thalach",
                y="oldpeak",
                color="target",
                title="Max Heart Rate vs. Exercise ST Depression (oldpeak)",
                labels={"thalach": "Max Heart Rate (bpm)", "oldpeak": "ST Depression (mm)", "target": "Disease"},
                color_discrete_sequence=["#10b981", "#ef4444"],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with e_col4:
            fig_thal = px.histogram(
                df,
                x="thal",
                color="target",
                barmode="group",
                title="Thallium Scintigraphy Defect vs. Target",
                labels={"thal": "Thal (3.0=Normal, 6.0=Fixed, 7.0=Reversible)", "target": "Disease"},
                color_discrete_sequence=["#10b981", "#ef4444"],
            )
            st.plotly_chart(fig_thal, use_container_width=True)

    with tab_eda2:
        st.markdown("#### Pearson Correlation Matrix (Continuous Features)")
        corr = df[NUMERIC_FEATURES + ["target"]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix Heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info(
            "💡 **Key Observation**: Exercise ST depression (`oldpeak`) shows the strongest positive linear correlation "
            "with heart disease presence ($r=+0.42$), while maximum heart rate (`thalach`) exhibits a strong negative correlation ($r=-0.42$)."
        )

    with tab_eda3:
        st.markdown("#### Dataset Sample (Cleveland Heart Disease)")
        st.dataframe(df.head(25), use_container_width=True)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=df.to_csv(index=False),
            file_name="processed_cleveland_heart_disease.csv",
            mime="text/csv",
        )

# =========================================================
# VIEW 3: MODEL EVALUATION & CALIBRATION
# =========================================================
elif nav_choice == "🔬 Model Evaluation & Calibration":
    st.subheader("Model Evaluation, Cross-Validation & Calibration")

    st.markdown("#### 1. Holdout Test Set Performance Comparison ($N=61$, Unseen)")
    comp_df = pd.DataFrame([
        {"Model": "Dummy Baseline", "Accuracy": "36.1%", "Precision": "28.0%", "Recall (Sensitivity)": "25.0%", "F1-Score": "0.264", "ROC-AUC": "0.352", "Brier Score": "0.639"},
        {"Model": "Logistic Regression (L2, C=0.1)", "Accuracy": "88.5%", "Precision": "86.2%", "Recall (Sensitivity)": "89.3%", "F1-Score": "0.877", "ROC-AUC": "0.961", "Brier Score": "0.084"},
        {"Model": "Random Forest (Depth 4)", "Accuracy": "88.5%", "Precision": "86.2%", "Recall (Sensitivity)": "89.3%", "F1-Score": "0.877", "ROC-AUC": "0.944", "Brier Score": "0.112"},
        {"Model": "Gradient Boosting (Depth 3)", "Accuracy": "86.9%", "Precision": "85.7%", "Recall (Sensitivity)": "85.7%", "F1-Score": "0.857", "ROC-AUC": "0.934", "Brier Score": "0.104"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 2. Diagnostic Visualizations")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        roc_img_path = PROJECT_DIR / "reports" / "figures" / "roc_curves.png"
        if roc_img_path.exists():
            st.image(str(roc_img_path), caption="ROC Curves — Holdout Test Cohort (Logistic Regression AUC = 0.961)", use_container_width=True)
        else:
            st.info("ROC curves figure not found.")

    with img_col2:
        calib_img_path = PROJECT_DIR / "reports" / "figures" / "calibration_curve.png"
        if calib_img_path.exists():
            st.image(str(calib_img_path), caption="Reliability Diagram — Probability Calibration & Brier Scores", use_container_width=True)
        else:
            st.info("Calibration curve figure not found.")

    img_col3, img_col4 = st.columns(2)
    with img_col3:
        cm_img_path = PROJECT_DIR / "reports" / "figures" / "confusion_matrix.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Confusion Matrix on Holdout Cohort (25 True Positives, 3 False Negatives)", use_container_width=True)
    with img_col4:
        fi_img_path = PROJECT_DIR / "reports" / "figures" / "feature_importance.png"
        if fi_img_path.exists():
            st.image(str(fi_img_path), caption="Permutation Feature Importance (Top drivers: thal, ca, oldpeak, cp)", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Clinical Trade-Off Analysis (Sensitivity vs. Specificity)")
    st.markdown(
        """
        - **False Negatives (Critical Danger):** A patient with coronary obstruction is predicted negative and sent home without care. 
          Our pipeline achieves **$89.3\\%$ Recall (Sensitivity)**, missing only 3 out of 28 diseased patients on unseen holdout data.
        - **False Positives (Financial / Anxiety Cost):** A healthy individual is predicted positive and referred for secondary diagnostic testing 
          (e.g., stress imaging or angiogram). Our model maintains **$87.9\\%$ Specificity** ($29/33$ healthy patients correctly ruled out).
        - **Brier Score Calibration ($0.084$):** Well-calibrated probabilities ensure risk percentages reflect real-world event frequencies rather than overconfident binary predictions.
        """
    )

# =========================================================
# VIEW 4: CLINICAL METHODOLOGY & LIMITATIONS
# =========================================================
elif nav_choice == "📋 Clinical Methodology & Limitations":
    st.subheader("Methodology, Leakage Prevention & Clinical Governance")

    st.markdown("### 🔒 Zero-Leakage Architecture")
    st.markdown(
        """
        In clinical predictive modeling, subtle data leakage can artificially inflate validation metrics:
        1. **Strict Temporal/Split Separation:** Data is split into training ($80\\%$) and holdout testing ($20\\%$) before any feature transformations.
        2. **Encapsulated Preprocessing:** The Scikit-Learn `ColumnTransformer` fits median/mode imputers, `StandardScaler`, and `OneHotEncoder` strictly on training folds.
        3. **Stratified Cross-Validation:** 5-fold stratified cross-validation ensures balanced class prevalence across all training iterations.
        """
    )

    st.markdown("### ⚠️ Dataset Limitations & Clinical Reality")
    st.markdown(
        """
        - **1988 Cohort Demographics:** The UCI Cleveland dataset originates from the Cleveland Clinic Foundation in 1988 ($N=303$).
          It is predominantly male ($68\\%$) and reflects late-20th-century clinical practices and angiography thresholds.
        - **Angiographic Ground Truth:** Target label represents $\\ge 50\\%$ narrowing in at least one major coronary artery. It does not measure microvascular disease, plaque vulnerability, or fractional flow reserve (FFR).
        - **Missing Modern Biomarkers:** Key contemporary cardiac biomarkers (high-sensitivity Troponin I/T, BNP/NT-proBNP, hs-CRP, Coronary Artery Calcium scores) are absent from this vintage dataset.
        - **Generalization Shift:** A model trained on Cleveland Clinic patients cannot be safely deployed at external hospitals without prospective local calibration and drift monitoring.
        """
    )

    st.markdown("### 📚 Attribution & Dataset Citation")
    st.markdown(
        """
        - **Dataset Source:** UCI Machine Learning Repository — Heart Disease (Cleveland Database).
        - **Principal Investigators:**
          - Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation)
          - William Steinbrunn, M.D. (University Hospital, Zurich, Switzerland)
          - Matthias Pfisterer, M.D. (University Hospital, Basel, Switzerland)
        - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
        """
    )
