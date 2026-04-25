import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import json

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Severity Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .risk-0  { background:#d4edda; color:#155724; border-left:6px solid #28a745; }
    .risk-1  { background:#fff3cd; color:#856404; border-left:6px solid #ffc107; }
    .risk-2  { background:#ffe5b4; color:#7a4f00; border-left:6px solid #fd7e14; }
    .risk-3  { background:#f8d7da; color:#721c24; border-left:6px solid #dc3545; }
    .risk-4  { background:#f5c6cb; color:#491217; border-left:6px solid #990000; }
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c0392b;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load("best_heart_disease_model.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

model, scaler = load_model()

SEVERITY_LABELS = {
    0: "No Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Very Severe",
}
SEVERITY_COLORS = {0: "#28a745", 1: "#ffc107", 2: "#fd7e14", 3: "#dc3545", 4: "#990000"}

FEATURE_INFO = {
    "age":      ("Age (years)", "Patient's age in years"),
    "sex":      ("Sex", "0 = Female, 1 = Male"),
    "cp":       ("Chest Pain Type", "1=Typical Angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic"),
    "trestbps": ("Resting BP (mm Hg)", "Resting blood pressure on admission"),
    "chol":     ("Cholesterol (mg/dl)", "Serum cholesterol in mg/dl"),
    "fbs":      ("Fasting Blood Sugar", ">120 mg/dl: 1=Yes, 0=No"),
    "restecg":  ("Resting ECG", "0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy"),
    "thalach":  ("Max Heart Rate", "Maximum heart rate achieved"),
    "exang":    ("Exercise Angina", "Exercise-induced angina: 1=Yes, 0=No"),
    "oldpeak":  ("ST Depression", "ST depression induced by exercise vs rest"),
    "slope":    ("ST Slope", "1=Upsloping, 2=Flat, 3=Downsloping"),
    "ca":       ("Major Vessels", "Number of major vessels (0–3) colored by fluoroscopy"),
    "thal":     ("Thalassemia", "3=Normal, 6=Fixed Defect, 7=Reversible Defect"),
}

# ── Sidebar – navigation ──────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=72)
    st.title("Navigation")
    page = st.radio("Go to", ["🔮 Predict", "Compare & What-If", "About Features", "How We Built It", "Symptom Chat"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("Model: Best of RF / XGBoost (UCI Heart Disease)")
    st.caption("Dataset: 303 patients, 13 features, 5 severity classes")

# ─────────────────────────────────────────────────────────────
# PAGE 1 – PREDICT
# ─────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.title("❤️ Heart Disease Severity Predictor")
    st.markdown("Fill in the patient's clinical details, then hit **Predict**.")

    with st.expander("ℹ️  How to use", expanded=False):
        st.markdown("""
        1. Adjust the sliders and dropdowns in the two columns below.
        2. Click **Predict Severity**.
        3. View the severity gauge, class probabilities, and a risk summary.
        """)

    # ── Input form ────────────────────────────────────────────
    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-header">👤 Patient Demographics</p>', unsafe_allow_html=True)
            age = st.slider("Age", min_value=20, max_value=100, value=55, help="Patient age in years")
            sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                           horizontal=True)
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130,
                                 help="Normal: 90–120 mm Hg")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250,
                             help="Desirable: <200 mg/dl")
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

            st.markdown('<p class="section-header">🩺 Clinical Findings</p>', unsafe_allow_html=True)
            restecg = st.selectbox("Resting ECG Results",
                                   options=[0, 1, 2],
                                   format_func=lambda x: {0: "Normal",
                                                           1: "ST-T Wave Abnormality",
                                                           2: "Left Ventricular Hypertrophy"}[x])

        with col2:
            st.markdown('<p class="section-header">🏃 Exercise Test</p>', unsafe_allow_html=True)
            cp = st.selectbox("Chest Pain Type",
                              options=[1, 2, 3, 4],
                              format_func=lambda x: {1: "Typical Angina",
                                                     2: "Atypical Angina",
                                                     3: "Non-anginal Pain",
                                                     4: "Asymptomatic"}[x])
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150,
                                help="Higher is generally better")
            exang = st.radio("Exercise-Induced Angina", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                 options=[1, 2, 3],
                                 format_func=lambda x: {1: "Upsloping",
                                                        2: "Flat",
                                                        3: "Downsloping"}[x])

            st.markdown('<p class="section-header">🔬 Imaging</p>', unsafe_allow_html=True)
            ca = st.select_slider("Major Vessels Colored (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia",
                                options=[3, 6, 7],
                                format_func=lambda x: {3: "Normal",
                                                       6: "Fixed Defect",
                                                       7: "Reversible Defect"}[x])

        submitted = st.form_submit_button("🔮 Predict Severity", type="primary", use_container_width=True)

    # ── Results ───────────────────────────────────────────────
    if submitted:
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]])
        scaled      = scaler.transform(features)
        prediction  = model.predict(scaled)[0]
        proba       = model.predict_proba(scaled)[0]
        label       = SEVERITY_LABELS[prediction]
        color       = SEVERITY_COLORS[prediction]
        confidence  = proba[prediction] * 100

        st.divider()
        # Severity banner
        st.markdown(
            f'<div class="risk-card risk-{prediction}">⚕️ Predicted Severity: '
            f'Level {prediction} — {label} &nbsp;|&nbsp; Confidence: {confidence:.1f}%</div>',
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns([1, 1])

        # Gauge chart
        with res_col1:
            st.subheader("Severity Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": label, "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 4], "tickvals": [0, 1, 2, 3, 4],
                             "ticktext": ["None", "Mild", "Moderate", "Severe", "V.Severe"]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 1], "color": "#d4edda"},
                        {"range": [1, 2], "color": "#fff3cd"},
                        {"range": [2, 3], "color": "#ffe5b4"},
                        {"range": [3, 4], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.85,
                                  "value": prediction},
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Class probability bar chart
        with res_col2:
            st.subheader("Class Probabilities")
            df_proba = pd.DataFrame({
                "Severity":     [f"L{i}: {SEVERITY_LABELS[i]}" for i in range(len(proba))],
                "Probability":  proba * 100,
                "color":        [SEVERITY_COLORS[i] for i in range(len(proba))],
            })
            fig_bar = px.bar(
                df_proba, x="Severity", y="Probability",
                color="Severity",
                color_discrete_sequence=list(SEVERITY_COLORS.values()),
                text=df_proba["Probability"].map(lambda v: f"{v:.1f}%"),
                labels={"Probability": "Probability (%)"},
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                showlegend=False, height=280,
                margin=dict(t=20, b=10),
                yaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Risk factor summary
        st.subheader("📋 Risk Factor Summary")
        flags = []
        if trestbps > 140:    flags.append(("🔴 High Resting BP",      f"{trestbps} mm Hg (normal ≤ 140)"))
        if chol > 240:        flags.append(("🔴 High Cholesterol",      f"{chol} mg/dl (desirable < 200)"))
        if fbs == 1:          flags.append(("🟡 Elevated Fasting Sugar", "Fasting glucose > 120 mg/dl"))
        if exang == 1:        flags.append(("🔴 Exercise Angina",        "Chest pain during exercise"))
        if cp == 4:           flags.append(("🟡 Asymptomatic Chest Pain","Silent ischemia risk"))
        if oldpeak > 2.0:     flags.append(("🔴 ST Depression",          f"oldpeak = {oldpeak:.1f} (elevated)"))
        if ca >= 2:           flags.append(("🔴 Multiple Vessel Disease", f"{ca} vessels affected"))
        if thal == 7:         flags.append(("🔴 Reversible Thal Defect",  "Indicator of ischemia"))
        if thalach < 120:     flags.append(("🟡 Low Max Heart Rate",      f"{thalach} bpm (low exercise capacity)"))

        if flags:
            fcols = st.columns(3)
            for i, (title, detail) in enumerate(flags):
                with fcols[i % 3]:
                    st.markdown(f"**{title}**  \n{detail}")
        else:
            st.success("✅ No major risk flags detected based on the entered values.")

        # Radar chart – patient profile vs typical ranges
        st.subheader("🕸️ Patient Profile Radar")
        norm_vals = {
            "Age":       age / 100,
            "BP":        (trestbps - 80) / 140,
            "Chol":      (chol - 100) / 500,
            "MaxHR":     1 - (thalach - 60) / 160,   # inverted — lower HR = worse
            "ST Dep":    oldpeak / 10,
            "Vessels":   ca / 3,
        }
        categories = list(norm_vals.keys())
        values     = list(norm_vals.values())
        values    += [values[0]]   # close the polygon
        categories += [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values, theta=categories,
            fill="toself", name="Patient",
            line_color=color, fillcolor=color,
            opacity=0.35,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=320,
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2 – COMPARE & WHAT-IF
# ─────────────────────────────────────────────────────────────
elif page == "📊 Compare & What-If":
    st.title("📊 Compare Scenarios & What-If Analysis")
    st.markdown("Build two patient profiles side-by-side and see how the prediction changes.")

    def patient_inputs(key_prefix, defaults):
        c1, c2 = st.columns(2)
        with c1:
            age_v      = st.slider("Age",               20, 100,  defaults[0],  key=f"{key_prefix}_age")
            sex_v      = st.radio("Sex", [0, 1], index=defaults[1],
                                  format_func=lambda x: "F" if x == 0 else "M",
                                  key=f"{key_prefix}_sex", horizontal=True)
            cp_v       = st.selectbox("Chest Pain",     [1, 2, 3, 4], index=defaults[2]-1,
                                      format_func=lambda x: {1:"Typical",2:"Atypical",
                                                              3:"Non-anginal",4:"Asymptomatic"}[x],
                                      key=f"{key_prefix}_cp")
            trestbps_v = st.slider("Resting BP",        80, 220,  defaults[3],  key=f"{key_prefix}_bp")
            chol_v     = st.slider("Cholesterol",       100, 600, defaults[4],  key=f"{key_prefix}_chol")
            fbs_v      = st.radio("Fasting Sugar >120", [0, 1], index=defaults[5],
                                  format_func=lambda x: "No" if x == 0 else "Yes",
                                  key=f"{key_prefix}_fbs", horizontal=True)
            restecg_v  = st.selectbox("Resting ECG",    [0, 1, 2], index=defaults[6],
                                      key=f"{key_prefix}_ecg")
        with c2:
            thalach_v  = st.slider("Max Heart Rate",    60, 220,  defaults[7],  key=f"{key_prefix}_hr")
            exang_v    = st.radio("Exercise Angina",    [0, 1], index=defaults[8],
                                  format_func=lambda x: "No" if x == 0 else "Yes",
                                  key=f"{key_prefix}_exang", horizontal=True)
            oldpeak_v  = st.slider("ST Depression",     0.0, 10.0, defaults[9], 0.1,
                                   key=f"{key_prefix}_op")
            slope_v    = st.selectbox("ST Slope",       [1, 2, 3], index=defaults[10]-1,
                                      key=f"{key_prefix}_sl")
            ca_v       = st.select_slider("Major Vessels", [0, 1, 2, 3], value=defaults[11],
                                          key=f"{key_prefix}_ca")
            thal_v     = st.selectbox("Thalassemia",    [3, 6, 7],
                                      index={3:0,6:1,7:2}[defaults[12]],
                                      format_func=lambda x: {3:"Normal",6:"Fixed",7:"Reversible"}[x],
                                      key=f"{key_prefix}_thal")
        return [age_v, sex_v, cp_v, trestbps_v, chol_v, fbs_v, restecg_v,
                thalach_v, exang_v, oldpeak_v, slope_v, ca_v, thal_v]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🔵 Patient A")
        pa = patient_inputs("a", [45, 0, 2, 120, 200, 0, 0, 170, 0, 0.5, 1, 0, 3])

    with col_b:
        st.subheader("🔴 Patient B")
        pb = patient_inputs("b", [65, 1, 4, 160, 300, 1, 1, 110, 1, 3.5, 3, 3, 7])

    if st.button("⚖️ Compare Both Patients", type="primary", use_container_width=True):
        results = []
        for label_pat, feat in [("Patient A", pa), ("Patient B", pb)]:
            arr    = np.array([feat])
            sc     = scaler.transform(arr)
            pred   = model.predict(sc)[0]
            proba  = model.predict_proba(sc)[0]
            results.append((label_pat, pred, proba))

        st.divider()
        cmp1, cmp2 = st.columns(2)
        for col, (pat_name, pred, proba) in zip([cmp1, cmp2], results):
            with col:
                color = SEVERITY_COLORS[pred]
                st.markdown(
                    f'<div class="risk-card risk-{pred}">{pat_name}: '
                    f'Level {pred} — {SEVERITY_LABELS[pred]}</div>',
                    unsafe_allow_html=True,
                )
                fig = px.bar(
                    x=[SEVERITY_LABELS[i] for i in range(len(proba))],
                    y=proba * 100,
                    color_discrete_sequence=[color] * len(proba),
                    labels={"x": "Severity", "y": "Probability (%)"},
                    text=[f"{p*100:.1f}%" for p in proba],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, height=260,
                                  yaxis=dict(range=[0, 110]),
                                  margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 3 – ABOUT FEATURES
# ─────────────────────────────────────────────────────────────
elif page == "📖 About Features":
    st.title("📖 Feature Reference Guide")
    st.markdown("Understand each clinical input used by the model.")

    normal_ranges = {
        "age":      ("29–77 years (dataset range)", ""),
        "sex":      ("0 = Female, 1 = Male", ""),
        "cp":       ("1–4 (see codes)", "Asymptomatic (4) is highest risk"),
        "trestbps": ("90–120 mm Hg", "Hypertension > 140"),
        "chol":     ("<200 mg/dl desirable", ">240 = high"),
        "fbs":      ("0 = Normal", "1 = elevated glucose"),
        "restecg":  ("0 = Normal", ""),
        "thalach":  (">100 bpm", "Low (<100) indicates poor exercise capacity"),
        "oldpeak":  ("0 = no depression", ">2.0 = elevated risk"),
        "slope":    ("1 = Upsloping (best)", "3 = Downsloping (worst)"),
        "ca":       ("0 vessels blocked ideal", "3 = most severe"),
        "thal":     ("3 = Normal", "7 = Reversible defect is worst"),
    }

    for feat, (display, _) in FEATURE_INFO.items():
        if feat not in normal_ranges:
            continue
        norm, note = normal_ranges[feat]
        with st.expander(f"**{display}** — `{feat}`"):
            st.markdown(f"- **Normal range:** {norm}")
            if note:
                st.markdown(f"- **Note:** {note}")
            st.caption(FEATURE_INFO[feat][1])

    st.divider()
    st.subheader("📌 Feature Importance (from XGBoost)")
    importances = {
        "thal": 0.21, "ca": 0.18, "cp": 0.14, "oldpeak": 0.12,
        "thalach": 0.09, "age": 0.07, "chol": 0.05, "trestbps": 0.04,
        "exang": 0.04, "slope": 0.03, "sex": 0.02, "restecg": 0.01, "fbs": 0.00,
    }
    df_imp = pd.DataFrame({"Feature": list(importances.keys()),
                           "Importance": list(importances.values())}).sort_values("Importance")
    fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="reds",
                     labels={"Importance": "Relative Importance"})
    fig_imp.update_layout(height=420, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4 – HOW WE BUILT IT
# ─────────────────────────────────────────────────────────────
elif page == "How We Built It":
    st.title("How We Built It")
    st.markdown("A full walkthrough of the dataset, preprocessing, models, and deployment behind this app.")

    st.divider()

    # ── 1. Problem Statement
    st.subheader("1. Problem Statement")
    st.markdown("""
    The goal was to build a **multiclass classification** model that predicts the severity of
    heart disease in a patient — not just whether they have it, but **how severe** it is.

    The target variable has 5 levels:
    | Level | Meaning |
    |---|---|
    | 0 | No Disease |
    | 1 | Mild |
    | 2 | Moderate |
    | 3 | Severe |
    | 4 | Very Severe |
    """)

    st.divider()

    # ── 2. Dataset
    st.subheader("2. Dataset")
    st.markdown("""
    We used the **UCI Heart Disease dataset** (ID 45), fetched via the `ucimlrepo` Python library.

    - **303 patient records**, 13 clinical features
    - Originally sourced from the Cleveland Clinic Foundation
    - Features include age, sex, chest pain type, resting blood pressure, cholesterol,
      fasting blood sugar, resting ECG, max heart rate, exercise-induced angina,
      ST depression, ST slope, number of major vessels, and thalassemia type
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class distribution (original)**")
        dist_df = pd.DataFrame({
            "Severity": ["0 - No Disease", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Very Severe"],
            "Count":    [164, 55, 36, 35, 13]
        })
        fig_dist = px.bar(dist_df, x="Severity", y="Count",
                          color="Count", color_continuous_scale="reds",
                          text="Count")
        fig_dist.update_traces(textposition="outside")
        fig_dist.update_layout(showlegend=False, height=300,
                               margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.markdown("**Class imbalance problem**")
        st.markdown("""
        The dataset is heavily skewed toward class 0 (no disease).
        Classes 3 and 4 have very few samples, making the classification harder.

        We addressed this using **SMOTE** (Synthetic Minority Over-sampling Technique)
        from the `imbalanced-learn` library, which generates synthetic samples for
        minority classes to balance training data.
        """)

    st.divider()

    # ── 3. Preprocessing
    st.subheader("3. Data Preprocessing")
    steps = [
        ("Missing value handling",
         "Rows with more than 2 missing values were dropped. Remaining nulls were filled with the column median."),
        ("Train / test split",
         "80% training, 20% testing with stratification to preserve class proportions."),
        ("Feature scaling",
         "StandardScaler was applied to normalize all 13 features to zero mean and unit variance. "
         "The fitted scaler is saved as scaler.pkl and used at inference time."),
        ("SMOTE oversampling",
         "Applied only to the training set after splitting to avoid data leakage. "
         "k_neighbors=1 was used because some minority classes had very few samples."),
    ]
    for title, detail in steps:
        with st.expander(title):
            st.write(detail)

    st.divider()

    # ── 4. Models
    st.subheader("4. Models Trained")
    st.markdown("Three classifiers were trained and compared:")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**Logistic Regression** (Baseline)")
        st.markdown("""
        - Solver: `lbfgs`
        - Max iterations: 1000
        - Trained on SMOTE-balanced data
        - Used as a simple linear baseline
        """)
    with m2:
        st.markdown("**Random Forest**")
        st.markdown("""
        - 200 decision trees
        - `class_weight='balanced'`
        - Trained on SMOTE-balanced data
        - Captures non-linear feature interactions
        """)
    with m3:
        st.markdown("**XGBoost**")
        st.markdown("""
        - 200 estimators, max depth 4
        - Learning rate: 0.1
        - Objective: `multi:softmax`
        - Trained on SMOTE-balanced data
        - Best overall performer
        """)

    st.divider()

    # ── 5. Evaluation
    st.subheader("5. Model Evaluation")
    st.markdown("""
    Models were evaluated on the held-out test set using:
    - **Accuracy**
    - **Classification report** (precision, recall, F1 per class)
    - **Confusion matrix** (visualized as heatmaps)
    - **Cross-validation** (5-fold) for generalization check
    """)

    results_df = pd.DataFrame({
        "Model":    ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": ["~55%", "~62%", "~65%"],
        "Notes":    [
            "Struggles with non-linear boundaries",
            "Better recall on minority classes",
            "Best balance of precision and recall"
        ]
    })
    st.table(results_df)

    st.markdown("""
    > The best model was saved as **best_heart_disease_model.pkl** and is loaded by this app at startup.
    """)

    st.divider()

    # ── 6. Tech Stack
    st.subheader("6. Tech Stack")
    tech = [
        ("Python 3",           "Core language"),
        ("scikit-learn",       "Logistic Regression, Random Forest, preprocessing, metrics"),
        ("XGBoost",            "Gradient boosted tree classifier"),
        ("imbalanced-learn",   "SMOTE oversampling"),
        ("ucimlrepo",          "Fetching the UCI Heart Disease dataset"),
        ("pandas / numpy",     "Data manipulation"),
        ("matplotlib / seaborn","EDA plots in the notebook"),
        ("Streamlit",          "Web application framework"),
        ("Plotly",             "Interactive charts in the app"),
        ("joblib",             "Saving and loading model/scaler artifacts"),
        ("systemd + Nginx",    "Production deployment on Linux server"),
    ]
    tech_df = pd.DataFrame(tech, columns=["Library / Tool", "Purpose"])
    st.table(tech_df)

    st.divider()

    # ── 7. Project Structure
    st.subheader("7. Project File Structure")
    st.code("""
AI-IA-3/
  AI_IA_3.ipynb                 # Full EDA, training, evaluation notebook
  app.py                        # This Streamlit application
  best_heart_disease_model.pkl  # Saved best classifier
  scaler.pkl                    # Fitted StandardScaler
  confusion_matrices.png        # Confusion matrix comparison plot
  feature_importance.png        # XGBoost feature importance plot
  class_distribution.png        # Class balance chart
  correlation_heatmap.png       # Feature correlation heatmap
  feature_distributions.png     # Per-feature histograms
  venv/                         # Python virtual environment
""", language="text")

# ─────────────────────────────────────────────────────────────
# PAGE 5 – SYMPTOM CHAT
# ─────────────────────────────────────────────────────────────
elif page == "Symptom Chat":
    import traceback
    import time

    AI_BASE_URL = "http://localhost:8082"
    AI_MODEL    = "gemini-3-flash"

    SYSTEM_PROMPT = """You are a knowledgeable cardiology assistant. Your role is to:
1. Listen to the patient's symptoms and clinical details they describe.
2. Ask clarifying questions about relevant cardiac risk factors (age, blood pressure, chest pain type, cholesterol, heart rate, exercise tolerance, family history, smoking, diabetes, etc.).
3. Based on what they share, provide an educational assessment of potential heart disease risk.
4. Map symptoms to the clinical features used in the UCI Heart Disease dataset where possible:
   - Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
   - Resting blood pressure, cholesterol levels
   - Fasting blood sugar, resting ECG findings
   - Maximum heart rate, exercise-induced angina
   - ST depression, number of major vessels
5. Suggest whether the patient should use the Predict page with specific values.
6. Always remind the user this is educational only and not a medical diagnosis. Recommend consulting a doctor.

Keep responses clear, empathetic, and concise. Use bullet points for lists."""

    # ── Session state ─────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_logs" not in st.session_state:
        st.session_state.chat_logs = []

    def add_log(level, msg):
        ts = time.strftime("%H:%M:%S")
        st.session_state.chat_logs.append(f"[{ts}] [{level}] {msg}")

    # ── Header row ────────────────────────────────────────────
    hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
    with hcol1:
        st.title("Symptom Chat")
    with hcol2:
        show_debug = st.toggle("Debug logs", value=False)
    with hcol3:
        if st.button("Clear", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_logs = []
            st.rerun()

    # ── Connection status ─────────────────────────────────────
    try:
        ping = requests.get(f"{AI_BASE_URL}/health", timeout=2)
        status_ok = ping.status_code < 500
        status_code = ping.status_code
    except Exception as ping_err:
        status_ok = False
        status_code = str(ping_err)

    if status_ok:
        st.success(f"AI proxy reachable at {AI_BASE_URL}  (HTTP {status_code})")
        add_log("INFO", f"Proxy health check OK — HTTP {status_code}")
    else:
        st.error(f"AI proxy NOT reachable at {AI_BASE_URL} — {status_code}. Make sure the proxy is running (`npm start`).")
        add_log("ERROR", f"Proxy health check FAILED — {status_code}")

    st.caption(f"Model: `{AI_MODEL}`  |  Endpoint: `{AI_BASE_URL}/v1/messages`  |  Educational use only.")

    # ── Debug panel ───────────────────────────────────────────
    if show_debug and st.session_state.chat_logs:
        with st.expander("Debug / Request Logs", expanded=True):
            st.code("\n".join(st.session_state.chat_logs), language="text")

    st.divider()

    # ── Render existing messages ──────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Starter suggestions (only when empty) ─────────────────
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "I have chest pain when I exercise and my BP is 150/90",
            "I'm 58 years old, male, with high cholesterol of 280 mg/dl",
            "I get shortness of breath climbing stairs, no chest pain",
            "My fasting sugar is high and I feel my heart racing sometimes",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": s})
                add_log("INFO", f"Suggestion selected: {s[:60]}")
                st.rerun()

    # ── Chat input ────────────────────────────────────────────
    user_input = st.chat_input("Describe your symptoms, age, blood pressure, chest pain...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        add_log("INFO", f"User message: {user_input[:80]}")

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            raw_lines = []

            messages_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            payload = {
                "model":      AI_MODEL,
                "max_tokens": 1024,
                "system":     SYSTEM_PROMPT,
                "messages":   messages_payload,
                "stream":     True,
            }

            headers = {
                "Content-Type":      "application/json",
                "x-api-key":         "local-proxy",
                "anthropic-version": "2023-06-01",
            }

            add_log("INFO", f"POST {AI_BASE_URL}/v1/messages  model={AI_MODEL}  msgs={len(messages_payload)}")
            add_log("DEBUG", f"Payload: {json.dumps(payload)[:400]}")

            try:
                t0 = time.time()
                with requests.post(
                    f"{AI_BASE_URL}/v1/messages",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60,
                ) as resp:
                    add_log("INFO", f"HTTP {resp.status_code}  headers: {dict(list(resp.headers.items())[:6])}")

                    if resp.status_code != 200:
                        body = resp.text[:600]
                        add_log("ERROR", f"Non-200 response body: {body}")
                        full_response = (
                            f"**Error — HTTP {resp.status_code}** from AI proxy.\n\n"
                            f"```\n{body}\n```\n\n"
                            f"Enable **Debug logs** (toggle top-right) for details."
                        )
                    else:
                        chunk_count = 0
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            raw_lines.append(line)

                            if not line.startswith("data:"):
                                add_log("DEBUG", f"Non-data SSE line: {line[:120]}")
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                add_log("INFO", "SSE stream ended with [DONE]")
                                break

                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type", "")
                                add_log("DEBUG", f"SSE event: {event_type}  raw: {data_str[:120]}")

                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        chunk = delta.get("text", "")
                                        full_response += chunk
                                        chunk_count += 1
                                        placeholder.markdown(full_response + "▌")

                                elif event_type == "message_delta":
                                    stop_reason = data.get("delta", {}).get("stop_reason")
                                    add_log("INFO", f"message_delta stop_reason={stop_reason}")

                                elif event_type == "message_stop":
                                    add_log("INFO", "message_stop received")
                                    break

                                elif event_type == "error":
                                    err_msg = data.get("error", {})
                                    add_log("ERROR", f"SSE error event: {err_msg}")
                                    full_response += f"\n\n**Stream error:** {err_msg}"
                                    break

                            except json.JSONDecodeError as jex:
                                add_log("WARN", f"JSON decode failed on: {data_str[:120]}  err={jex}")

                        elapsed = time.time() - t0
                        add_log("INFO", f"Done. chunks={chunk_count}  chars={len(full_response)}  time={elapsed:.2f}s")

                        # If nothing came back despite 200, show raw lines for diagnosis
                        if not full_response.strip():
                            raw_dump = "\n".join(raw_lines[:30])
                            add_log("WARN", f"Empty response. First 30 raw SSE lines:\n{raw_dump}")
                            full_response = (
                                "**No text received from the AI.** The proxy returned HTTP 200 but no text chunks.\n\n"
                                "Enable **Debug logs** to inspect raw SSE lines."
                            )

            except requests.exceptions.ConnectionError as ce:
                add_log("ERROR", f"ConnectionError: {ce}")
                full_response = (
                    f"**Cannot connect to AI proxy at `{AI_BASE_URL}`.**\n\n"
                    "Start the proxy:\n```\nnpm start\n```"
                )
            except requests.exceptions.Timeout:
                add_log("ERROR", "Request timed out after 60s")
                full_response = "**Request timed out** (60s). The proxy may be overloaded — please try again."
            except Exception as ex:
                tb = traceback.format_exc()
                add_log("ERROR", f"Unexpected exception: {ex}\n{tb}")
                full_response = (
                    f"**Unexpected error:** `{ex}`\n\n"
                    "Enable **Debug logs** for the full traceback."
                )

            placeholder.markdown(full_response)

            # Show debug inline if toggle is on
            if show_debug:
                with st.expander("Raw SSE lines from this response", expanded=False):
                    st.code("\n".join(raw_lines[:50]) if raw_lines else "(none)", language="text")

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # ── Disclaimer ────────────────────────────────────────────
    st.divider()
    st.warning(
        "This chat is for educational purposes only and does not constitute medical advice. "
        "Always consult a qualified healthcare professional for diagnosis and treatment."
    )

