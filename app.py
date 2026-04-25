import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

SEVERITY_LABELS = {
    0: "No Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Very Severe",
}

st.title("Heart Disease Severity Predictor")
st.markdown("Enter patient details below to predict heart disease severity (UCI dataset).")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4],
                      format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                              3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality",
                                                   2: "Left Ventricular Hypertrophy"}[x])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
                               value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3],
                         format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7],
                        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

if st.button("Predict Severity", type="primary"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]

    label = SEVERITY_LABELS[prediction]
    confidence = proba[prediction] * 100

    if prediction == 0:
        st.success(f"Prediction: **{prediction} — {label}**  ({confidence:.1f}% confidence)")
    elif prediction <= 2:
        st.warning(f"Prediction: **{prediction} — {label}**  ({confidence:.1f}% confidence)")
    else:
        st.error(f"Prediction: **{prediction} — {label}**  ({confidence:.1f}% confidence)")

    st.subheader("Class Probabilities")
    for cls, prob in enumerate(proba):
        st.progress(float(prob), text=f"Level {cls} ({SEVERITY_LABELS[cls]}): {prob*100:.1f}%")
