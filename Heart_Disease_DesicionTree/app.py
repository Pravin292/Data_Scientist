# ==========================================
# Decision Tree - Heart Disease Predictor
# ==========================================

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f8fbff;
    }
    h1 {
        color: #b30000;
        text-align: center;
    }
    .stButton>button {
        background-color: #b30000;
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction using Decision Tree")

st.write("Enter patient details below:")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("decision_tree_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])

with col2:
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thalassemia (1-3)", [1,2,3])

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heart Disease"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient likely has Heart Disease")
    else:
        st.success("✅ Low Risk: Patient likely does NOT have Heart Disease")