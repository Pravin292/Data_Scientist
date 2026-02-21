# ==========================================
# XGBoost - Diabetes Progression Predictor
# ==========================================

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f8fb;
    }
    h1 {
        color: #0c3c78;
        text-align: center;
    }
    .stButton>button {
        background-color: #0c3c78;
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Progression Prediction")
st.write("Predict disease progression score using XGBoost Regression")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("xgboost_diabetes_model.pkl")

# -----------------------------
# Input Fields
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (normalized)", value=0.0)
    sex = st.number_input("Sex (normalized)", value=0.0)
    bmi = st.number_input("BMI (normalized)", value=0.0)
    bp = st.number_input("Blood Pressure (normalized)", value=0.0)
    s1 = st.number_input("Serum Measurement S1", value=0.0)

with col2:
    s2 = st.number_input("Serum Measurement S2", value=0.0)
    s3 = st.number_input("Serum Measurement S3", value=0.0)
    s4 = st.number_input("Serum Measurement S4", value=0.0)
    s5 = st.number_input("Serum Measurement S5", value=0.0)
    s6 = st.number_input("Serum Measurement S6", value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Progression"):

    input_data = np.array([[age, sex, bmi, bp, s1,
                            s2, s3, s4, s5, s6]])

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Progression Score:")
    st.success(f"{round(prediction, 2)}")

    # Simple interpretation
    if prediction < 100:
        st.info("🟢 Low Disease Progression")
    elif prediction < 200:
        st.warning("🟡 Moderate Disease Progression")
    else:
        st.error("🔴 High Disease Progression Risk")