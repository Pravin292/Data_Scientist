# ==========================================
# KNN Breast Cancer Prediction - Streamlit
# ==========================================

import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f7f9;
    }
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #c0392b;
        text-align: center;
    }
    .stButton>button {
        background-color: #c0392b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Classification using KNN")

st.write("Enter patient diagnostic values below:")

# ----------------------------
# Load Model & Scaler
# ----------------------------
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Input Fields (30 Features)
# ----------------------------

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

inputs = []

col1, col2 = st.columns(2)

for i in range(15):
    value = col1.number_input(feature_names[i], value=0.0)
    inputs.append(value)

for i in range(15, 30):
    value = col2.number_input(feature_names[i], value=0.0)
    inputs.append(value)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Diagnosis"):

    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.success("✅ Prediction: Benign (Non-Cancerous)")
    else:
        st.error("⚠️ Prediction: Malignant (Cancerous)")