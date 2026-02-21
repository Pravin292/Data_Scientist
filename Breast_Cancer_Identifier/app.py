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
# Custom Styling "Midnight Glass"
# ----------------------------
st.markdown("""
<style>
/* General Body */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Headers */
h1, h2, h3 {
    color: #38bdf8 !important;
    font-weight: 700;
}

/* Glassmorphism Containers */
.glass-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

/* Input boxes & Dropdowns */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
}

.stNumberInput input:focus, .stSelectbox select:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important;
}

/* Submit Buttons */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6em 1.5em;
    border: none;
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 15px;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
    transform: translateY(-2px);
}

/* Output Cards */
.result-card-success {
    background: rgba(16, 185, 129, 0.15);
    border-left: 5px solid #10b981;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

.result-card-danger {
    background: rgba(239, 68, 68, 0.15);
    border-left: 5px solid #ef4444;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

.metric-text {
    font-size: 24px;
    font-weight: bold;
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🩺 Breast Cancer Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced KNN Classification Model for Malignancy Detection</p>", unsafe_allow_html=True)
st.markdown("---")

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

# Using form to prevent instant continuous reloading
with st.form("diagnostics_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔬 Patient Biometric Data", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    for i in range(15):
        value = col1.number_input(feature_names[i].title(), value=0.0)
        inputs.append(value)
    
    for i in range(15, 30):
        value = col2.number_input(feature_names[i].title(), value=0.0)
        inputs.append(value)
        
    submitted = st.form_submit_button("Predict Diagnosis")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Prediction Block
# ----------------------------
if submitted:
    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.markdown("""
            <div class="result-card-success">
                <div class="metric-text">✅ Diagnosis: Benign (Non-Cancerous)</div>
                <p style="color: #cbd5e1; margin-top: 5px;">No immediate malignant indicators naturally detected by the KNN model.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card-danger">
                <div class="metric-text">⚠️ Diagnosis: Malignant (Cancerous)</div>
                <p style="color: #cbd5e1; margin-top: 5px;">High probability of malignancy detected. Professional medical consultation advised.</p>
            </div>
        """, unsafe_allow_html=True)