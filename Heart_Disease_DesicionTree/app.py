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
# Custom Styling "Midnight Glass"
# -----------------------------
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
.stFormSubmitButton button {
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

.stFormSubmitButton button:hover {
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

st.markdown("<h1>❤️ Heart Disease Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced Decision Tree Cardiac Assessor</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("decision_tree_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

with st.form("cardiac_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🩺 Patient Cardiology Data", unsafe_allow_html=True)
    
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
        
    submitted = st.form_submit_button("Predict Cardiac Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown("""
            <div class="result-card-danger">
                <div class="metric-text">⚠️ High Risk Identified</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Patient indicates a high statistical probability of Heart Disease. Immediate medical observation is encouraged.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card-success">
                <div class="metric-text">✅ Low Risk Identified</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Patient falls into the lower probability bracket for significant Heart Disease.</p>
            </div>
        """, unsafe_allow_html=True)