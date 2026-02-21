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

.result-card-warning {
    background: rgba(245, 158, 11, 0.15);
    border-left: 5px solid #f59e0b;
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

st.markdown("<h1>🩺 Diabetes Progression Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Quantitative Disease Evaluation via XGBoost Regression</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("xgboost_diabetes_model.pkl")

# -----------------------------
# Input Fields
# -----------------------------
with st.form("diabetes_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔬 Patient Baselines (Normalized Data)", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", value=0.0)
        sex = st.number_input("Sex", value=0.0)
        bmi = st.number_input("Body Mass Index (BMI)", value=0.0)
        bp = st.number_input("Blood Pressure", value=0.0)
        s1 = st.number_input("Serum Measurement S1", value=0.0)

    with col2:
        s2 = st.number_input("Serum Measurement S2", value=0.0)
        s3 = st.number_input("Serum Measurement S3", value=0.0)
        s4 = st.number_input("Serum Measurement S4", value=0.0)
        s5 = st.number_input("Serum Measurement S5", value=0.0)
        s6 = st.number_input("Serum Measurement S6", value=0.0)
        
    submitted = st.form_submit_button("Predict Progression Score")
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = np.array([[age, sex, bmi, bp, s1,
                            s2, s3, s4, s5, s6]])

    prediction = model.predict(input_data)[0]

    if prediction < 100:
        st.markdown(f"""
            <div class="result-card-success">
                <div class="metric-text">🟢 Low Progression Range: {round(prediction, 2)}</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Disease expansion potential is statistically subdued.</p>
            </div>
        """, unsafe_allow_html=True)
    elif prediction < 200:
         st.markdown(f"""
            <div class="result-card-warning">
                <div class="metric-text">🟡 Moderate Progression Range: {round(prediction, 2)}</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Significant disease activity recorded. Standard containment protocol advised.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card-danger">
                <div class="metric-text">🔴 High Progression Risk: {round(prediction, 2)}</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Aggressive longitudinal escalation expected. Intensive treatment strongly recommended.</p>
            </div>
        """, unsafe_allow_html=True)