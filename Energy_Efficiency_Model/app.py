# ==========================================
# Energy Efficiency - Heating Load Predictor
# Random Forest Regression
# ==========================================

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="🏢",
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

st.markdown("<h1>🏢 Building Energy Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Thermodynamic Heating Load Predictor using Random Forest Ensembles</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("energy_rf_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

with st.form("energy_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🏛 Architectural Features", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        relative_compactness = st.number_input("Relative Compactness", value=0.90)
        surface_area = st.number_input("Surface Area", value=600.0)
        wall_area = st.number_input("Wall Area", value=300.0)
        roof_area = st.number_input("Roof Area", value=200.0)

    with col2:
        overall_height = st.selectbox("Overall Height", [3.5, 7.0])
        orientation = st.selectbox("Orientation (2-5)", [2, 3, 4, 5])
        glazing_area = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])
        glazing_area_distribution = st.selectbox("Glazing Area Distribution (0-5)", [0,1,2,3,4,5])

    submitted = st.form_submit_button("Predict Heating Load")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = np.array([[relative_compactness,
                            surface_area,
                            wall_area,
                            roof_area,
                            overall_height,
                            orientation,
                            glazing_area,
                            glazing_area_distribution]])

    prediction = model.predict(input_data)[0]

    if prediction < 20:
        st.markdown(f"""
            <div class="result-card-success">
                <div class="metric-text">🏆 Highly Efficient: {round(prediction, 2)} kWh/m²</div>
                <p style="color: #cbd5e1; margin-top: 5px;">This structural layout requires very low heating demand to maintain thermal equilibrium.</p>
            </div>
        """, unsafe_allow_html=True)
    elif prediction < 35:
        st.markdown(f"""
            <div class="result-card-warning">
                <div class="metric-text">⚖️ Moderate Efficiency: {round(prediction, 2)} kWh/m²</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Standard heating regulation necessary. Modest structural improvements could reduce consumption.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card-danger">
                <div class="metric-text">⚠️ Low Efficiency: {round(prediction, 2)} kWh/m²</div>
                <p style="color: #cbd5e1; margin-top: 5px;">High heating demand detected. Severe structural modifications recommended to optimize thermal isolation.</p>
            </div>
        """, unsafe_allow_html=True)