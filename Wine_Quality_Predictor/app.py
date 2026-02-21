import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Intelligence Pro",
    page_icon="🍷",
    layout="wide"
)

# ---------------- CUSTOM CSS "Midnight Glass" ----------------
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

# ---------------- LOAD MODEL ----------------
model = joblib.load("wine_quality_model.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1>🍷 Wine Intelligence Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced Random Forest Quality Assessor Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- FORM ----------------
with st.form("prediction_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧪 Phsyiochemical Variables", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", value=7.4, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", value=0.7, step=0.01)
        citric_acid = st.number_input("Citric Acid", value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", value=1.9, step=0.1)

    with col2:
        chlorides = st.number_input("Chlorides", value=0.076, step=0.001, format="%.3f")
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
        density = st.number_input("Density", value=0.9978, step=0.0001, format="%.4f")

    with col3:
        pH = st.number_input("pH", value=3.51, step=0.01)
        sulphates = st.number_input("Sulphates", value=0.56, step=0.01)
        alcohol = st.number_input("Alcohol", value=9.4, step=0.1)

    submitted = st.form_submit_button("Analyze Wine Composition")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if submitted:
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH,
                            sulphates, alcohol]])

    prediction = model.predict(input_data)
    quality = round(prediction[0], 2)

    if quality >= 7:
        grade = "🏆 Elite Reserve"
        style_class = "result-card-success"
        color = "#10b981"
    elif quality >= 5:
        grade = "🍷 Vintage Standard"
        style_class = "result-card-warning"
        color = "#f59e0b"
    else:
        grade = "🍂 Entry Blend"
        style_class = "result-card-danger"
        color = "#ef4444"

    st.markdown(f"""
        <div class="{style_class}">
            <div class="metric-text">{grade} (Score: {quality})</div>
            <p style="color: #cbd5e1; margin-top: 5px;">Based on the physiochemical composition, this wine exhibits characteristics aligning with {grade.lower()} standards.</p>
        </div>
    """, unsafe_allow_html=True)