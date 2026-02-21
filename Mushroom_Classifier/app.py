import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="centered"
)

# ---------------- Custom Styling "Midnight Glass" ----------------
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

st.markdown("<h1>🍄 Fungi Classification Module</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Gradient Boosting Edibility Detector</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ---------------- User Input ----------------
input_data = {}

with st.form("mushroom_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔢 Biological Features", unsafe_allow_html=True)
    
    # Generate dynamic columns
    columns = st.columns(3)
    
    for i, feature in enumerate(feature_columns):
        col = columns[i % 3]
        input_data[feature] = col.number_input(f"{feature.replace('_', ' ').title()}", value=0.0)

    submitted = st.form_submit_button("Analyze Biological Traits")
    st.markdown('</div>', unsafe_allow_html=True)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# ---------------- Prediction ----------------
if submitted:
    prediction = model.predict(input_df)[0]

    if str(prediction).lower() in ["e", "edible", "0"]:
        st.markdown(f"""
            <div class="result-card-success">
                <div class="metric-text">✅ Classification: Edible ({prediction})</div>
                <p style="color: #cbd5e1; margin-top: 5px;">This fungus exhibits standard characteristics associated with non-toxic species.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card-danger">
                <div class="metric-text">☠️ Classification: Poisonous ({prediction})</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Highly toxic characteristics detected. Do not consume.</p>
            </div>
        """, unsafe_allow_html=True)