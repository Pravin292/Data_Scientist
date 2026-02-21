# ==========================================
# Decision Tree - Heart Disease Predictor
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
from utils import load_model_and_metrics, predict_heart_disease, generate_feature_importance_plot, get_risk_info

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
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700; }
.glass-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }
.stNumberInput input, .stSelectbox select, .stTextInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; }
.stNumberInput input:focus, .stSelectbox select:focus { border: 1px solid #38bdf8 !important; box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 8px; font-weight: 600; padding: 0.6em 1.5em; border: none; transition: all 0.3s ease; width: 100%; margin-top: 15px; }
.stButton>button:hover { background: linear-gradient(90deg, #60a5fa, #a78bfa); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); transform: translateY(-2px); }
.result-card-success { background: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-danger { background: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; padding: 20px; border-radius: 10px; margin-top: 20px; }
.metric-text { font-size: 24px; font-weight: bold; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD RESOURCES ----------------
model, metrics = load_model_and_metrics()

if model is None:
    st.error("Model artifacts missing. Execute `python model_training.py` within the local component directory first.")
    st.stop()

# ---------------- SIDEBAR: INPUT PARAMETERS ----------------
st.sidebar.markdown("### 🫀 Patient Physiology")
st.sidebar.caption("Provide baseline metrics for Cardiac Assessment.")

age = st.sidebar.number_input("Age", value=50, step=1)
sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
cp = st.sidebar.selectbox("Chest Pain Type (CP)", options=[1, 2, 3, 4])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", value=120, step=1)
chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", value=200, step=1)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x==1 else "False")
restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", value=150, step=1)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3])
ca = st.sidebar.number_input("Number of Major Vessels (0-3)", value=0, min_value=0, max_value=3, step=1)
thal = st.sidebar.selectbox("Thalassemia", options=[3, 6, 7])

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("Run Diagnostic Profiling")


# ---------------- HEADER ----------------
st.markdown("<h1>❤️ Cardiac Wellness Risk Assessor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced Decision Tree Physiological Modeling</p>", unsafe_allow_html=True)
st.markdown("---")


# ---------------- UI TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Predictive Analysis", "📊 Entropy & Modeling Metrics", "📁 Source Informatics"])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Inference Framework")
    
    if analyze_btn:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        
        prediction_class = predict_heart_disease(model, input_data)
        grade, style_class = get_risk_info(prediction_class)

        st.markdown(f"""
            <div class="{style_class}">
                <div class="metric-text">{grade} (Result Class: {prediction_class})</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Based on the provided physiological metrics, this analysis has resulted in the above diagnostic risk pathway.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Please enter metrics in the physiological sidebar and click **Run Diagnostic Profiling**.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Decision Tree Hyperparameters")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    col2.metric("Splitting Criterion", metrics["Criterion"])
    col3.metric("Maximum Depth", metrics["Max Depth"])
    
    st.markdown("---")
    fig = generate_feature_importance_plot(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📥 Heart Disease Analytics Set")
    try:
        df = pd.read_csv("data/heart.csv")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Excerpt: Top 100 biological records (UCI Machine Learning Repository).")
    except Exception as e:
        st.error(f"Failed to mount local filesystem records: {e}")
    st.markdown('</div>', unsafe_allow_html=True)