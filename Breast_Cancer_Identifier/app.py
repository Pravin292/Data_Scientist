# ==========================================
# Breast Cancer Predictor - Streamlit
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
from utils import load_model_and_metrics, generate_feature_importance_plot

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

# ----------------------------
# Load Modalities
# ----------------------------
model, features_list, metrics = load_model_and_metrics()

if model is None:
    st.error("Model artifacts missing. Execute `python model_training.py` within the local component directory first.")
    st.stop()


# ----------------------------
# Input Variables Sidebar
# ----------------------------
st.sidebar.markdown("### 🔬 Patient Biometric Data")
st.sidebar.caption("Provide cellular variables derived from standard biopsies.")

# Creating input vector for all features dynamically
inputs = []
for feature in features_list:
    display_name = feature.replace('_', ' ').title()
    val = st.sidebar.number_input(display_name, value=0.0)
    inputs.append(val)

analyze_btn = st.sidebar.button("Predict Diagnosis")

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1>🩺 Breast Cancer Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced Random Forest Classification Model for Malignancy Detection</p>", unsafe_allow_html=True)
st.markdown("---")


# ----------------------------
# UI Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Patient Diagnosis", "📊 Tree Hierarchy Metrics", "📁 Source Informatic Logs"])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Inference Framework")
    
    if analyze_btn:
        input_array = np.array([inputs])
        prediction = model.predict(input_array)
        
        # 0: Malignant, 1: Benign
        if prediction[0] == 1:
            st.markdown("""
                <div class="result-card-success">
                    <div class="metric-text">✅ Diagnosis: Benign (Non-Cancerous)</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">No active malignant structures naturally detected by the analytical model.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-card-danger">
                    <div class="metric-text">⚠️ Diagnosis: Malignant (Cancerous)</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">High probability of malignancy detected. Professional medical consultation advised immediately.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Please enter metrics in the physiological sidebar and click **Predict Diagnosis**.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Random Forest Model Profile")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    col2.metric("Precision Index", f"{metrics['Precision']:.3f}")
    col3.metric("Recall Rating", f"{metrics['Recall']:.3f}")
    
    st.markdown("---")
    fig = generate_feature_importance_plot(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📥 Underlying Biopsy Samples")
    try:
        df = pd.read_csv("data/breast_cancer.csv")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Excerpt: Top 100 records generated dynamically via scikit-learn dataset factory.")
    except Exception as e:
        st.error(f"Failed to mount local filesystem records: {e}")
    st.markdown('</div>', unsafe_allow_html=True)