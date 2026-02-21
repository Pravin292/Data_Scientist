import streamlit as st
import pandas as pd
import joblib

# ---------------- Load model ----------------
model = joblib.load("milk_quality_xgb_model.pkl")
features = joblib.load("features.pkl")

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Milk Quality Prediction",
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

st.markdown("<h1>🥛 Milk Intelligence Grading System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Extreme Gradient Boosting (XGBoost) Quality Assessor</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Input form ----------------
with st.form("milk_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧪 Dairy Chemical Properties", unsafe_allow_html=True)
    user_input = {}
    
    col1, col2 = st.columns(2)
    with col1:
        user_input["ph"] = st.number_input("pH Level", min_value=3.0, max_value=10.0, step=0.1)
        user_input["temperature"] = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, step=0.5)
        user_input["taste"] = st.selectbox("Taste", [0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
        
    with col2:
        user_input["odor"] = st.selectbox("Odor", [0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
        user_input["fat"] = st.selectbox("Fat Content", [0, 1], format_func=lambda x: "Optimal" if x == 1 else "Not Optimal")
        user_input["turbidity"] = st.selectbox("Turbidity", [0, 1], format_func=lambda x: "High" if x == 1 else "Low")
        user_input["colour"] = st.number_input("Colour Value", min_value=200, max_value=300, step=1)

    submitted = st.form_submit_button("Assess Milk Quality")
    input_df = pd.DataFrame([user_input])[features]
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Prediction ----------------
if submitted:
    prediction = model.predict(input_df)[0]

    if prediction == 2:
        st.markdown("""
            <div class="result-card-success">
                <div class="metric-text">🏆 Grade: High Quality</div>
                <p style="color: #cbd5e1; margin-top: 5px;">This dairy batch meets premium optimal standards.</p>
            </div>
        """, unsafe_allow_html=True)
    elif prediction == 1:
         st.markdown("""
            <div class="result-card-warning">
                <div class="metric-text">⚖️ Grade: Medium Quality</div>
                <p style="color: #cbd5e1; margin-top: 5px;">This dairy batch meets average acceptable thresholds.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card-danger">
                <div class="metric-text">⚠️ Grade: Low Quality</div>
                <p style="color: #cbd5e1; margin-top: 5px;">This dairy batch fails quality testing parameters. Not fit for premium distribution.</p>
            </div>
        """, unsafe_allow_html=True)