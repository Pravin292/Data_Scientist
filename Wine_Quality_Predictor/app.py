import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Intelligence Pro",
    page_icon="🍷",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #000000, #1a0008, #2a0012);
    color: #f2f2f2;
}

/* Title */
.main-title {
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #800020, #b11226, #ff003c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 40px;
}

/* Glass Panel */
.glass-panel {
    background: rgba(20, 0, 10, 0.75);
    border-radius: 20px;
    padding: 40px;
    box-shadow:
        0 10px 40px rgba(0,0,0,0.8),
        0 0 20px rgba(128, 0, 32, 0.4);
    border: 1px solid rgba(128,0,32,0.4);
}

/* Inputs */
.stNumberInput input {
    background-color: #0f0f0f;
    color: white;
    border: 1px solid #800020;
    border-radius: 10px;
}

.stNumberInput input:focus {
    border: 1px solid #ff003c;
    box-shadow: 0 0 8px #ff003c;
}

/* Button */
.stFormSubmitButton button {
    background: linear-gradient(90deg, #800020, #b11226);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 12px 30px;
    border: none;
    transition: 0.3s ease;
}

.stFormSubmitButton button:hover {
    background: linear-gradient(90deg, #ff003c, #800020);
    box-shadow: 0 0 15px rgba(255, 0, 60, 0.7);
}

/* Result */
.result-card {
    margin-top: 40px;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(145deg, #1a0008, #330014);
    box-shadow:
        0 10px 30px rgba(0,0,0,0.9),
        0 0 20px rgba(255,0,60,0.5);
    border: 1px solid #800020;
}

.grade {
    margin-top: 10px;
    font-size: 20px;
    color: #ff4d6d;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("wine_quality_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🍷 Wine Intelligence Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Maroon & Black Premium ML Dashboard</div>', unsafe_allow_html=True)

# ---------------- FORM ----------------
with st.form("prediction_form"):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", value=7.4, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", value=0.7, step=0.01)
        citric_acid = st.number_input("Citric Acid", value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", value=1.9, step=0.1)

    with col2:
        chlorides = st.number_input("Chlorides", value=0.076, step=0.001, format="%.3f")
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34)
        density = st.number_input("Density", value=0.9978, step=0.0001, format="%.4f")

    with col3:
        pH = st.number_input("pH", value=3.51, step=0.01)
        sulphates = st.number_input("Sulphates", value=0.56, step=0.01)
        alcohol = st.number_input("Alcohol", value=9.4, step=0.1)

    submitted = st.form_submit_button("Analyze Wine")

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
    elif quality >= 5:
        grade = "🍷 Vintage Standard"
    else:
        grade = "🍂 Entry Blend"

    st.markdown(f"""
        <div class="result-card">
            Predicted Quality Score: {quality}
            <div class="grade">{grade}</div>
        </div>
    """, unsafe_allow_html=True)