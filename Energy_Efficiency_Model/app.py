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
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #eef2f7;
    }
    h1 {
        color: #0a3d62;
        text-align: center;
    }
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏢 Building Energy Efficiency Prediction")
st.write("Predict Heating Load using Random Forest Regression")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("energy_rf_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

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

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heating Load"):

    input_data = np.array([[relative_compactness,
                            surface_area,
                            wall_area,
                            roof_area,
                            overall_height,
                            orientation,
                            glazing_area,
                            glazing_area_distribution]])

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Heating Load:")
    st.success(f"{round(prediction, 2)} kWh/m²")

    # Simple efficiency indicator
    if prediction < 20:
        st.info("🏆 High Energy Efficiency Building")
    elif prediction < 35:
        st.warning("⚖️ Moderate Energy Efficiency")
    else:
        st.error("⚠️ Low Energy Efficiency – High Heating Demand")