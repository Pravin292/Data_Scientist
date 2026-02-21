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

st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom belongs to a specific class using a trained Gradient Boosting model.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ---------------- User Input ----------------
st.subheader("🔢 Enter Feature Values")

input_data = {}

for feature in feature_columns:
    input_data[feature] = st.number_input(
        f"{feature}",
        value=0.0
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]

    st.success(f"✅ Predicted Class: **{prediction}**")