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

st.title("🥛 Milk Quality Prediction")
st.caption("XGBoost-based classification model")

st.divider()

# ---------------- Input form ----------------
st.subheader("Enter Milk Parameters")

user_input = {}

user_input["ph"] = st.number_input("pH Level", min_value=3.0, max_value=10.0, step=0.1)
user_input["temperature"] = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, step=0.5)

user_input["taste"] = st.selectbox("Taste", [0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
user_input["odor"] = st.selectbox("Odor", [0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
user_input["fat"] = st.selectbox("Fat Content", [0, 1], format_func=lambda x: "Optimal" if x == 1 else "Not Optimal")
user_input["turbidity"] = st.selectbox("Turbidity", [0, 1], format_func=lambda x: "High" if x == 1 else "Low")

user_input["colour"] = st.number_input("Colour Value", min_value=200, max_value=300, step=1)

input_df = pd.DataFrame([user_input])[features]

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict Milk Quality"):
    prediction = model.predict(input_df)[0]

    grade_map = {0: "Low", 1: "Medium", 2: "High"}

    st.success(f"🧪 Predicted Milk Quality: **{grade_map[prediction]}**")