import streamlit as st
import joblib
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write(
    "This app predicts the **diabetes outcome value** using a "
    "**Gradient Boosting Regression model**."
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_gradient_boosting_model.pkl")
    feature_columns = joblib.load("diabetes_feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ---------------- User Input ----------------
st.subheader("🔢 Enter Patient Data")

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
    st.success(f"📊 Predicted Outcome Value: **{prediction:.2f}**")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Model: Gradient Boosting Regressor | Built with Streamlit")