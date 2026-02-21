import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ---------------- Page config ----------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of a digit (0–9)")

# ---------------- Load dataset ----------------
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- Scaling (CRITICAL FIX) ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Sidebar: Model choice ----------------
st.sidebar.header("Model Selection")

model_type = st.sidebar.radio(
    "Choose Non-Linear Model",
    ["RBF SVM", "Polynomial SVM"]
)

# ---------------- Train model ----------------
if model_type == "RBF SVM":
    model = SVC(
        kernel="rbf",
        C=50,
        gamma=0.01
    )
else:
    model = SVC(
        kernel="poly",
        degree=4,     # higher degree captures curves better
        C=50,
        gamma=0.01,
        coef0=1
    )

model.fit(X_train, y_train)

# ---------------- Model accuracy ----------------
accuracy = model.score(X_test, y_test)

# ---------------- Image upload ----------------
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")

    # Invert colors (match dataset)
    image = Image.fromarray(255 - np.array(image))

    # Resize gently
    image = image.resize((8, 8), Image.BILINEAR)

    st.image(image, caption="Processed Image (8×8)", width=200)

    # Convert to array
    image_array = np.array(image).astype(np.float32)

    # Normalize like digits dataset
    image_array = (image_array / 255.0) * 16

    # Flatten
    image_array = image_array.reshape(1, -1)

    # SCALE INPUT (THIS IS HUGE)
    image_array = scaler.transform(image_array)

    # Predict
    prediction = model.predict(image_array)[0]

    # Output
    st.subheader("Prediction Result")
    st.success(f"Predicted Digit: **{prediction}**")

    st.subheader("Model Performance")
    st.info(f"Model Used: **{model_type}**")
    st.info(f"Accuracy on Test Data: **{accuracy:.4f}**")

    st.warning(
        "⚠️ Note: This model uses 8×8 images. "
        "Complex digits (like 2, 5, 8) may still be confused."
    )