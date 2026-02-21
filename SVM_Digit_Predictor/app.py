import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ---------------- Page config ----------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

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

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(56, 189, 248, 0.2);
}

/* Output Cards */
.result-card-success {
    background: rgba(16, 185, 129, 0.15);
    border-left: 5px solid #10b981;
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

st.markdown("<h1>✍️ Advanced Optical Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Computer Vision Support Vector Machine (SVM) Classification Kernel</p>", unsafe_allow_html=True)
st.markdown("---")

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
st.sidebar.markdown("### ⚙️ Kernel Engineering Parameters")

model_type = st.sidebar.radio(
    "Select Non-Linear Transformation Function:",
    ["RBF SVM", "Polynomial SVM"]
)

st.sidebar.markdown("---")
st.sidebar.caption("The RBF (Radial Basis Function) generally demonstrates extreme robust flexibility across high-dimensional geometric spaces.")

# ---------------- Train model ----------------
if model_type == "RBF SVM":
    model = SVC(kernel="rbf", C=50, gamma=0.01)
else:
    model = SVC(kernel="poly", degree=4, C=50, gamma=0.01, coef0=1)

model.fit(X_train, y_train)

# ---------------- Model accuracy ----------------
accuracy = model.score(X_test, y_test)

# ---------------- Image upload ----------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 📤 Image Upload Sequence")
uploaded_file = st.file_uploader(
    "Import scanned numerical digit matrices (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")

    # Invert colors (match dataset)
    image = Image.fromarray(255 - np.array(image))

    # Resize gently
    image = image.resize((8, 8), Image.BILINEAR)

    st.image(image, caption="Compressed Processing Matrix (8×8 resolution)", width=200)

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
    st.markdown(f"""
        <div class="result-card-success">
            <div class="metric-text">🧩 Statistical Classification: <b>{prediction}</b></div>
            <p style="color: #cbd5e1; margin-top: 5px;">Model successfully mapped input vector space. (Kernel: {model_type}, Precision: {accuracy:.4f})</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption("⚠️ Note: Geometric deformations or massive visual noise in raw uploads may result in probabilistic edge-case failures.")