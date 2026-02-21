# ❤️ Heart Disease Assessor (Midnight Glass Edition)

An advanced **Decision Tree** cardiac risk profiling system based on the Cleveland heart disease dataset.

## 🏗 Project Structure
- `app.py`: Physiological profiling dashboard with risk assessment tabs.
- `model_training.py`: Script for fetching data, cleaning, and training the Entropy-based Decision Tree.
- `utils.py`: Utility functions for risk grading and importance plotting.
- `requirements.txt`: Environment dependencies.
- `data/`: Persistent storage for heart disease records.

## 🚀 Usage Instructions

### 1. Architecture Setup
Generate the model and local data:
```bash
python3 model_training.py
```

### 2. Assessment
Launch the clinical inference UI:
```bash
streamlit run app.py
```

## 🧠 Model Profile
- Splitting Criterion: **Entropy**
- Assessment Focus: Maximum Heart Rate, CP Type, and Major Vessel counts.
- UI Layout: Sidebar for patient physiology metrics.