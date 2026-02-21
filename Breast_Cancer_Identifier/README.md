# 🩺 Breast Cancer Identifier (Midnight Glass Edition)

A professional **Random Forest** diagnostic tool for cellular malignancy detection.

## 🏗 Project Structure
- `app.py`: Biometric diagnostic UI with wide-layout tabs.
- `model_training.py`: Logic to fetch the Sklearn dataset, train the RF classifier, and dump weights.
- `utils.py`: Handles complex biomarker importance visualization.
- `requirements.txt`: Clean environment specifications.
- `data/`: CSV biopsy records directory.

## 🚀 Usage Instructions

### 1. Diagnostic Matrix Generation
Retrain the model engine:
```bash
python3 model_training.py
```

### 2. Launch Predictor
```bash
streamlit run app.py
```

## 🔬 Scientific Core
- Model: **Random Forest Classifier** (100 Estimators)
- Features: Analyses 30 unique cellular biomarkers.
- Performance: Includes Precision and Recall metrics within the dashboard.
