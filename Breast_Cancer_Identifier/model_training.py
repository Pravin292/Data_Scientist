import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.datasets import load_breast_cancer

def train_breast_cancer_model():
    print("Fetching breast cancer dataset from sklearn...")
    try:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/breast_cancer.csv", index=False)
        print("Dataset saved to data/breast_cancer.csv")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Features and Target
    X = df.drop(columns=['target'])
    y = df['target'] # 0: Malignant, 1: Benign

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    features_list = list(X.columns)
    
    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Model": "Random Forest Classifier"
    }

    # Feature Importance for Explanation
    feature_importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    metrics["feature_importance"] = feature_importance
    
    # Dump files required for the application
    joblib.dump(model, "tree_model.pkl")
    joblib.dump(features_list, "feature_columns.pkl")
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining complete. Model and metrics saved.")

if __name__ == "__main__":
    train_breast_cancer_model()
