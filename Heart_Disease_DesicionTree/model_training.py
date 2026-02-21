import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_heart_disease_model():
    print("Fetching heart disease dataset...")
    # Using public dataset from UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    try:
        df = pd.read_csv(url, header=None, names=columns, na_values="?")
        # Drop rows with missing values for simplicity in baseline
        df = df.dropna()
        
        # Convert target into binary classification (0 = no disease, 1-4 = disease)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/heart.csv", index=False)
        print("Dataset saved to data/heart.csv")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Features and Target
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Decision Tree Classifiers (Entropy & Gini)...")
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    metrics = {
        "Accuracy": round(acc, 4),
        "Criterion": "Entropy",
        "Max Depth": 5
    }

    # Feature Importance
    feature_importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    metrics["feature_importance"] = feature_importance
    
    # Dump files required for the application
    joblib.dump(model, "decision_tree_model.pkl")
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining complete. Model and metrics saved.")

if __name__ == "__main__":
    train_heart_disease_model()
