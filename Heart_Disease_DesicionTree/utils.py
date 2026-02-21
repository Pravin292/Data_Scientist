import joblib
import json
import pandas as pd
import numpy as np
import plotly.express as px

def load_model_and_metrics():
    """Loads the serialized model and performance metrics."""
    try:
        model = joblib.load("decision_tree_model.pkl")
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return model, metrics
    except FileNotFoundError:
        return None, None

def predict_heart_disease(model, input_data):
    """Runs inference on user input."""
    prediction = model.predict(input_data)
    # The new target mapped is 0 or 1
    return int(prediction[0])

def generate_feature_importance_plot(metrics):
    """Generates a styled Plotly chart for feature importance."""
    fi = metrics.get('feature_importance', {})
    if not fi:
        return None
        
    df_fi = pd.DataFrame({
        'Feature': list(fi.keys()),
        'Importance': list(fi.values())
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_fi, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Decision Tree Feature Importance (Entropy)'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#38bdf8'),
        xaxis=dict(title="Information Gain (Importance)", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="")
    )
    fig.update_traces(marker_color='#ef4444')
    
    return fig

def get_risk_info(prediction_class):
    """Returns stylistic parameters based on prediction."""
    if prediction_class == 1:
        return "⚠️ High Risk Detected", "result-card-danger"
    else:
        return "✅ Low Risk", "result-card-success"
