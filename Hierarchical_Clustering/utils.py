import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def generate_cluster_scatter(df, labels, x_col, y_col):
    """Generates a styled Plotly scatter plot for clusters."""
    df_plot = df.copy()
    df_plot['Cluster'] = [f"Cluster {l}" for l in labels]
    
    fig = px.scatter(
        df_plot, 
        x=x_col, 
        y=y_col, 
        color='Cluster',
        title=f'Cluster Distribution: {x_col} vs {y_col}',
        template="plotly_dark"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#38bdf8')
    )
    return fig

def plot_dendrogram_matplotlib(Z, labels=None):
    """Plots the dendrogram using Matplotlib (since Plotly doesn't Have a native simple one without ff)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
    
    # Styling for Midnight Glass
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.set_title("Hierarchical Clustering Dendrogram", color='#38bdf8', size=15)
    ax.tick_params(colors='#e2e8f0')
    for spine in ax.spines.values():
        spine.set_color('#38bdf8')
        
    return fig
