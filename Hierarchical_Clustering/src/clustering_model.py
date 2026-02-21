import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as sch
import os
import pathlib

# Get the absolute path of the script location
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_data():
    """Generates a synthetic customer dataset for clustering."""
    print("generating dataset...")
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, random_state=42)
    # Simulate realistic 'Annual Income (k$)' and 'Spending Score (1-100)'
    X[:, 0] = np.abs(X[:, 0] * 10) + 15
    X[:, 1] = np.abs(X[:, 1] * 10) + 10
    
    # Introduce some synthetic missing values to demonstrate preprocessing
    X[5, 0] = np.nan
    X[12, 1] = np.nan
    
    df = pd.DataFrame(X, columns=['Annual_Income', 'Spending_Score'])
    df.to_csv(DATASET_DIR / 'customer_data.csv', index=False)
    return df

def preprocess_data(df):
    """Handles missing values and scales features."""
    print("Preprocessing data...")
    # Fill missing values with median
    df.fillna(df.median(), inplace=True)
    
    # Scale dataset
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features

def plot_dendrogram(X, method='ward', metric='euclidean', title='Dendrogram'):
    """Plots a hierarchical dendrogram."""
    print(f"Plotting dendrogram using linkage='{method}'... ")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Customers')
    plt.ylabel(f'{metric.capitalize()} Distance')
    
    sch.dendrogram(sch.linkage(X, method=method, metric=metric))
    plt.axhline(y=10, color='r', linestyle='--')
    plt.savefig(OUTPUT_DIR / f'dendrogram_{method}.png')
    plt.close()

def execute_clustering(X, n_clusters, linkage='ward', metric='euclidean'):
    """Performs Agglomerative Clustering and returns labels."""
    print(f"Fitting Agglomerative Clustering (Linkage: {linkage} | Metric: {metric})...")
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
    y_hc = hc.fit_predict(X)
    return y_hc

def plot_clusters(df, y_hc, filename):
    """Generates a 2D visualization of the clusters."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Annual_Income'], y=df['Spending_Score'], hue=y_hc, palette='viridis', s=100)
    plt.title("Customer Segments (Hierarchical Clustering)")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Cluster")
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

if __name__ == "__main__":
    # 1. Generate & Load Data
    df = generate_synthetic_data()
    
    # 2. Preprocess
    scaled_data = preprocess_data(df)
    
    # 3. Exploratory Visualization: Dendrograms
    plot_dendrogram(scaled_data, method='ward', metric='euclidean', title='Ward Linkage (Euclidean)')
    plot_dendrogram(scaled_data, method='average', metric='euclidean', title='Average Linkage (Euclidean)')
    plot_dendrogram(scaled_data, method='single', metric='euclidean', title='Single Linkage (Euclidean)')
    
    # 4. Clustering (K=4, Ward)
    y_segments = execute_clustering(scaled_data, n_clusters=4, linkage='ward')
    plot_clusters(df, y_segments, 'final_clusters_ward.png')
    
    print("Project executed successfully! Check the 'output' folder for visual logs.")
