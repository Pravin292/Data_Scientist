import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
import joblib
import json

def perform_clustering_analysis(df, n_clusters=3, linkage_method='ward', affinity='euclidean'):
    """
    Core ML logic for Hierarchical Clustering.
    Separated from UI to maintain modularity.
    """
    # Preprocessing: Handle only numeric columns for clustering
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None, None, "No numeric data found for clustering."

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Perform Agglomerative Clustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_method, 
        metric=affinity # sklearn 1.2+ uses metric instead of affinity for AgglomerativeClustering
    )
    cluster_labels = model.fit_predict(scaled_data)

    # Compute Linkage for Dendrogram (scipy)
    # Note: Scipy linkage only supports 'euclidean' for 'ward'
    Z = linkage(scaled_data, method=linkage_method, metric=affinity if linkage_method != 'ward' else 'euclidean')

    results = {
        "labels": cluster_labels.tolist(),
        "n_clusters": n_clusters,
        "linkage": linkage_method,
        "metric": affinity
    }

    return results, Z, None

def save_clustering_artifacts(results, Z, filename="clustering_results.json"):
    """Saves the results for persistence if needed."""
    with open(filename, "w") as f:
        json.dump(results, f)
    joblib.dump(Z, "linkage_matrix.pkl")
