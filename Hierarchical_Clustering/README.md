# Hierarchical Customer Segmentation

An end-to-end Machine Learning project demonstrating the implementation of **Agglomerative Hierarchical Clustering** for customer segmentation. This repository showcases distance metric evaluations, linkage methods, and robust data preprocessing techniques.

---

## 🚀 Project Overview

In unsupervised learning, we often don't have predefined labels. Hierarchical clustering is an algorithm that groups similar objects into clusters. The endpoint is a set of clusters, where each cluster is distinct from every other cluster, and the objects within each cluster are broadly similar to each other.

This project uses a synthetic customer dataset comprising **Annual Income** and **Spending Score** to establish behavioral purchasing segments.

---

## 🧠 Methodology

### 1. Data Preprocessing
Real-world datasets are rarely perfect. The pipeline automatically:
- Identifies missing `NaN` values and imputes them using the statistical median.
- Extensively scales numerical features using `StandardScaler` so that `Annual Income` (magnitudes of thousands) doesn't mathematically dominate `Spending Score` (0-100).

### 2. Distance Metrics
The core of clustering relies on defining "similarity". This project evaluates:
- **Euclidean Distance**: The ordinary straight-line distance between two points.
- **Manhattan Distance (City Block)**: The distance between two points measured along axes at right angles.

### 3. Linkage Criteria
Agglomerative clustering uses a "bottom-up" approach. Each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. The project computes:
- **Ward Linkage**: Minimizes the variance of the clusters being merged (often yields the most uniform clusters).
- **Single Linkage**: Merges based on the minimum distance between components of two clusters.
- **Average Linkage**: Merges based on the average distance between all components of two clusters.

### 4. Dendrogram Visualization
A dendrogram is a tree-like diagram that records the sequences of merges or splits. By drawing a horizontal threshold line, we can deterministically find the optimal number of clusters (in this case, `k=4`).

---

## 📂 Repository Structure

```
Hierarchical_Clustering/
│
├── dataset/ 
│   └── customer_data.csv                 # Generated dataset
├── notebooks/
│   └── hierarchical_clustering.ipynb     # Interactive Jupyter Notebook
├── src/
│   └── clustering_model.py               # Main OOP execution script
├── output/                               # Generated visual plots
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🛠️ How to Run

1. **Install Dependencies**  
Ensure you have the requirements loaded:
```bash
pip install -r requirements.txt
```

2. **Execute the Pipeline**  
Navigate to the source directory and run the model script:
```bash
cd src
python clustering_model.py
```

3. **Check Outputs**  
The script will actively generate datasets and populate the `output/` directory with `dendrogram_ward.png`, `dendrogram_single.png`, `dendrogram_average.png`, and a final scatterplot of the clusters.

---
© Pravin | Machine Learning Engineer
