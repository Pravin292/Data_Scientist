# 🕸️ Hierarchical Clustering Analytics (Midnight Glass Edition)

A high-fidelity **Unsupervised Learning** dashboard for performing Agglomerative Clustering on custom datasets.

## 🏗 Project Structure
- `app.py`: Streamlit UI with CSV uploader, linkage selection, and interactive tabs.
- `model_training.py`: Core logic for `StandardScaler` fitting, `AgglomerativeClustering`, and Scipy linkage calculation.
- `utils.py`: Visualization suite featuring Matplotlib dendrograms and Plotly scatter plots.
- `requirements.txt`: Project-specific dependencies.
- `data/`: Directory for persistent or uploaded datasets.

## 🚀 Usage Instructions

### 1. Requirements
Ensure dependencies are satisfied:
```bash
pip install -r requirements.txt
```

### 2. Launch Dashboard
Run the analytics engine:
```bash
streamlit run app.py
```

## 📊 Capabilities
- **Flexible Linkage**: Support for Ward, Complete, Average, and Single methods.
- **Dynamic Metrics**: Toggle between Euclidean, Manhattan, and Cosine distances.
- **Interactive Tree**: Generates a professional dendrogram to visualize cluster merging.
- **Cluster Visualization**: Selectable X/Y axes to inspect cluster separation in 2D space.
