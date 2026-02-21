import streamlit as st
import pandas as pd
import numpy as np
from model_training import perform_clustering_analysis
from utils import generate_cluster_scatter, plot_dendrogram_matplotlib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Hierarchical Clustering Analytics",
    page_icon="🕸️",
    layout="wide"
)

# -----------------------------
# Custom Styling "Midnight Glass"
# -----------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700; }
.glass-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }
.stSelectbox select, .stNumberInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 8px; font-weight: 600; padding: 0.6em 1.5em; border: none; transition: all 0.3s ease; width: 100%; margin-top: 15px; }
.stButton>button:hover { background: linear-gradient(90deg, #60a5fa, #a78bfa); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🕸️ Hierarchical Clustering Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Agglomerative Analysis with Dendrogram Visualization</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("📁 Data & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Loaded!")
    
    st.sidebar.markdown("---")
    n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=10, value=3)
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    
    # Distance Metric (Note: ward only works with euclidean)
    metrics = ["euclidean", "l1", "l2", "manhattan", "cosine"]
    affinity = st.sidebar.selectbox("Distance Metric", metrics if linkage_method != 'ward' else ["euclidean"])
    
    run_analysis = st.sidebar.button("Execute Clustering Analysis")

    # -----------------------------
    # Main UI Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Cluster Visualization", "🌳 Dendrogram Analysis", "📋 Raw Data"])

    with tab1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🧬 Agglomerative Cluster Distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col_x, col_y = st.columns(2)
            x_axis = col_x.selectbox("X-Axis Feature", numeric_cols, index=0)
            y_axis = col_y.selectbox("Y-Axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            if run_analysis:
                results, Z, error = perform_clustering_analysis(df, n_clusters, linkage_method, affinity)
                if error:
                    st.error(error)
                else:
                    fig = generate_cluster_scatter(df, results['labels'], x_axis, y_axis)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Initiate analysis via the sidebar button to view clusters.")
        else:
            st.warning("Dataset requires at least 2 numeric features for scatter visualization.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🪵 Hierarchical Tree Structure")
        if run_analysis:
            results, Z, error = perform_clustering_analysis(df, n_clusters, linkage_method, affinity)
            if not error:
                fig_dendro = plot_dendrogram_matplotlib(Z)
                st.pyplot(fig_dendro)
                st.caption(f"Strategy: {linkage_method.capitalize()} | Metric: {affinity.capitalize()}")
        else:
            st.info("Dendrogram will be generated upon analysis execution.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 📥 Loaded Dataset Excerpt")
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="glass-panel" style="text-align: center; padding: 50px;">', unsafe_allow_html=True)
    st.markdown("### 📤 Awaiting Dataset Upload")
    st.markdown("Please upload a CSV file via the sidebar to start the hierarchical clustering pipeline.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dummy data for demonstration
    if st.checkbox("Load Sample Coffee Shop Data"):
        # We can simulate a sample dataset if they don't have one
        sample_data = {
            "Transaction_Amount": np.random.randint(10, 100, 100),
            "Items_Count": np.random.randint(1, 10, 100),
            "Customer_Rating": np.random.uniform(1, 5, 100),
            "Visit_Frequency": np.random.randint(1, 30, 100)
        }
        df_sample = pd.DataFrame(sample_data)
        st.write("Sample Data Preview:")
        st.dataframe(df_sample.head())
        # To avoid complex logic here, I'll just suggest they upload a file.
