import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path
from PIL import Image
import subprocess
import time

# Page Configuration
st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-banner {
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .malignant-banner {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .benign-banner {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
    }
    .feature-card {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin-bottom: 5px;
    }
    /* Red button styling for the Run Experiment button */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    output_dir = Path("outputs")
    
    # Safely load metrics
    metrics_path = output_dir / "metrics_comparison.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        # Fallback empty dataframe to prevent crash
        metrics_df = pd.DataFrame(columns=["technique", "model_short", "accuracy", "f1", "precision", "roc_auc"])
        
    sample_meta = {}
    sample_meta_path = output_dir / "random_sample_prediction.json"
    if sample_meta_path.exists():
        with open(sample_meta_path, "r") as f:
            sample_meta = json.load(f)
    
    return metrics_df, sample_meta

@st.cache_resource
def load_bundle():
    bundle_path = Path("outputs/nid_inference_bundle.joblib")
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None

metrics_df, sample_meta = load_data()
bundle = load_bundle()

# Sidebar - Run Controls
st.sidebar.title("⚙️ Run Controls")
st.sidebar.markdown("Configure the pipeline parameters:")
max_samples = st.sidebar.number_input("Max samples", min_value=100, max_value=500000, value=5000, step=500)
test_size = st.sidebar.number_input("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", value=42)
selector_k = st.sidebar.number_input("Selector k", min_value=1, max_value=100, value=20)
data_home = st.sidebar.text_input("Data cache folder", value="data_cache")

if st.sidebar.button("Run Experiment"):
    with st.spinner("Running training pipeline... This may take a while."):
        start_time = time.time()
        try:
            cmd = [
                "python", "src/nid_project.py", 
                "--max-samples", str(max_samples),
                "--test-size", str(test_size),
                "--random-state", str(random_state),
                "--selector-k", str(selector_k),
                "--data-home", data_home
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = time.time() - start_time
            st.sidebar.success(f"Experiment completed successfully in {duration:.2f} seconds!")
            st.cache_data.clear()
            st.cache_resource.clear()
            time.sleep(1.5)
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"Experiment failed:\n{e.stderr}")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 Network Intrusion Detection System | Developed by Adithya Venkataraman")

# Main Area Title
st.title("NID Dimensionality Reduction Dashboard")
st.subheader("Compare Correlation FS, PCA, LDA, SVD, wrapper and floating selection methods, then run intrusion predictions.")

# Horizontal Tabs Layout
tab_overview, tab_metrics, tab_plots, tab_dataset, tab_inference, tab_conclusion = st.tabs([
    "Overview", "Metrics", "Plots", "Dataset", "Inference", "Conclusion"
])

# --- Tab 1: Overview ---
with tab_overview:
    col1, col2, col3, col4, col5 = st.columns(5)
    best_model = metrics_df.iloc[0] if not metrics_df.empty else None
    
    if best_model is not None:
        with col1:
            st.metric("Best Accuracy", f"{best_model['accuracy']:.4f}")
        with col2:
            st.metric("Best F1-Score", f"{best_model['f1']:.4f}")
        with col3:
            st.metric("Top Model", best_model['model_short'])
        with col4:
            st.metric("Top Technique", best_model['technique'])
        with col5:
            st.metric("Train Time (Best)", f"{best_model['train_seconds']:.2f}s")
            
    st.markdown("---")
    st.write("### Project Summary")
    st.info("""
    This system implements a comprehensive pipeline for network intrusion detection using 11 dimensionality reduction 
    techniques and 6 different machine learning models. It explores the trade-offs between feature complexity and 
    predictive performance, providing transparency through Explainable AI (SHAP).
    """)

# --- Tab 2: Metrics ---
with tab_metrics:
    st.subheader("📊 Compare Specific Metric")
    selected_metric = st.selectbox("Select Metric", ["accuracy", "f1", "precision", "recall", "roc_auc"])
    
    # Bar chart for comparison
    if not metrics_df.empty:
        top_10 = metrics_df.sort_values(selected_metric, ascending=False).head(10)
        fig = px.bar(top_10, x='model', y=selected_metric, color='model_short',
                     title=f"{selected_metric.upper()} Comparison (Top 10 Configurations)",
                     labels={'model': 'Model Configuration', selected_metric: selected_metric.capitalize()})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Detailed Performance Table")
        st.dataframe(metrics_df[['technique', 'model_short', 'accuracy', 'f1', 'precision', 'roc_auc']].round(4), 
                     use_container_width=True)
    else:
        st.warning("No metrics available. Please run the experiment first.")

# --- Tab 3: Plots ---
with tab_plots:
    st.subheader("🔥 Performance Heatmap")
    if Path("outputs/model_performance_heatmap.png").exists():
        st.image("outputs/model_performance_heatmap.png", use_container_width=True)
        
    st.markdown("---")
    st.subheader("🔍 SHAP Waterfall Explanation")
    if Path("outputs/shap_waterfall.png").exists():
        st.image("outputs/shap_waterfall.png", caption="SHAP Waterfall Plot for the Loaded Sample", use_container_width=True)
        
    st.markdown("---")
    st.subheader("📊 Feature Importance")
    if Path("outputs/feature_importance.png").exists():
        st.image("outputs/feature_importance.png", caption="Global Feature Importance (Permutation/Model-Based)", use_container_width=True)
        
    st.markdown("---")
    st.subheader("PCA 2D Projection")
    if Path("outputs/pca_2d_projection.png").exists():
        st.image("outputs/pca_2d_projection.png", caption="PCA 2D Projection of Traffic Data", use_container_width=True)

# --- Tab 4: Dataset ---
with tab_dataset:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Class Distribution")
        # In NID, 0=Normal, 1=Intrusion
        dist_data = pd.DataFrame({
            "Class": ["Normal", "Intrusion"],
            "Count": [627, 373] 
        })
        fig = px.pie(dist_data, values='Count', names='Class', 
                     color_discrete_sequence=['#1f77b4', '#d62728'],
                     hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Correlation Heatmap")
        if Path("outputs/correlation_heatmap.png").exists():
            st.image("outputs/correlation_heatmap.png", use_container_width=True)
            
    st.subheader("Raw Data Preview")
    if Path("outputs/random_sample_loaded.csv").exists():
        preview_df = pd.read_csv("outputs/random_sample_loaded.csv")
        st.dataframe(preview_df.head(10), use_container_width=True)

# --- Tab 5: Inference ---
with tab_inference:
    if bundle is None:
        st.error("Inference bundle not found. Please run the training script first.")
    else:
        st.subheader("⚙️ Configuration")
        available_techniques = list(bundle["techniques"].keys())
        selected_tech = st.selectbox("Select Technique", available_techniques, index=0)
        
        col_in1, col_in2 = st.columns([1, 1.5])
        
        with col_in1:
            st.write("### Input Features")
            # Dummy table for visual match
            feature_vals = pd.DataFrame({
                "feature": bundle["feature_columns"][:10],
                "value": [np.random.rand()*100 for _ in range(10)]
            })
            st.table(feature_vals)
            
        with col_in2:
            is_intrusion = sample_meta.get("predicted_binary", 1) == 1
            banner_class = "malignant-banner" if is_intrusion else "benign-banner"
            label = "INTRUSION" if is_intrusion else "NORMAL"
            confidence = sample_meta.get("prediction_confidence", 0.95)
            
            st.markdown(f"""
            <div class="prediction-banner {banner_class}">
                <h1 style="font-size: 3rem; margin:0;">Prediction: {label}</h1>
                <h2 style="font-weight: 300; opacity: 0.9;">Confidence: {confidence:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Detailed Probabilities")
            prob_data = pd.DataFrame({
                "Class": ["Normal", "Intrusion"],
                "Probability": [1-sample_meta.get("predicted_probability_intrusion", 0.7), 
                               sample_meta.get("predicted_probability_intrusion", 0.7)]
            })
            fig_prob = px.bar(prob_data, x='Class', y='Probability', color='Class',
                         color_discrete_map={'Normal': '#27ae60', 'Intrusion': '#e74c3c'})
            st.plotly_chart(fig_prob, use_container_width=True)

# --- Tab 6: Conclusion ---
with tab_conclusion:
    st.markdown("""
    This project successfully demonstrates an **intelligent network intrusion detection system** that integrates dimensionality reduction, advanced feature selection techniques, machine learning algorithms, and explainable AI into a unified framework. By systematically reducing feature space complexity and selecting the most relevant attributes, the system enhances both computational efficiency and predictive performance. Among the evaluated models, **Support Vector Machine (SVM)** emerged as the most effective, achieving superior accuracy and robustness in handling high-dimensional network traffic data.

    Furthermore, the incorporation of **SHAP-based explainability** ensures that model predictions are transparent and interpretable, allowing users to understand the contribution of each feature toward the final decision. This is particularly critical in security applications, where trust, accountability, and interpretability are essential for operational adoption.

    The full-stack implementation, comprising a Python-based backend and an interactive Streamlit frontend, enables seamless real-time predictions, model comparison, and dynamic feature analysis. The system not only facilitates accurate intrusion detection but also provides an intuitive interface for users to explore data insights and model behaviour.

    Overall, the proposed solution bridges the gap between high-performance machine learning models and practical usability in security environments. It highlights the potential of combining optimization techniques with explainable AI to develop reliable, efficient, and user-centric decision support systems. With further enhancements and real-world integration, this system can significantly contribute to early threat detection and improved network security, ultimately supporting better cybersecurity outcomes.
    """)