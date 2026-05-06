import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
SRC_SCRIPT = ROOT / "src" / "nid_project.py"


def run_experiment(max_samples: int, test_size: float, random_state: int, data_home: str, selector_k: int) -> dict:
    cmd = [
        sys.executable,
        str(SRC_SCRIPT),
        "--max-samples",
        str(max_samples),
        "--test-size",
        str(test_size),
        "--random-state",
        str(random_state),
        "--data-home",
        data_home,
        "--selector-k",
        str(selector_k),
    ]

    start = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - start
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed": elapsed,
        "cmd": " ".join(cmd),
    }


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing: {path.name}. Run experiment to generate it.")


def transform_by_technique_artifact(artifact: dict, x_scaled):
    kind = artifact.get("kind")
    if kind == "identity":
        return x_scaled
    if kind in {"pca", "lda", "svd"}:
        return artifact["transformer"].transform(x_scaled)
    if kind == "index_select":
        return x_scaled[:, artifact["selected_idx"]]
    return artifact["selector"].transform(x_scaled)


def predict_from_uploaded_df(df: pd.DataFrame, bundle: dict, technique_key: str, use_best_overall: bool) -> tuple[pd.DataFrame, str]:
    required_cols = bundle["feature_columns"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:8])}")

    x = df[required_cols].copy()
    x_prepared = bundle["preprocessor"].transform(x)
    x_scaled = bundle["scaler"].transform(x_prepared)

    if use_best_overall:
        model = bundle["best_overall_model"]
        technique_key = bundle["best_overall_technique_key"]
        model_name = bundle.get("best_overall_model_name", "best_overall_model")
    else:
        model = bundle["best_models_per_technique"][technique_key]
        model_name = bundle["best_model_names_per_technique"][technique_key]

    artifact = bundle["techniques"][technique_key]
    x_final = transform_by_technique_artifact(artifact, x_scaled)

    y_pred = model.predict(x_final)
    if hasattr(model, "predict_proba"):
        intrusion_score = model.predict_proba(x_final)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_score = model.decision_function(x_final)
        intrusion_score = 1 / (1 + np.exp(-raw_score))
    else:
        intrusion_score = np.full(shape=len(y_pred), fill_value=np.nan)

    result = df.copy()
    result["predicted_class"] = np.where(y_pred == 1, "intrusion", "normal")
    result["predicted_binary"] = y_pred
    result["intrusion_score"] = intrusion_score
    result["prediction_confidence"] = np.maximum(intrusion_score, 1 - intrusion_score)
    return result, model_name


def render_overview(meta: dict | None, metrics_df: pd.DataFrame | None, sample_meta: dict | None) -> None:
    st.subheader("Experiment Overview")
    if meta:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples Used", meta.get("samples_used", "-"))
        col2.metric("Original Features", meta.get("original_feature_count", "-"))
        col3.metric("PCA Components", meta.get("pca_components_retained", "-"))
        reduction = meta.get("dimension_reduction_ratio")
        reduction_pct = f"{reduction * 100:.2f}%" if isinstance(reduction, (int, float)) else "-"
        col4.metric("Reduction", reduction_pct)

        st.write(f"Best Overall Model: `{meta.get('best_model_overall', '-')}`")
        st.write(f"Best Overall Technique: `{meta.get('best_technique_overall', '-')}`")
    else:
        st.info("No experiment metadata found yet.")

    if sample_meta:
        st.subheader("Random Sample Loaded")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Index Loaded", sample_meta.get("sample_index_loaded", "-"))
        c2.metric("Predicted Class", sample_meta.get("predicted_class", "-"))
        conf = sample_meta.get("prediction_confidence")
        c3.metric("Prediction Confidence", f"{conf:.4f}" if isinstance(conf, (int, float)) else "-")

    if metrics_df is not None and not metrics_df.empty:
        st.subheader("Top Models")
        st.dataframe(metrics_df.head(10), use_container_width=True)


def render_metrics(metrics_df: pd.DataFrame | None) -> None:
    st.subheader("Model Benchmark")
    if metrics_df is None or metrics_df.empty:
        st.info("Run the experiment first to populate metrics.")
        return

    st.dataframe(metrics_df, use_container_width=True)

    metric_col = st.selectbox(
        "Select metric for bar chart",
        ["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds", "predict_seconds"],
        index=3,
    )
    chart_df = metrics_df[["model", metric_col]].set_index("model")
    st.bar_chart(chart_df)

    show_image_if_exists(OUTPUTS_DIR / "model_performance_heatmap.png", "Model Performance Heatmap (F1)")


def render_plots() -> None:
    st.subheader("Core Visualizations")
    show_image_if_exists(OUTPUTS_DIR / "pca_2d_projection.png", "PCA 2D Projection")
    show_image_if_exists(OUTPUTS_DIR / "pca_explained_variance.png", "PCA Explained Variance")
    show_image_if_exists(OUTPUTS_DIR / "correlation_heatmap.png", "Correlation Heatmap")
    show_image_if_exists(OUTPUTS_DIR / "feature_distribution.png", "Feature Distribution")
    show_image_if_exists(
        OUTPUTS_DIR / "filter_methods_accuracy_comparison.png",
        "Filter Methods - Accuracy Comparison",
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(OUTPUTS_DIR / "confusion_matrix_baseline.png", "Best No-DR Confusion Matrix")
    with col2:
        show_image_if_exists(OUTPUTS_DIR / "confusion_matrix_pca.png", "Best PCA Confusion Matrix")

    show_image_if_exists(OUTPUTS_DIR / "confusion_matrix_best_overall.png", "Best Overall Confusion Matrix")
    show_image_if_exists(OUTPUTS_DIR / "feature_importance.png", "Feature Importance")
    show_image_if_exists(OUTPUTS_DIR / "shap_waterfall.png", "SHAP Waterfall Explanation")
    show_image_if_exists(
        OUTPUTS_DIR / "prediction_confidence_distribution.png",
        "Prediction Confidence Distribution",
    )

    st.subheader("Dimensionality Reduction Evidence")
    col3, col4 = st.columns(2)
    with col3:
        show_image_if_exists(OUTPUTS_DIR / "dataset_before_dr_preview.png", "Before DR")
    with col4:
        show_image_if_exists(OUTPUTS_DIR / "dataset_after_dr_preview.png", "After PCA DR")


def render_dataset_views() -> None:
    st.subheader("Dataset Views")

    feature_file = OUTPUTS_DIR / "feature_columns.txt"
    if feature_file.exists():
        feature_lines = feature_file.read_text(encoding="utf-8").splitlines()
        st.write(f"Total original columns: **{len(feature_lines)}**")
        st.code("\n".join(feature_lines), language="text")

    for title, filename in [
        ("Raw Sample", "raw_dataset_sample.csv"),
        ("Preprocessed Sample", "preprocessed_dataset_sample.csv"),
        ("PCA Sample", "pca_dataset_sample.csv"),
        ("Metrics", "metrics_comparison.csv"),
        ("Technique Summary", "technique_summary.csv"),
        ("Feature Importance", "feature_importance.csv"),
    ]:
        st.markdown(f"### {title}")
        path = OUTPUTS_DIR / filename
        df = load_csv(path)
        if df is None:
            st.info(f"Missing: {filename}")
            continue
        st.dataframe(df.head(300), use_container_width=True)
        st.download_button(
            label=f"Download {filename}",
            data=path.read_bytes(),
            file_name=filename,
            mime="text/csv",
        )


def render_inference() -> None:
    st.subheader("Predict Intrusion on New Input")
    bundle = load_bundle(OUTPUTS_DIR / "nid_inference_bundle.joblib")
    if bundle is None:
        st.info("Inference bundle not found. Run an experiment first.")
        return

    technique_map = bundle.get("technique_name_map", {})
    available_keys = list(technique_map.keys())

    use_best_overall = st.checkbox("Use best overall model (recommended)", value=True)

    if not use_best_overall:
        chosen_key = st.selectbox(
            "Choose technique for inference",
            available_keys,
            format_func=lambda k: technique_map.get(k, k),
        )
    else:
        chosen_key = bundle.get("best_overall_technique_key", available_keys[0])
        st.write(f"Using best overall technique: `{technique_map.get(chosen_key, chosen_key)}`")

    template_df = pd.DataFrame(columns=bundle["feature_columns"])
    st.download_button(
        "Download Input Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="nid_input_template.csv",
        mime="text/csv",
    )

    with st.expander("Required Input Columns"):
        st.code("\n".join(bundle["feature_columns"]), language="text")

    upload = st.file_uploader("Upload CSV with network connection rows", type=["csv"], key="infer_upload")
    if upload is None:
        st.caption("Tip: use the template, fill rows, and upload.")
        return

    input_df = pd.read_csv(upload)
    if input_df.empty:
        st.error("Uploaded CSV has no rows. Add at least one data row and upload again.")
        return

    st.write("Preview of uploaded input")
    st.dataframe(input_df.head(50), use_container_width=True)

    if st.button("Predict Intrusion", type="primary"):
        try:
            pred_df, model_name = predict_from_uploaded_df(input_df, bundle, chosen_key, use_best_overall)
            st.success(f"Prediction complete using: {model_name}")
            st.dataframe(pred_df.head(300), use_container_width=True)

            intrusion_count = int((pred_df["predicted_binary"] == 1).sum())
            normal_count = int((pred_df["predicted_binary"] == 0).sum())
            c1, c2 = st.columns(2)
            c1.metric("Predicted Intrusions", intrusion_count)
            c2.metric("Predicted Normal", normal_count)

            st.download_button(
                "Download Predictions CSV",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="nid_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="NID Dimensionality Reduction Dashboard", layout="wide")
    st.title("Network Intrusion Detection Dashboard")
    st.caption(
        "Compare Correlation FS, PCA, LDA, SVD, wrapper and floating selection methods, then run intrusion predictions."
    )

    with st.sidebar:
        st.header("Run Controls")
        max_samples = st.number_input("Max samples", min_value=1000, max_value=100000, value=5000, step=1000)
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
        selector_k = st.number_input("Selector k", min_value=5, max_value=40, value=20, step=1)
        data_home = st.text_input("Data cache folder", value="data_cache")
        run_clicked = st.button("Run Experiment", type="primary", use_container_width=True)

    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    if run_clicked:
        with st.spinner("Running full experiment..."):
            st.session_state.last_run = run_experiment(
                int(max_samples),
                float(test_size),
                int(random_state),
                data_home,
                int(selector_k),
            )

    if st.session_state.last_run:
        result = st.session_state.last_run
        if result["returncode"] == 0:
            st.success(f"Run completed in {result['elapsed']:.2f}s")
        else:
            st.error(f"Run failed in {result['elapsed']:.2f}s")

        with st.expander("Last run command and logs", expanded=False):
            st.code(result["cmd"], language="bash")
            st.text_area("STDOUT", result["stdout"], height=200)
            st.text_area("STDERR", result["stderr"], height=120)

    metrics_df = load_csv(OUTPUTS_DIR / "metrics_comparison.csv")
    meta = load_json(OUTPUTS_DIR / "experiment_meta.json")
    sample_meta = load_json(OUTPUTS_DIR / "random_sample_prediction.json")

    tab_overview, tab_metrics, tab_plots, tab_data, tab_infer = st.tabs(
        ["Overview", "Metrics", "Plots", "Dataset", "Inference"]
    )

    with tab_overview:
        render_overview(meta, metrics_df, sample_meta)
    with tab_metrics:
        render_metrics(metrics_df)
    with tab_plots:
        render_plots()
    with tab_data:
        render_dataset_views()
    with tab_infer:
        render_inference()


if __name__ == "__main__":
    main()
