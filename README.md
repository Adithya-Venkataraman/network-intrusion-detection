# Network Intrusion Detection with Multiple Dimensionality Reduction Techniques

This project builds a full NID pipeline and benchmarks these techniques in one app:

- Correlation-based Feature Selection
- PCA
- LDA
- SVD
- Filter-based SelectKBest
- Wrapper-based RFE
- Sequential Floating Forward Selection (SFFS)
- Sequential Floating Backward Selection (SFBS)

Each technique is compared across models (`LogisticRegression`, `SVM-RBF`, `KNN`, `RandomForest`) and visualized in Streamlit.

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run Full Experiment

```powershell
python src/nid_project.py --max-samples 5000 --test-size 0.2 --random-state 42 --data-home data_cache --selector-k 20
```

## 3) Launch Web App

```powershell
python -m streamlit run streamlit_app.py
```

## 4) What You Get (Outputs)

Saved in `outputs/`:

- `metrics_comparison.csv` (full model performance table)
- `technique_summary.csv` (best per technique)
- `model_performance_heatmap.png`
- `filter_methods_accuracy_comparison.png`
- `pca_2d_projection.png`
- `correlation_heatmap.png`
- `feature_distribution.png`
- `feature_importance.png` + `feature_importance.csv`
- `shap_waterfall.png`
- `prediction_confidence_distribution.png`
- `random_sample_prediction.json` (sample index loaded + confidence)
- `confusion_matrix_baseline.png`, `confusion_matrix_pca.png`, `confusion_matrix_best_overall.png`
- `dataset_before_dr_preview.png`, `dataset_after_dr_preview.png`
- `nid_inference_bundle.joblib` (trained artifacts for inference)

## 5) Intrusion Prediction on New CSV

Template file:

- `outputs/nid_input_template.csv`

CLI inference:

```powershell
python src/predict_intrusion.py --input path_to_new_data.csv --mode auto
```

For a specific technique:

```powershell
python src/predict_intrusion.py --input path_to_new_data.csv --mode per_technique --technique svd
```

Allowed technique keys:

- `no_dr`, `corr_fs`, `pca`, `lda`, `svd`, `filter_kbest`, `wrapper_rfe`, `sffs`, `sfbs`

## 6) Streamlit Sections for Report Screenshots

- Overview: final observations + random sample index loaded + prediction confidence
- Metrics: model performance table + heatmap
- Plots: PCA 2D, correlation, distributions, confusion matrices, feature importance, SHAP waterfall
- Dataset: raw/preprocessed/PCA samples and downloadable files
- Inference: upload CSV and see `normal` vs `intrusion`

## 7) Deploy for Others

1. Push repo to GitHub
2. Open [share.streamlit.io](https://share.streamlit.io)
3. Select repo + branch
4. Entrypoint: `streamlit_app.py`
5. Deploy and share app URL
