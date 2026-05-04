# Network Intrusion Detection using Dimensionality Reduction

This project builds a Network Intrusion Detection (NID) pipeline and compares:

- Baseline (No DR): `LogisticRegression`, `SVM-RBF`, `KNN`, `RandomForest`
- Reduced (PCA): same model set on PCA-transformed features

Dataset used by default: `KDDCup99` (`subset=SA`, `percent10=True`) from scikit-learn.

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run Experiment

```powershell
python src/nid_project.py --max-samples 5000 --test-size 0.2 --random-state 42 --data-home data_cache
```

## 3) Generated Outputs

All artifacts are saved in `outputs/`:

- `metrics_comparison.csv`: model-wise Accuracy, Precision, Recall, F1, ROC-AUC, training/prediction time
- `experiment_meta.json`: feature counts, PCA components retained, explained variance
- `pca_explained_variance.png`: cumulative explained variance curve
- `confusion_matrix_baseline.png`: best model confusion matrix in No-DR phase
- `confusion_matrix_pca.png`: best model confusion matrix in PCA phase
- `nid_inference_bundle.joblib`: saved preprocessors + trained models for prediction
- `nid_input_template.csv`: CSV template for inference input

## 4) Streamlit Frontend

Run dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

Features in UI:

- Run new experiments from sidebar controls
- Compare all models across No-DR and PCA phases
- View confusion matrices, PCA variance plot, and before/after DR previews
- Inspect raw/preprocessed/PCA dataset samples and download CSV files
- Upload new CSV rows and predict `normal` vs `intrusion`

## 5) Intrusion Prediction on New Input

After training, use:

```powershell
python src/predict_intrusion.py --input path_to_new_data.csv --mode auto
```

Output:

- `outputs/nid_predictions.csv` with:
  - `predicted_class` (`normal` or `intrusion`)
  - `predicted_binary` (`0` or `1`)
  - `intrusion_score` (higher means more attack-like)

Important:

- Input CSV must include all required feature columns.
- Input CSV must have at least one data row (header-only file will fail).

## 6) Deploy for Others (Streamlit Cloud)

1. Commit project files (`README.md`, `requirements.txt`, `streamlit_app.py`, `src/`, `report/`).
2. Push repo to GitHub.
3. Open [share.streamlit.io](https://share.streamlit.io).
4. Create app with entrypoint file: `streamlit_app.py`.
5. Share generated public URL.

## 7) Notes

- Target is converted to binary:
  - `normal.` -> 0
  - any attack class -> 1
- The script uses stratified train/test split.
- If runtime is high, reduce `--max-samples`.

## 8) Suggested Extensions

- Replace PCA with Autoencoder latent features
- Add multi-class attack type classification
- Evaluate other models (XGBoost, LightGBM, SVM)
- Try NSL-KDD or UNSW-NB15 for a more modern dataset
