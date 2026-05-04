# Network Intrusion Detection using Dimensionality Reduction

## 1. Title Page

- **Project Title:** Network Intrusion Detection using Dimensionality Reduction (PCA)
- **Student Name:** ____________________
- **Roll Number:** ____________________
- **Course / Department:** ____________________
- **Institution:** ____________________
- **Submission Date:** ____________________

## 2. Abstract

Network Intrusion Detection Systems (NIDS) classify network traffic as normal or malicious. Real-world network traffic datasets are often high-dimensional, which increases training cost and may include redundant/noisy features. This project applies dimensionality reduction using Principal Component Analysis (PCA) and compares model performance against a baseline without dimensionality reduction. We use the KDDCup99 dataset (subset SA) and frame the task as binary classification (normal vs intrusion). A Random Forest classifier is trained on standardized full features (baseline) and on PCA-reduced features (reduced model). Performance is measured using Accuracy, Precision, Recall, F1-score, and ROC-AUC, along with training/prediction time. Results show whether PCA can maintain competitive detection quality while reducing feature space and computational overhead. The project demonstrates the practical benefit of dimensionality reduction in cybersecurity pipelines and provides a reproducible workflow for NID experiments.

## 3. Introduction

Network infrastructure is constantly exposed to cyber threats such as denial-of-service attacks, probing, and unauthorized access. Automated intrusion detection based on machine learning helps security teams detect anomalous behavior at scale. However, NID data typically contains many correlated features, making models computationally heavy and potentially less robust.

Dimensionality reduction can address this by mapping high-dimensional features into a lower-dimensional space that preserves key structure. In this project, PCA is used to reduce dimensions before model training. We analyze the trade-off between detection effectiveness and computational efficiency.

## 4. Problem Statement

Can dimensionality reduction (PCA) reduce feature dimensionality and training time in a Network Intrusion Detection task while preserving or improving detection performance compared to a baseline model without dimensionality reduction?

## 5. Objectives

- Build a machine learning pipeline for binary network intrusion detection.
- Train a baseline classifier on full preprocessed features.
- Apply PCA for dimensionality reduction and retrain the same classifier.
- Compare both approaches using classification metrics and runtime.
- Analyze the impact of feature reduction on model performance.

## 6. Dataset Description

- **Dataset:** KDDCup99 (scikit-learn fetcher, subset=`SA`, percent10=`True`)
- **Domain:** Network traffic connection records
- **Original features:** 41
- **Target classes:** multiple attack types + normal traffic
- **Project target mapping:**
  - `normal.` -> 0 (normal)
  - all other labels -> 1 (intrusion)

### 6.1 Data Preparation

- Decode categorical byte-string columns.
- Encode categorical features using `OrdinalEncoder`.
- Standardize all features using `StandardScaler`.
- Split data into train/test with stratification.

## 7. Methodology

### 7.1 Baseline Model (Without Dimensionality Reduction)

1. Preprocess data (encoding + scaling).
2. Train `RandomForestClassifier`.
3. Evaluate on test set.

### 7.2 Reduced Model (With PCA)

1. Apply PCA on scaled training features.
2. Retain components explaining ~95% variance.
3. Train `RandomForestClassifier` on PCA-transformed features.
4. Evaluate on transformed test set.

### 7.3 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training time and prediction time

## 8. Implementation Details

- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, scikit-learn, Matplotlib
- **Main script:** `src/nid_project.py`
- **Command used:**

```powershell
python src/nid_project.py --max-samples 5000 --test-size 0.2 --random-state 42 --data-home data_cache
```

## 9. Results

Fill values from `outputs/metrics_comparison.csv` and `outputs/experiment_meta.json`.

### 9.1 Quantitative Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) | Predict Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RandomForest (No DR) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.1784 | 0.2297 |
| RandomForest + PCA | 0.9990 | 1.0000 | 0.9706 | 0.9851 | 1.0000 | 2.0620 | 0.1721 |

### 9.2 Dimensionality Reduction Summary

- Original feature count: 41
- PCA components retained: 20
- Explained variance: 0.9600 (~95.9983%)
- Reduction ratio: 0.5122 (~51.22%)

### 9.3 Visualizations

- PCA explained variance plot (`outputs/pca_explained_variance.png`)
- Baseline confusion matrix (`outputs/confusion_matrix_baseline.png`)
- PCA confusion matrix (`outputs/confusion_matrix_pca.png`)
- Dataset preview before DR (`outputs/dataset_before_dr_preview.png`)
- Dataset preview after DR (`outputs/dataset_after_dr_preview.png`)

## 10. Discussion

- Compare whether PCA reduced computational cost.
- Discuss any performance gain/loss after dimensionality reduction.
- Explain why reduced dimensions may remove noise but also lose fine-grained information.
- Mention dataset limitations (class imbalance, synthetic nature of KDDCup99).

## 11. Conclusion

This project demonstrates a complete NID workflow and evaluates the role of PCA in reducing feature dimensionality. In our run, PCA reduced dimensions by about 51.22% (41 to 20) while preserving very high classification quality (Accuracy 99.9%, ROC-AUC 1.0). Compared to the no-DR baseline, the PCA model trained and predicted slightly faster, with only a small drop in recall. Therefore, PCA provided a favorable efficiency-performance trade-off for this binary intrusion detection setup.

## 12. Future Work

- Use NSL-KDD or UNSW-NB15 for modern benchmarking.
- Compare PCA with nonlinear reduction (Autoencoder, UMAP).
- Extend to multi-class attack-type detection.
- Add threshold tuning and cost-sensitive learning for security-critical recall.

## 13. References

1. KDD Cup 1999 Data: UCI KDD Archive.
2. Scikit-learn documentation: `fetch_kddcup99`, PCA, RandomForest.
3. Additional papers/articles used in your analysis.

## 14. Appendix

- Command-line logs
- Additional plots/tables
- Source code snippets from `src/nid_project.py`
