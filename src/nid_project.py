import argparse
import json
import time
from pathlib import Path

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _decode_if_bytes(value):
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def load_nid_dataset(
    max_samples: int, random_state: int, data_home: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = fetch_kddcup99(
        subset="SA",
        percent10=True,
        shuffle=True,
        random_state=random_state,
        as_frame=True,
        data_home=str(data_home),
    )

    x = dataset.data.copy()
    y_raw = dataset.target.copy()

    for column in x.columns:
        if x[column].dtype == object:
            x[column] = x[column].map(_decode_if_bytes)

    y_raw = y_raw.map(_decode_if_bytes)
    y = (y_raw != "normal.").astype(int).to_numpy()

    if max_samples and max_samples < len(x):
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )

    return x, y


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_start

    pred_start = time.perf_counter()
    y_pred = model.predict(x_test)
    pred_seconds = time.perf_counter() - pred_start

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
        roc_auc = roc_auc_score(y_test, y_score)
    else:
        roc_auc = np.nan

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "train_seconds": train_seconds,
        "predict_seconds": pred_seconds,
        "y_pred": y_pred,
        "fitted_model": model,
    }


def run_benchmark(model_defs, phase_name: str, x_train, y_train, x_test, y_test) -> list[dict]:
    results = []
    for model_name, model in model_defs.items():
        full_name = f"{model_name} ({phase_name})"
        print(f"Training {full_name}...")
        results.append(evaluate_model(full_name, clone(model), x_train, y_train, x_test, y_test))
    return results


def plot_explained_variance(pca: PCA, output_dir: Path) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str, output_dir: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def save_table_image(df: pd.DataFrame, title: str, filename: str, output_dir: Path) -> None:
    preview = df.copy()
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Network Intrusion Detection with and without PCA dimensionality reduction"
    )
    parser.add_argument("--max-samples", type=int, default=50000, help="Max rows to use")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-home",
        type=str,
        default="data_cache",
        help="Local directory for dataset download/cache",
    )
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_home = Path(args.data_home)
    data_home.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    x, y = load_nid_dataset(
        max_samples=args.max_samples,
        random_state=args.random_state,
        data_home=data_home,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in x_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )

    print("Preprocessing features...")
    x_train_prepared = preprocessor.fit_transform(x_train)
    x_test_prepared = preprocessor.transform(x_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_prepared)
    x_test_scaled = scaler.transform(x_test_prepared)

    before_preview = x_train.iloc[:8, :12].copy()
    save_table_image(
        before_preview,
        f"Before Dimensionality Reduction (raw features shape: {x_train.shape})",
        "dataset_before_dr_preview.png",
        output_dir,
    )

    model_defs = {
        "LogisticRegression": LogisticRegression(max_iter=1200, random_state=args.random_state),
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=args.random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
    }

    print("Training baseline models (without dimensionality reduction)...")
    baseline_results = run_benchmark(
        model_defs,
        "No DR",
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
    )

    print("Applying PCA...")
    pca = PCA(n_components=0.95, random_state=args.random_state)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    after_preview = pd.DataFrame(x_train_pca).iloc[:8, :12]
    after_preview.columns = [f"pc{i+1}" for i in range(after_preview.shape[1])]
    save_table_image(
        after_preview.round(3),
        f"After PCA Dimensionality Reduction (shape: {x_train_pca.shape})",
        "dataset_after_dr_preview.png",
        output_dir,
    )

    print("Training models on PCA-reduced features...")
    pca_results = run_benchmark(
        model_defs,
        "PCA",
        x_train_pca,
        y_train,
        x_test_pca,
        y_test,
    )

    all_results = baseline_results + pca_results
    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in {"y_pred", "fitted_model"}} for r in all_results]
    )
    summary = summary.sort_values(
        by=["f1", "accuracy", "roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary.to_csv(output_dir / "metrics_comparison.csv", index=False)

    best_no_dr = max(baseline_results, key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]))
    best_pca = max(pca_results, key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]))
    best_overall = max(all_results, key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]))
    best_overall_mode = "PCA" if "(PCA)" in best_overall["model"] else "No DR"

    experiment_meta = {
        "dataset": "KDDCup99 (subset=SA, percent10=True)",
        "task": "Binary classification (normal vs intrusion)",
        "samples_used": int(len(x)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "original_feature_count": int(x_train_scaled.shape[1]),
        "pca_components_retained": int(x_train_pca.shape[1]),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "dimension_reduction_ratio": float(1 - (x_train_pca.shape[1] / x_train_scaled.shape[1])),
        "best_model_no_dr": best_no_dr["model"],
        "best_model_pca": best_pca["model"],
        "best_model_overall": best_overall["model"],
        "best_model_overall_mode": best_overall_mode,
    }

    with (output_dir / "experiment_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(experiment_meta, fp, indent=2)

    inference_bundle = {
        "feature_columns": list(x.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "pca": pca,
        "best_no_dr_model_name": best_no_dr["model"],
        "best_pca_model_name": best_pca["model"],
        "best_overall_model_name": best_overall["model"],
        "best_overall_mode": best_overall_mode,
        "best_no_dr_model": best_no_dr["fitted_model"],
        "best_pca_model": best_pca["fitted_model"],
    }
    joblib.dump(inference_bundle, output_dir / "nid_inference_bundle.joblib")

    plot_explained_variance(pca, output_dir)
    plot_confusion_matrix(
        y_test,
        best_no_dr["y_pred"],
        f"Confusion Matrix - Best No DR ({best_no_dr['model']})",
        "confusion_matrix_baseline.png",
        output_dir,
    )
    plot_confusion_matrix(
        y_test,
        best_pca["y_pred"],
        f"Confusion Matrix - Best PCA ({best_pca['model']})",
        "confusion_matrix_pca.png",
        output_dir,
    )

    print("\n=== Experiment Summary ===")
    print(summary.to_string(index=False))
    print("\nArtifacts written to:")
    print(f"- {output_dir / 'metrics_comparison.csv'}")
    print(f"- {output_dir / 'experiment_meta.json'}")
    print(f"- {output_dir / 'pca_explained_variance.png'}")
    print(f"- {output_dir / 'confusion_matrix_baseline.png'}")
    print(f"- {output_dir / 'confusion_matrix_pca.png'}")
    print(f"- {output_dir / 'dataset_before_dr_preview.png'}")
    print(f"- {output_dir / 'dataset_after_dr_preview.png'}")
    print(f"- {output_dir / 'nid_inference_bundle.joblib'}")


if __name__ == "__main__":
    main()
