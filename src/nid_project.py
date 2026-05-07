import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier


try:
    from mlxtend.feature_selection import SequentialFeatureSelector as MlxtendSFS
except ImportError as exc:
    raise ImportError("mlxtend is required. Install with: pip install mlxtend") from exc

try:
    import shap
except ImportError as exc:
    raise ImportError("shap is required. Install with: pip install shap") from exc


TECHNIQUE_NAMES = {
    "no_dr": "No DR",
    "corr_fs": "Correlation FS",
    "pca": "PCA",
    "lda": "LDA",
    "svd": "SVD",
    "filter_kbest": "Filter (SelectKBest)",
    "wrapper_rfe": "Wrapper (RFE)",
    "sfs": "SFS",
    "sbs": "SBS",
    "sffs": "SFFS",
    "sfbs": "SFBS",
}


def _decode_if_bytes(value):
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def load_nid_dataset(max_samples: int, random_state: int, data_home: Path) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    dataset = fetch_kddcup99(
        subset="SA",
        percent10=True,
        shuffle=True,
        random_state=random_state,
        as_frame=True,
        data_home=str(data_home),
    )

    x = dataset.data.copy()
    y_raw = dataset.target.copy().map(_decode_if_bytes)

    for column in x.columns:
        if x[column].dtype == object:
            x[column] = x[column].map(_decode_if_bytes)

    y = (y_raw != "normal.").astype(int).to_numpy()

    if max_samples and max_samples < len(x):
        x, _, y, _, y_raw, _ = train_test_split(
            x,
            y,
            y_raw,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )

    return x, y, y_raw


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_start

    pred_start = time.perf_counter()
    y_pred = model.predict(x_test)
    pred_seconds = time.perf_counter() - pred_start

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(x_test)
        y_prob = 1 / (1 + np.exp(-raw))
    else:
        y_prob = np.full(len(y_pred), 0.5)

    roc_auc = roc_auc_score(y_test, y_prob)
    confidence = np.maximum(y_prob, 1 - y_prob)

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
        "y_prob": y_prob,
        "confidence": confidence,
        "fitted_model": model,
    }


def run_models_for_technique(
    technique_key: str,
    x_train,
    y_train,
    x_test,
    y_test,
    model_defs: dict,
) -> list[dict]:
    results = []
    for model_short, model in model_defs.items():
        full_name = f"{model_short} ({TECHNIQUE_NAMES[technique_key]})"
        print(f"Training {full_name}...")
        res = evaluate_model(full_name, clone(model), x_train, y_train, x_test, y_test)
        res["technique_key"] = technique_key
        res["technique"] = TECHNIQUE_NAMES[technique_key]
        res["model_short"] = model_short
        results.append(res)
    return results


def save_table_image(df: pd.DataFrame, title: str, filename: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str, output_dir: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_pca_explained_variance(pca: PCA, output_dir: Path) -> None:
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


def plot_pca_2d_projection(x_scaled, y, output_dir: Path, random_state: int) -> None:
    pca2 = PCA(n_components=2, random_state=random_state)
    x2 = pca2.fit_transform(x_scaled)
    df = pd.DataFrame({"pc1": x2[:, 0], "pc2": x2[:, 1], "class": y})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="class", palette={0: "#1f77b4", 1: "#d62728"}, alpha=0.7)
    plt.title("PCA 2D Projection (0=Normal, 1=Intrusion)")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_2d_projection.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(x_scaled, feature_names: list[str], output_dir: Path) -> None:
    show_n = min(25, len(feature_names))
    df = pd.DataFrame(x_scaled[:, :show_n], columns=feature_names[:show_n])
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap (First {show_n} Preprocessed Features)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_feature_distribution(x_train_raw: pd.DataFrame, y_train: np.ndarray, output_dir: Path) -> None:
    numeric_cols = x_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return

    diffs = []
    for col in numeric_cols:
        m0 = x_train_raw.loc[y_train == 0, col].mean()
        m1 = x_train_raw.loc[y_train == 1, col].mean()
        diffs.append((col, abs(m1 - m0)))

    top_cols = [c for c, _ in sorted(diffs, key=lambda t: t[1], reverse=True)[:4]]
    plot_df = x_train_raw[top_cols].copy()
    plot_df["class"] = y_train

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_cols):
        sns.histplot(
            data=plot_df,
            x=col,
            hue="class",
            stat="density",
            common_norm=False,
            element="step",
            ax=axes[i],
            bins=30,
        )
        axes[i].set_title(f"Feature Distribution: {col}")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distribution.png", dpi=150)
    plt.close()


def plot_model_performance_heatmap(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = metrics_df.pivot_table(index="technique", columns="model_short", values="f1", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Model Performance Heatmap (F1 Score)")
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance_heatmap.png", dpi=150)
    plt.close()


def plot_filter_methods_accuracy_comparison(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    filter_keys = ["corr_fs", "pca", "lda", "svd"]
    model_order = ["RandomForest", "DecisionTree", "SVM-RBF", "GaussianNB", "KNN"]
    method_labels = {
        "corr_fs": "Correlation",
        "pca": "PCA",
        "lda": "LDA",
        "svd": "SVD",
    }

    plot_df = metrics_df[metrics_df["technique_key"].isin(filter_keys)].copy()
    if plot_df.empty:
        return

    plot_df["method"] = plot_df["technique_key"].map(method_labels)
    plot_df = plot_df[plot_df["model_short"].isin(model_order)]

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="method",
        y="accuracy",
        hue="model_short",
        order=[method_labels[k] for k in filter_keys],
        hue_order=model_order,
    )
    plt.ylim(0, 1.02)
    plt.title("Filter Methods - Accuracy Comparison")
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_dir / "filter_methods_accuracy_comparison.png", dpi=150)
    plt.close()


def build_technique_datasets(
    x_train_scaled,
    x_test_scaled,
    y_train,
    preprocessed_feature_names: list[str],
    random_state: int,
    selector_k: int,
) -> dict:
    techniques = {}

    techniques["no_dr"] = {
        "kind": "identity",
        "x_train": x_train_scaled,
        "x_test": x_test_scaled,
        "feature_names": preprocessed_feature_names,
    }

    corr_df = pd.DataFrame(x_train_scaled, columns=preprocessed_feature_names).corr().abs()
    upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    corr_selected_names = [c for c in preprocessed_feature_names if c not in to_drop]
    corr_selected_idx = [preprocessed_feature_names.index(c) for c in corr_selected_names]
    x_train_corr = x_train_scaled[:, corr_selected_idx]
    x_test_corr = x_test_scaled[:, corr_selected_idx]
    techniques["corr_fs"] = {
        "kind": "index_select",
        "selected_idx": corr_selected_idx,
        "x_train": x_train_corr,
        "x_test": x_test_corr,
        "feature_names": corr_selected_names,
    }

    pca = PCA(n_components=0.95, random_state=random_state)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    techniques["pca"] = {
        "kind": "pca",
        "transformer": pca,
        "x_train": x_train_pca,
        "x_test": x_test_pca,
        "feature_names": [f"pc{i+1}" for i in range(x_train_pca.shape[1])],
    }

    lda = LinearDiscriminantAnalysis(n_components=1)
    x_train_lda = lda.fit_transform(x_train_scaled, y_train)
    x_test_lda = lda.transform(x_test_scaled)
    techniques["lda"] = {
        "kind": "lda",
        "transformer": lda,
        "x_train": x_train_lda,
        "x_test": x_test_lda,
        "feature_names": ["lda1"],
    }

    svd_components = min(selector_k, x_train_scaled.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    x_train_svd = svd.fit_transform(x_train_scaled)
    x_test_svd = svd.transform(x_test_scaled)
    techniques["svd"] = {
        "kind": "svd",
        "transformer": svd,
        "x_train": x_train_svd,
        "x_test": x_test_svd,
        "feature_names": [f"svd{i+1}" for i in range(x_train_svd.shape[1])],
    }

    selector_filter = SelectKBest(score_func=mutual_info_classif, k=selector_k)
    x_train_filter = selector_filter.fit_transform(x_train_scaled, y_train)
    x_test_filter = selector_filter.transform(x_test_scaled)
    filter_mask = selector_filter.get_support()
    techniques["filter_kbest"] = {
        "kind": "selector",
        "selector": selector_filter,
        "x_train": x_train_filter,
        "x_test": x_test_filter,
        "feature_names": [n for n, keep in zip(preprocessed_feature_names, filter_mask) if keep],
    }

    rfe = RFE(
        estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        n_features_to_select=selector_k,
        step=1,
    )
    x_train_rfe = rfe.fit_transform(x_train_scaled, y_train)
    x_test_rfe = rfe.transform(x_test_scaled)
    rfe_mask = rfe.get_support()
    techniques["wrapper_rfe"] = {
        "kind": "selector",
        "selector": rfe,
        "x_train": x_train_rfe,
        "x_test": x_test_rfe,
        "feature_names": [n for n, keep in zip(preprocessed_feature_names, rfe_mask) if keep],
    }

    sfs = MlxtendSFS(
        LogisticRegression(max_iter=1000, random_state=random_state),
        k_features=selector_k,
        forward=True,
        floating=False,
        scoring="f1",
        cv=3,
        n_jobs=1,
    )
    sfs.fit(x_train_scaled, y_train)
    x_train_sfs = sfs.transform(x_train_scaled)
    x_test_sfs = sfs.transform(x_test_scaled)
    sfs_idx = list(sfs.k_feature_idx_)
    techniques["sfs"] = {
        "kind": "selector",
        "selector": sfs,
        "x_train": x_train_sfs,
        "x_test": x_test_sfs,
        "feature_names": [preprocessed_feature_names[i] for i in sfs_idx],
    }

    sbs = MlxtendSFS(
        LogisticRegression(max_iter=1000, random_state=random_state),
        k_features=selector_k,
        forward=False,
        floating=False,
        scoring="f1",
        cv=3,
        n_jobs=1,
    )
    sbs.fit(x_train_scaled, y_train)
    x_train_sbs = sbs.transform(x_train_scaled)
    x_test_sbs = sbs.transform(x_test_scaled)
    sbs_idx = list(sbs.k_feature_idx_)
    techniques["sbs"] = {
        "kind": "selector",
        "selector": sbs,
        "x_train": x_train_sbs,
        "x_test": x_test_sbs,
        "feature_names": [preprocessed_feature_names[i] for i in sbs_idx],
    }

    sffs = MlxtendSFS(
        LogisticRegression(max_iter=1000, random_state=random_state),
        k_features=selector_k,
        forward=True,
        floating=True,
        scoring="f1",
        cv=3,
        n_jobs=1,
    )
    sffs.fit(x_train_scaled, y_train)
    x_train_sffs = sffs.transform(x_train_scaled)
    x_test_sffs = sffs.transform(x_test_scaled)
    sffs_idx = list(sffs.k_feature_idx_)
    techniques["sffs"] = {
        "kind": "selector",
        "selector": sffs,
        "x_train": x_train_sffs,
        "x_test": x_test_sffs,
        "feature_names": [preprocessed_feature_names[i] for i in sffs_idx],
    }

    sfbs = MlxtendSFS(
        LogisticRegression(max_iter=1000, random_state=random_state),
        k_features=selector_k,
        forward=False,
        floating=True,
        scoring="f1",
        cv=3,
        n_jobs=1,
    )
    sfbs.fit(x_train_scaled, y_train)
    x_train_sfbs = sfbs.transform(x_train_scaled)
    x_test_sfbs = sfbs.transform(x_test_scaled)
    sfbs_idx = list(sfbs.k_feature_idx_)
    techniques["sfbs"] = {
        "kind": "selector",
        "selector": sfbs,
        "x_train": x_train_sfbs,
        "x_test": x_test_sfbs,
        "feature_names": [preprocessed_feature_names[i] for i in sfbs_idx],
    }

    return techniques, pca


def save_feature_importance(
    best_result: dict,
    x_test_transformed,
    y_test,
    feature_names: list[str],
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    model = best_result["fitted_model"]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importance = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        perm = permutation_importance(
            model,
            x_test_transformed,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=random_state,
            n_jobs=1,
        )
        importance = perm.importances_mean

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    top = imp_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"], color="#2a9d8f")
    plt.title("Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    return imp_df


def save_shap_waterfall(
    rf_result: dict,
    x_test_transformed,
    feature_names: list[str],
    output_dir: Path,
    random_state: int,
) -> tuple[int, float, float, int]:
    x_df = pd.DataFrame(x_test_transformed, columns=feature_names)
    rng = np.random.default_rng(random_state)
    sample_idx = int(rng.integers(0, len(x_df)))

    model = rf_result["fitted_model"]
    explainer = shap.Explainer(model, x_df)
    explanation = explainer(x_df.iloc[[sample_idx]])

    if explanation.values.ndim == 3:
        one = shap.Explanation(
            values=explanation.values[0, :, 1],
            base_values=explanation.base_values[0, 1],
            data=explanation.data[0],
            feature_names=feature_names,
        )
    else:
        one = explanation[0]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(one, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=150)
    plt.close()

    prob = rf_result["y_prob"]
    pred = rf_result["y_pred"]
    confidence = np.maximum(prob, 1 - prob)
    return sample_idx, float(prob[sample_idx]), float(confidence[sample_idx]), int(pred[sample_idx])


def save_prediction_confidence_plot(best_result: dict, output_dir: Path) -> None:
    conf = best_result["confidence"]
    plt.figure(figsize=(8, 5))
    sns.histplot(conf, bins=30, kde=True, color="#457b9d")
    plt.title("Prediction Confidence Distribution (Best Model)")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_confidence_distribution.png", dpi=150)
    plt.close()


def transform_with_technique(artifact: dict, x_scaled):
    if artifact["kind"] == "identity":
        return x_scaled
    if artifact["kind"] in {"pca", "lda", "svd"}:
        return artifact["transformer"].transform(x_scaled)
    if artifact["kind"] == "index_select":
        return x_scaled[:, artifact["selected_idx"]]
    return artifact["selector"].transform(x_scaled)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NID project with multiple dimensionality reduction techniques and model comparison"
    )
    parser.add_argument("--max-samples", type=int, default=5000, help="Max rows to use")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-home",
        type=str,
        default="data_cache",
        help="Local directory for dataset download/cache",
    )
    parser.add_argument(
        "--selector-k",
        type=int,
        default=20,
        help="Features to keep for filter/wrapper/floating selection methods",
    )
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_home = Path(args.data_home)
    data_home.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    x, y, y_raw = load_nid_dataset(args.max_samples, args.random_state, data_home)

    x_train, x_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        x,
        y,
        y_raw,
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
    preprocessed_feature_names = categorical_cols + numeric_cols

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_prepared)
    x_test_scaled = scaler.transform(x_test_prepared)

    # Data preview artifacts
    before_preview = x_train.iloc[:8, :12].copy()
    save_table_image(
        before_preview,
        f"Before Dimensionality Reduction (raw features shape: {x_train.shape})",
        "dataset_before_dr_preview.png",
        output_dir,
    )

    # Technique construction
    selector_k = min(args.selector_k, x_train_scaled.shape[1] - 1)
    print("Building dimensionality reduction / feature selection techniques...")
    techniques, pca_for_variance = build_technique_datasets(
        x_train_scaled,
        x_test_scaled,
        y_train,
        preprocessed_feature_names,
        args.random_state,
        selector_k,
    )

    after_preview = pd.DataFrame(techniques["pca"]["x_train"]).iloc[:8, :12]
    after_preview.columns = [f"pc{i+1}" for i in range(after_preview.shape[1])]
    save_table_image(
        after_preview.round(3),
        f"After PCA Dimensionality Reduction (shape: {techniques['pca']['x_train'].shape})",
        "dataset_after_dr_preview.png",
        output_dir,
    )

    model_defs = {
        "LogisticRegression": LogisticRegression(max_iter=1200, random_state=args.random_state),
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=args.random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "DecisionTree": DecisionTreeClassifier(random_state=args.random_state),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
    }

    all_results = []
    for technique_key, t_data in techniques.items():
        print(f"Running models for technique: {TECHNIQUE_NAMES[technique_key]}")
        results = run_models_for_technique(
            technique_key,
            t_data["x_train"],
            y_train,
            t_data["x_test"],
            y_test,
            model_defs,
        )
        all_results.extend(results)

    metrics_df = pd.DataFrame(
        [
            {
                "technique_key": r["technique_key"],
                "technique": r["technique"],
                "model_short": r["model_short"],
                "model": r["model"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "train_seconds": r["train_seconds"],
                "predict_seconds": r["predict_seconds"],
            }
            for r in all_results
        ]
    ).sort_values(by=["f1", "accuracy", "roc_auc"], ascending=[False, False, False])
    metrics_df.to_csv(output_dir / "metrics_comparison.csv", index=False)

    technique_summary = (
        metrics_df.sort_values(by=["f1", "accuracy", "roc_auc"], ascending=[False, False, False])
        .groupby("technique", as_index=False)
        .first()
    )
    technique_summary.to_csv(output_dir / "technique_summary.csv", index=False)

    best_overall = max(all_results, key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]))
    best_per_technique = {}
    for tk in techniques.keys():
        filtered = [r for r in all_results if r["technique_key"] == tk]
        best_per_technique[tk] = max(filtered, key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]))

    best_rf = max(
        [r for r in all_results if r["model_short"] == "RandomForest"],
        key=lambda r: (r["f1"], r["accuracy"], r["roc_auc"]),
    )

    # Save plots
    plot_pca_explained_variance(pca_for_variance, output_dir)
    plot_pca_2d_projection(x_train_scaled, y_train, output_dir, args.random_state)
    plot_correlation_heatmap(x_train_scaled, preprocessed_feature_names, output_dir)
    plot_feature_distribution(x_train, y_train, output_dir)
    plot_model_performance_heatmap(metrics_df, output_dir)
    plot_filter_methods_accuracy_comparison(metrics_df, output_dir)

    plot_confusion_matrix(
        y_test,
        best_overall["y_pred"],
        f"Confusion Matrix - Best Overall ({best_overall['model']})",
        "confusion_matrix_best_overall.png",
        output_dir,
    )

    plot_confusion_matrix(
        y_test,
        best_per_technique["no_dr"]["y_pred"],
        f"Confusion Matrix - Best No DR ({best_per_technique['no_dr']['model']})",
        "confusion_matrix_baseline.png",
        output_dir,
    )

    plot_confusion_matrix(
        y_test,
        best_per_technique["pca"]["y_pred"],
        f"Confusion Matrix - Best PCA ({best_per_technique['pca']['model']})",
        "confusion_matrix_pca.png",
        output_dir,
    )

    best_technique_artifact = techniques[best_overall["technique_key"]]
    save_feature_importance(
        best_overall,
        best_technique_artifact["x_test"],
        y_test,
        best_technique_artifact["feature_names"],
        output_dir,
        args.random_state,
    )

    sample_idx, sample_prob, sample_conf, sample_pred = save_shap_waterfall(
        best_rf,
        techniques[best_rf["technique_key"]]["x_test"],
        techniques[best_rf["technique_key"]]["feature_names"],
        output_dir,
        args.random_state,
    )

    save_prediction_confidence_plot(best_overall, output_dir)

    random_sample = x_test.iloc[[sample_idx]].copy()
    random_sample["true_binary"] = y_test[sample_idx]
    random_sample["true_label"] = "intrusion" if y_test[sample_idx] == 1 else "normal"
    random_sample.to_csv(output_dir / "random_sample_loaded.csv", index=False)

    sample_meta = {
        "sample_index_loaded": sample_idx,
        "prediction_model": best_rf["model"],
        "prediction_technique": TECHNIQUE_NAMES[best_rf["technique_key"]],
        "predicted_binary": int(sample_pred),
        "predicted_class": "intrusion" if sample_pred == 1 else "normal",
        "predicted_probability_intrusion": sample_prob,
        "prediction_confidence": sample_conf,
        "true_binary": int(y_test[sample_idx]),
        "true_class": "intrusion" if y_test[sample_idx] == 1 else "normal",
    }
    with (output_dir / "random_sample_prediction.json").open("w", encoding="utf-8") as fp:
        json.dump(sample_meta, fp, indent=2)

    observations = [
        "Final observations:",
        f"- Best overall model: {best_overall['model']}.",
        f"- Best technique: {TECHNIQUE_NAMES[best_overall['technique_key']]}.",
        f"- PCA reduced features from {x_train_scaled.shape[1]} to {techniques['pca']['x_train'].shape[1]}.",
        "- Feature selection techniques (Filter/RFE/SFFS/SFBS) were benchmarked in the same pipeline.",
        "- See model_performance_heatmap.png and metrics_comparison.csv for final model comparison.",
    ]
    (output_dir / "final_observations.txt").write_text("\n".join(observations), encoding="utf-8")

    experiment_meta = {
        "dataset": "KDDCup99 (subset=SA, percent10=True)",
        "task": "Binary classification (normal vs intrusion)",
        "samples_used": int(len(x)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "original_feature_count": int(x_train_scaled.shape[1]),
        "pca_components_retained": int(techniques["pca"]["x_train"].shape[1]),
        "pca_explained_variance": float(np.sum(pca_for_variance.explained_variance_ratio_)),
        "dimension_reduction_ratio": float(
            1 - (techniques["pca"]["x_train"].shape[1] / x_train_scaled.shape[1])
        ),
        "best_model_overall": best_overall["model"],
        "best_technique_overall": TECHNIQUE_NAMES[best_overall["technique_key"]],
        "best_model_per_technique": {
            TECHNIQUE_NAMES[k]: best_per_technique[k]["model"] for k in best_per_technique
        },
        "selector_k": selector_k,
    }
    with (output_dir / "experiment_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(experiment_meta, fp, indent=2)

    # Inference bundle
    inference_bundle = {
        "feature_columns": list(x.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "techniques": {
            k: {
                "kind": v["kind"],
                "feature_names": v["feature_names"],
                "transformer": v.get("transformer"),
                "selector": v.get("selector"),
                "selected_idx": v.get("selected_idx"),
            }
            for k, v in techniques.items()
        },
        "technique_name_map": TECHNIQUE_NAMES,
        "best_overall_model_name": best_overall["model"],
        "best_overall_technique_key": best_overall["technique_key"],
        "best_overall_model": best_overall["fitted_model"],
        "best_models_per_technique": {
            k: best_per_technique[k]["fitted_model"] for k in best_per_technique
        },
        "best_model_names_per_technique": {
            k: best_per_technique[k]["model"] for k in best_per_technique
        },
    }
    joblib.dump(inference_bundle, output_dir / "nid_inference_bundle.joblib")

    # Dataset exports
    raw_with_label = x.copy()
    raw_with_label["label_raw"] = y_raw
    raw_with_label["target_binary"] = y
    raw_with_label.head(500).to_csv(output_dir / "raw_dataset_sample.csv", index=False)

    preprocessed_df = pd.DataFrame(x_train_prepared, columns=preprocessed_feature_names)
    preprocessed_df["target_binary"] = y_train
    preprocessed_df.head(500).to_csv(output_dir / "preprocessed_dataset_sample.csv", index=False)

    pca_df = pd.DataFrame(techniques["pca"]["x_train"], columns=techniques["pca"]["feature_names"])
    pca_df["target_binary"] = y_train
    pca_df.head(500).to_csv(output_dir / "pca_dataset_sample.csv", index=False)

    (output_dir / "feature_columns.txt").write_text("\n".join(x.columns), encoding="utf-8")
    pd.DataFrame(columns=x.columns).to_csv(output_dir / "nid_input_template.csv", index=False)

    print("\n=== Final Best ===")
    print(f"Best overall model: {best_overall['model']}")
    print(f"Best overall technique: {TECHNIQUE_NAMES[best_overall['technique_key']]}")
    print("\nArtifacts written to outputs/ (metrics, plots, bundle, and report evidence files).")


if __name__ == "__main__":
    main()
