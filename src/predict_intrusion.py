import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def transform_by_technique_artifact(artifact: dict, x_scaled):
    kind = artifact.get("kind")
    if kind == "identity":
        return x_scaled
    if kind in {"pca", "lda", "svd"}:
        return artifact["transformer"].transform(x_scaled)
    if kind == "index_select":
        return x_scaled[:, artifact["selected_idx"]]
    return artifact["selector"].transform(x_scaled)


def predict(df: pd.DataFrame, bundle: dict, technique_key: str, use_best_overall: bool) -> tuple[pd.DataFrame, str, str]:
    if df.empty:
        raise ValueError("Input CSV has no rows. Add at least 1 data row under required columns.")

    required_cols = bundle["feature_columns"]
    missing = [c for c in required_cols if c not in df.columns]
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
        score = model.predict_proba(x_final)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(x_final)
        score = 1 / (1 + np.exp(-raw))
    else:
        score = np.full(len(y_pred), np.nan)

    out = df.copy()
    out["predicted_class"] = np.where(y_pred == 1, "intrusion", "normal")
    out["predicted_binary"] = y_pred
    out["intrusion_score"] = score
    out["prediction_confidence"] = np.maximum(score, 1 - score)
    return out, model_name, technique_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict network intrusion on new rows")
    parser.add_argument("--input", required=True, help="Input CSV with KDD features")
    parser.add_argument("--output", default="outputs/nid_predictions.csv", help="Output CSV path")
    parser.add_argument("--bundle", default="outputs/nid_inference_bundle.joblib", help="Saved artifact path")
    parser.add_argument(
        "--mode",
        choices=["auto", "best_overall", "per_technique"],
        default="auto",
        help="Prediction mode",
    )
    parser.add_argument(
        "--technique",
        default="no_dr",
        help="Technique key when mode=per_technique. Ex: no_dr, corr_fs, pca, lda, svd, filter_kbest, wrapper_rfe, sffs, sfbs",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError("Inference bundle not found. Run src/nid_project.py first.")

    bundle = joblib.load(bundle_path)
    input_df = pd.read_csv(args.input)

    if args.mode in {"auto", "best_overall"}:
        use_best_overall = True
        technique_key = bundle.get("best_overall_technique_key", "no_dr")
    else:
        use_best_overall = False
        technique_key = args.technique

    if technique_key not in bundle.get("techniques", {}):
        available = ", ".join(bundle.get("techniques", {}).keys())
        raise ValueError(f"Unknown technique '{technique_key}'. Available: {available}")

    pred_df, model_name, used_technique = predict(input_df, bundle, technique_key, use_best_overall)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    intrusion_count = int((pred_df["predicted_binary"] == 1).sum())
    print(f"Technique used: {used_technique}")
    print(f"Model used: {model_name}")
    print(f"Rows scored: {len(pred_df)}")
    print(f"Predicted intrusion rows: {intrusion_count}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
