import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict(df: pd.DataFrame, bundle: dict, mode: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "Input CSV has no rows. Add at least 1 data row under the required feature columns."
        )

    required_cols = bundle["feature_columns"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:8])}")

    x = df[required_cols].copy()
    x_prepared = bundle["preprocessor"].transform(x)
    x_scaled = bundle["scaler"].transform(x_prepared)

    if mode == "no_dr":
        model = bundle["best_no_dr_model"]
        x_final = x_scaled
    else:
        model = bundle["best_pca_model"]
        x_final = bundle["pca"].transform(x_scaled)

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
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict network intrusion on new rows")
    parser.add_argument("--input", required=True, help="Input CSV with KDD features")
    parser.add_argument("--output", default="outputs/nid_predictions.csv", help="Output CSV path")
    parser.add_argument(
        "--bundle",
        default="outputs/nid_inference_bundle.joblib",
        help="Path to saved training artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "no_dr", "pca"],
        default="auto",
        help="Prediction path",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError("Inference bundle not found. Run src/nid_project.py first.")

    bundle = joblib.load(bundle_path)
    input_df = pd.read_csv(args.input)

    mode = args.mode
    if mode == "auto":
        mode = "pca" if bundle.get("best_overall_mode") == "PCA" else "no_dr"

    pred_df = predict(input_df, bundle, mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    intrusion_count = int((pred_df["predicted_binary"] == 1).sum())
    print(f"Mode used: {mode}")
    print(f"Rows scored: {len(pred_df)}")
    print(f"Predicted intrusion rows: {intrusion_count}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
