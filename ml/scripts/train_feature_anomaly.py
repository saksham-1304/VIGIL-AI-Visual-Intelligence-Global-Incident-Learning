from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml.anomaly.feature_anomaly import save_iforest, train_isolation_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feature-based anomaly detector")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--contamination", type=float, default=0.03)
    return parser.parse_args()


def maybe_log_mlflow(params: dict, metrics: dict, artifact_path: Path) -> None:
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment("incident-intel-anomaly")
        with mlflow.start_run(run_name="isolation-forest"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path))
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


def main() -> None:
    args = parse_args()
    features_df = pd.read_csv(args.features)

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {"is_anomaly", "source"} and np.issubdtype(features_df[col].dtype, np.number)
    ]
    x = features_df[feature_cols].astype(float).to_numpy()

    model = train_isolation_forest(
        x,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_iforest(model, str(output_path))

    metrics = {
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    maybe_log_mlflow(
        params=metrics,
        metrics={"trained": 1.0},
        artifact_path=metrics_path,
    )

    print(f"Isolation Forest saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
