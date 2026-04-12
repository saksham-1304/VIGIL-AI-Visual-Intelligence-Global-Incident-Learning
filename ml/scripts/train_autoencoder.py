from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml.anomaly.autoencoder import save_autoencoder, train_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reconstruction-based anomaly detector")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device.lower() != "auto":
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def maybe_log_mlflow(params: dict, metrics: dict, artifact_path: Path) -> None:
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment("incident-intel-anomaly")
        with mlflow.start_run(run_name="autoencoder"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path))
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    features_df = pd.read_csv(args.features)

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {"is_anomaly", "source"} and np.issubdtype(features_df[col].dtype, np.number)
    ]
    x = features_df[feature_cols].astype(float).to_numpy()

    model, history = train_autoencoder(
        x,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=resolved_device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_autoencoder(model, str(output_path), input_dim=x.shape[1], latent_dim=args.latent_dim)

    metrics = {
        "final_loss": float(history.losses[-1]),
        "min_loss": float(min(history.losses)),
        "epochs": args.epochs,
        "device": resolved_device,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    maybe_log_mlflow(
        params={
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "device": resolved_device,
        },
        metrics=metrics,
        artifact_path=metrics_path,
    )

    print(f"Autoencoder saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Training device: {resolved_device}")


if __name__ == "__main__":
    main()
