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
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--multi-gpu", action="store_true", help="Use all visible GPUs via DataParallel")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory to store epoch checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint dir")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Training heartbeat interval")
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

    gpu_count = 0
    try:
        import torch

        gpu_count = torch.cuda.device_count() if resolved_device.startswith("cuda") else 0
    except Exception:
        gpu_count = 0

    checkpoint_dir = args.checkpoint_dir.strip() or None
    multi_gpu_used = bool(args.multi_gpu and resolved_device.startswith("cuda") and gpu_count > 1)

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {"is_anomaly", "source"} and np.issubdtype(features_df[col].dtype, np.number)
    ]
    x = features_df[feature_cols].astype(float).to_numpy()

    print(
        f"Starting autoencoder training on {len(x)} samples with {len(feature_cols)} features "
        f"| epochs={args.epochs} batch_size={args.batch_size} device={resolved_device}",
        flush=True,
    )

    model, history = train_autoencoder(
        x,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=resolved_device,
        use_amp=args.amp,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        multi_gpu=args.multi_gpu,
        heartbeat_seconds=max(0, args.heartbeat_seconds),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_autoencoder(model, str(output_path), input_dim=x.shape[1], latent_dim=args.latent_dim)

    metrics = {
        "final_loss": float(history.losses[-1]),
        "min_loss": float(min(history.losses)),
        "epochs": args.epochs,
        "device": resolved_device,
        "gpu_count": gpu_count,
        "multi_gpu_used": int(multi_gpu_used),
        "amp_enabled": int(args.amp and resolved_device.startswith("cuda")),
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
            "gpu_count": gpu_count,
            "multi_gpu_requested": args.multi_gpu,
            "multi_gpu_used": multi_gpu_used,
            "amp": args.amp,
            "num_workers": args.num_workers,
            "checkpoint_dir": checkpoint_dir or "",
            "checkpoint_interval": args.checkpoint_interval,
            "resume": args.resume,
        },
        metrics=metrics,
        artifact_path=metrics_path,
    )

    print(f"Autoencoder saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Training device: {resolved_device}")
    print(f"Visible GPU count: {gpu_count}")
    print(f"Multi-GPU used: {multi_gpu_used}")


if __name__ == "__main__":
    main()
