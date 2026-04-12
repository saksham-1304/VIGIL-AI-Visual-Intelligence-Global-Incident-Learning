from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ml.anomaly.autoencoder import load_autoencoder, reconstruction_error
from ml.anomaly.feature_anomaly import anomaly_scores, load_iforest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly models")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--ae", type=str, required=True)
    parser.add_argument("--iforest", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
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


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    df = pd.read_csv(args.features)
    feature_cols = [
        col
        for col in df.columns
        if col not in {"is_anomaly", "source"} and np.issubdtype(df[col].dtype, np.number)
    ]

    x = df[feature_cols].astype(float).to_numpy()
    y_true = df["is_anomaly"].astype(int).to_numpy() if "is_anomaly" in df.columns else np.zeros(len(df), dtype=int)

    ae_model = load_autoencoder(args.ae, device=resolved_device)
    ae_error = reconstruction_error(ae_model, x, device=resolved_device)
    ae_score = np.clip(ae_error * 8.0, 0, 1)

    iforest_model = load_iforest(args.iforest)
    iforest_score = anomaly_scores(iforest_model, x)

    hybrid_score = 0.5 * ae_score + 0.5 * iforest_score

    report = {
        "dataset": {
            "rows": int(len(df)),
            "features": feature_cols,
            "anomaly_ratio": float(np.mean(y_true)) if len(y_true) else 0.0,
        },
        "runtime": {"device": resolved_device},
        "models": {
            "autoencoder": classification_metrics(y_true, ae_score, threshold=0.55),
            "isolation_forest": classification_metrics(y_true, iforest_score, threshold=0.5),
            "hybrid": classification_metrics(y_true, hybrid_score, threshold=0.5),
        },
    }

    best_model = max(report["models"].items(), key=lambda item: item[1]["f1"])
    report["best_model"] = {"name": best_model[0], "f1": best_model[1]["f1"]}

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluation report written to {report_path}")


if __name__ == "__main__":
    main()
