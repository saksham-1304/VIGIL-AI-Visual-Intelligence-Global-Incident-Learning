from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from ml.anomaly.autoencoder import load_autoencoder, reconstruction_error
from ml.anomaly.feature_anomaly import anomaly_scores, load_iforest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate human feedback loop and threshold recalibration impact")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--ae", type=str, required=True)
    parser.add_argument("--iforest", type=str, required=True)
    parser.add_argument("--eval-report", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feedback-samples", type=int, default=800)
    parser.add_argument("--label-noise", type=float, default=0.03)
    parser.add_argument("--random-seed", type=int, default=42)
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


def sigmoid_score(decision: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(3.0 * decision))


def yolo_semantic_score(df: pd.DataFrame) -> np.ndarray:
    def series(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    linear = (
        0.55 * series("yolo_risk_count")
        + 0.15 * series("yolo_object_count")
        + 0.20 * series("yolo_area_ratio")
        + 0.30 * series("motion_score")
        + 0.35 * series("fall_flag")
        + 0.25 * series("wrong_way_flag")
        - 0.85
    )
    return np.clip((1.0 / (1.0 + np.exp(-linear))).to_numpy(dtype=float), 0.0, 1.0)


def threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 91)


def choose_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid():
        y_pred = (y_score >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    if len(np.unique(y_true)) >= 2:
        pr_auc = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def build_scores(df: pd.DataFrame, x: np.ndarray, ae_path: str, iforest_path: str, device: str) -> dict[str, np.ndarray]:
    ae_model = load_autoencoder(ae_path, device=device)
    ae_error = reconstruction_error(ae_model, x, device=device)
    ae_score = np.clip(ae_error * 8.0, 0.0, 1.0)

    iforest_model = load_iforest(iforest_path)
    iforest_score = anomaly_scores(iforest_model, x)

    hybrid = 0.5 * ae_score + 0.5 * iforest_score
    yolo = yolo_semantic_score(df)
    fusion = np.clip(0.4 * hybrid + 0.4 * yolo + 0.2 * ae_score, 0.0, 1.0)

    return {
        "autoencoder": ae_score,
        "isolation_forest": iforest_score,
        "hybrid": hybrid,
        "yolo_semantic": yolo,
        "hybrid_yolo_fusion": fusion,
    }


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.features)
    if "is_anomaly" not in df.columns:
        raise ValueError("features.csv must contain is_anomaly for feedback simulation")

    feature_cols = [
        col
        for col in df.columns
        if col not in {"is_anomaly", "source", "incident_class"} and np.issubdtype(df[col].dtype, np.number)
    ]

    x = df[feature_cols].astype(float).to_numpy()
    y_true = df["is_anomaly"].astype(int).to_numpy()

    if len(y_true) < 50:
        raise ValueError("Need at least 50 labeled samples for feedback simulation")

    device = resolve_device(args.device)
    scores = build_scores(df, x, ae_path=args.ae, iforest_path=args.iforest, device=device)

    eval_report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    selected_model = str(eval_report.get("best_model", {}).get("name", "hybrid"))
    if selected_model not in scores:
        selected_model = "hybrid"

    selected_score = scores[selected_model]
    base_threshold = float(eval_report.get("best_model", {}).get("threshold", 0.5))

    rng = np.random.default_rng(args.random_seed)
    indices = np.arange(len(y_true))
    rng.shuffle(indices)

    feedback_samples = int(np.clip(args.feedback_samples, 20, max(20, len(indices) - 20)))
    feedback_idx = indices[:feedback_samples]
    eval_idx = indices[feedback_samples:]

    feedback_labels = y_true[feedback_idx].copy()
    noise_rate = float(np.clip(args.label_noise, 0.0, 0.5))
    if noise_rate > 0:
        flip_mask = rng.random(len(feedback_labels)) < noise_rate
        feedback_labels[flip_mask] = 1 - feedback_labels[flip_mask]

    recal_threshold = choose_best_threshold(feedback_labels, selected_score[feedback_idx])

    before = metrics(y_true[eval_idx], selected_score[eval_idx], threshold=base_threshold)
    after = metrics(y_true[eval_idx], selected_score[eval_idx], threshold=recal_threshold)

    report = {
        "model": selected_model,
        "feedback_samples": int(len(feedback_idx)),
        "evaluation_samples": int(len(eval_idx)),
        "label_noise": noise_rate,
        "thresholds": {
            "baseline": base_threshold,
            "recalibrated": recal_threshold,
            "delta": float(recal_threshold - base_threshold),
        },
        "metrics_before": before,
        "metrics_after": after,
        "improvement": {
            "precision_delta": float(after["precision"] - before["precision"]),
            "recall_delta": float(after["recall"] - before["recall"]),
            "f1_delta": float(after["f1"] - before["f1"]),
            "pr_auc_delta": float(after["pr_auc"] - before["pr_auc"]),
            "roc_auc_delta": float(after["roc_auc"] - before["roc_auc"]),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Feedback simulation complete | model={selected_model} "
        f"f1_before={before['f1']:.4f} f1_after={after['f1']:.4f}",
        flush=True,
    )
    print(f"Saved feedback report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
