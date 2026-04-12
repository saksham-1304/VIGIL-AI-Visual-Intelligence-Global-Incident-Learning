from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.anomaly.autoencoder import load_autoencoder, reconstruction_error
from ml.anomaly.feature_anomaly import anomaly_scores, load_iforest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly models with split-safe research diagnostics")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--ae", type=str, required=True)
    parser.add_argument("--iforest", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cross-scene-min-rows", type=int, default=25)
    parser.add_argument("--baseline-max-train", type=int, default=20000)
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


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 91)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return float(np.quantile(y_score, 0.95))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid():
        y_pred = (y_score >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "positive_predictions": 0,
        }

    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "positive_predictions": int(np.sum(y_pred)),
    }


def summarize_score_distribution(score: np.ndarray) -> dict:
    if len(score) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "mean": float(np.mean(score)),
        "std": float(np.std(score)),
        "min": float(np.min(score)),
        "max": float(np.max(score)),
        "p50": float(np.quantile(score, 0.50)),
        "p90": float(np.quantile(score, 0.90)),
        "p95": float(np.quantile(score, 0.95)),
        "p99": float(np.quantile(score, 0.99)),
    }


def yolo_semantic_score(df: pd.DataFrame) -> np.ndarray:
    def series(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    yolo_risk = series("yolo_risk_count")
    yolo_object_count = series("yolo_object_count")
    yolo_area_ratio = series("yolo_area_ratio")
    motion_score = series("motion_score")
    fall_flag = series("fall_flag")
    wrong_way_flag = series("wrong_way_flag")

    linear = (
        0.55 * yolo_risk
        + 0.15 * yolo_object_count
        + 0.20 * yolo_area_ratio
        + 0.30 * motion_score
        + 0.35 * fall_flag
        + 0.25 * wrong_way_flag
        - 0.85
    )
    score = 1.0 / (1.0 + np.exp(-linear))
    return np.clip(score.to_numpy(dtype=float), 0.0, 1.0)


def build_split_masks(
    df: pd.DataFrame,
    source_col: str | None,
    holdout_ratio: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = len(df)
    if rows < 2:
        train_mask = np.ones(rows, dtype=bool)
        test_mask = np.zeros(rows, dtype=bool)
        return train_mask, test_mask, {"strategy": "single_row", "holdout_ratio": 0.0}

    ratio = float(np.clip(holdout_ratio, 0.05, 0.5))
    rng = np.random.default_rng(random_seed)

    if source_col is not None and source_col in df.columns and int(df[source_col].nunique()) >= 2:
        sources = sorted(df[source_col].astype(str).unique())
        shuffled = np.array(sources, dtype=object)
        rng.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * ratio))
        test_sources = set(shuffled[:test_count].tolist())

        test_mask = df[source_col].astype(str).isin(test_sources).to_numpy()
        if int(np.sum(test_mask)) == 0 or int(np.sum(~test_mask)) == 0:
            idx = np.arange(rows)
            rng.shuffle(idx)
            n_test = max(1, int(rows * ratio))
            test_mask = np.zeros(rows, dtype=bool)
            test_mask[idx[:n_test]] = True
            strategy = "row_random_fallback"
        else:
            strategy = "source_holdout"

        return (~test_mask), test_mask, {
            "strategy": strategy,
            "holdout_ratio": ratio,
            "test_sources": sorted(test_sources),
            "source_count": len(sources),
        }

    idx = np.arange(rows)
    rng.shuffle(idx)
    n_test = max(1, int(rows * ratio))
    test_mask = np.zeros(rows, dtype=bool)
    test_mask[idx[:n_test]] = True
    return (~test_mask), test_mask, {
        "strategy": "row_random",
        "holdout_ratio": ratio,
    }


def per_class_incident_recall(
    class_labels: pd.Series,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    y_pred = (y_score >= threshold).astype(int)
    result: dict[str, dict] = {}

    class_labels = class_labels.astype(str)
    for cls in sorted(class_labels.unique()):
        idx = (class_labels == cls).to_numpy()
        y_cls = y_true[idx]
        yhat_cls = y_pred[idx]

        positives = int(np.sum(y_cls == 1))
        negatives = int(np.sum(y_cls == 0))
        support = int(np.sum(idx))

        if positives > 0:
            recall = float(np.sum((yhat_cls == 1) & (y_cls == 1)) / max(1, positives))
            result[cls] = {
                "support": support,
                "positive_support": positives,
                "negative_support": negatives,
                "recall": recall,
            }
        else:
            fpr = float(np.sum((yhat_cls == 1) & (y_cls == 0)) / max(1, negatives))
            result[cls] = {
                "support": support,
                "positive_support": positives,
                "negative_support": negatives,
                "false_positive_rate": fpr,
            }

    return result


def cross_scene_diagnostics(
    source: pd.Series,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    min_rows: int,
) -> dict:
    rows = []
    source = source.astype(str)

    for src in sorted(source.unique()):
        mask = (source == src).to_numpy()
        if int(np.sum(mask)) < min_rows:
            continue

        y_src = y_true[mask]
        s_src = y_score[mask]
        metrics = classification_metrics(y_src, s_src, threshold)
        rows.append(
            {
                "source": src,
                "rows": int(np.sum(mask)),
                "anomaly_ratio": float(np.mean(y_src)) if len(y_src) else 0.0,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    if not rows:
        return {
            "eligible_sources": 0,
            "min_rows": int(min_rows),
            "source_metrics": [],
            "summary": {},
        }

    f1_values = [item["f1"] for item in rows]
    recall_values = [item["recall"] for item in rows]

    return {
        "eligible_sources": len(rows),
        "min_rows": int(min_rows),
        "source_metrics": rows,
        "summary": {
            "f1_mean": float(np.mean(f1_values)),
            "f1_std": float(np.std(f1_values)),
            "recall_mean": float(np.mean(recall_values)),
            "recall_std": float(np.std(recall_values)),
        },
    }


def _sigmoid_outlier(decision: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(3.0 * decision))


def subsample_rows(x: np.ndarray, max_rows: int, random_seed: int) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def ocsvm_scores(
    x_train_fit: np.ndarray,
    x_train_eval: np.ndarray,
    x_test: np.ndarray,
    random_seed: int,
    max_train_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_train_fit = subsample_rows(x_train_fit, max_rows=max_train_rows, random_seed=random_seed)
    if len(x_train_fit) < 50:
        return None

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_train_fit)
    x_train_scaled = scaler.transform(x_train_eval)
    x_test_scaled = scaler.transform(x_test)

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    model.fit(x_fit_scaled)

    train_score = _sigmoid_outlier(model.decision_function(x_train_scaled))
    test_score = _sigmoid_outlier(model.decision_function(x_test_scaled))
    return train_score.astype(float), test_score.astype(float)


def lof_scores(
    x_train_fit: np.ndarray,
    x_train_eval: np.ndarray,
    x_test: np.ndarray,
    random_seed: int,
    max_train_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_train_fit = subsample_rows(x_train_fit, max_rows=max_train_rows, random_seed=random_seed)
    if len(x_train_fit) < 80:
        return None

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_train_fit)
    x_train_scaled = scaler.transform(x_train_eval)
    x_test_scaled = scaler.transform(x_test)

    n_neighbors = min(35, len(x_fit_scaled) - 1)
    if n_neighbors < 5:
        return None

    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    model.fit(x_fit_scaled)

    train_score = _sigmoid_outlier(model.decision_function(x_train_scaled))
    test_score = _sigmoid_outlier(model.decision_function(x_test_scaled))
    return train_score.astype(float), test_score.astype(float)


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    df = pd.read_csv(args.features)

    class_col = "incident_class" if "incident_class" in df.columns else None
    source_col = "source" if "source" in df.columns else None

    feature_cols = [
        col
        for col in df.columns
        if col not in {"is_anomaly", "source", "incident_class"} and np.issubdtype(df[col].dtype, np.number)
    ]

    x = df[feature_cols].astype(float).to_numpy()
    y_true = df["is_anomaly"].astype(int).to_numpy() if "is_anomaly" in df.columns else np.zeros(len(df), dtype=int)

    train_mask, test_mask, split_info = build_split_masks(
        df=df,
        source_col=source_col,
        holdout_ratio=args.holdout_ratio,
        random_seed=args.random_seed,
    )

    x_train = x[train_mask]
    x_test = x[test_mask]
    y_train = y_true[train_mask]
    y_test = y_true[test_mask]

    ae_model = load_autoencoder(args.ae, device=resolved_device)
    ae_error = reconstruction_error(ae_model, x, device=resolved_device)
    ae_score_all = np.clip(ae_error * 8.0, 0, 1)

    iforest_model = load_iforest(args.iforest)
    iforest_score_all = anomaly_scores(iforest_model, x)

    yolo_score_all = yolo_semantic_score(df)
    hybrid_all = 0.5 * ae_score_all + 0.5 * iforest_score_all
    fusion_all = np.clip(0.4 * hybrid_all + 0.4 * yolo_score_all + 0.2 * ae_score_all, 0.0, 1.0)

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "autoencoder": (ae_score_all[train_mask], ae_score_all[test_mask]),
        "isolation_forest": (iforest_score_all[train_mask], iforest_score_all[test_mask]),
        "hybrid": (hybrid_all[train_mask], hybrid_all[test_mask]),
        "yolo_semantic": (yolo_score_all[train_mask], yolo_score_all[test_mask]),
        "hybrid_yolo_fusion": (fusion_all[train_mask], fusion_all[test_mask]),
    }

    normal_mask_train = y_train == 0
    x_train_normal = x_train[normal_mask_train] if int(np.sum(normal_mask_train)) > 0 else x_train

    ocsvm = ocsvm_scores(
        x_train_fit=x_train_normal,
        x_train_eval=x_train,
        x_test=x_test,
        random_seed=args.random_seed,
        max_train_rows=max(500, args.baseline_max_train),
    )
    if ocsvm is not None:
        candidates["one_class_svm"] = ocsvm

    lof = lof_scores(
        x_train_fit=x_train_normal,
        x_train_eval=x_train,
        x_test=x_test,
        random_seed=args.random_seed + 7,
        max_train_rows=max(500, args.baseline_max_train),
    )
    if lof is not None:
        candidates["local_outlier_factor"] = lof

    models: dict[str, dict] = {}
    ablations: list[dict] = []

    for model_name, (train_score, test_score) in candidates.items():
        threshold = find_best_threshold(y_train, train_score)
        train_metrics = classification_metrics(y_train, train_score, threshold=threshold)
        test_metrics = classification_metrics(y_test, test_score, threshold=threshold)

        models[model_name] = {
            **test_metrics,
            "train_f1": train_metrics["f1"],
            "train_pr_auc": train_metrics["pr_auc"],
            "calibrated_on": "train_split",
            "test_rows": int(len(y_test)),
        }

        ablations.append(
            {
                "model": model_name,
                "f1": test_metrics["f1"],
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "threshold": test_metrics["threshold"],
                "train_f1": train_metrics["f1"],
            }
        )

    ablations = sorted(ablations, key=lambda item: item["f1"], reverse=True)
    best_model_name = ablations[0]["model"] if ablations else "hybrid"
    best_metrics = models[best_model_name]

    best_train_score, best_test_score = candidates[best_model_name]

    class_recall = {}
    if class_col is not None and int(np.sum(test_mask)) > 0:
        class_recall = per_class_incident_recall(
            class_labels=df.loc[test_mask, class_col],
            y_true=y_test,
            y_score=best_test_score,
            threshold=best_metrics["threshold"],
        )

    cross_scene = {}
    if source_col is not None and int(np.sum(test_mask)) > 0:
        cross_scene = cross_scene_diagnostics(
            source=df.loc[test_mask, source_col],
            y_true=y_test,
            y_score=best_test_score,
            threshold=best_metrics["threshold"],
            min_rows=max(5, args.cross_scene_min_rows),
        )

    report = {
        "dataset": {
            "rows": int(len(df)),
            "features": feature_cols,
            "anomaly_ratio": float(np.mean(y_true)) if len(y_true) else 0.0,
            "sources": int(df[source_col].nunique()) if source_col is not None else 0,
            "incident_classes": int(df[class_col].nunique()) if class_col is not None else 0,
            "split": {
                **split_info,
                "train_rows": int(np.sum(train_mask)),
                "test_rows": int(np.sum(test_mask)),
            },
        },
        "runtime": {"device": resolved_device},
        "models": models,
        "ablations": ablations,
        "best_model": {
            "name": best_model_name,
            "f1": best_metrics["f1"],
            "threshold": best_metrics["threshold"],
            "pr_auc": best_metrics["pr_auc"],
        },
        "diagnostics": {
            "per_class_incident_recall": class_recall,
            "cross_scene": cross_scene,
        },
        "calibration": {
            "recommended_model": best_model_name,
            "recommended_threshold": best_metrics["threshold"],
            "reference_score_stats": summarize_score_distribution(best_train_score),
            "reference_split": "train",
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluation report written to {report_path}")
    print(
        f"Best model: {best_model_name} | test_f1={best_metrics['f1']:.4f} "
        f"test_pr_auc={best_metrics['pr_auc']:.4f} threshold={best_metrics['threshold']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
