from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class ModelOpsService:
    def __init__(
        self,
        *,
        base_threshold: float,
        min_threshold: float = 0.35,
        max_threshold: float = 0.95,
        recent_window: int = 4000,
        baseline_report_path: str = "artifacts/eval_report.json",
    ) -> None:
        self._lock = threading.Lock()
        self._threshold = float(np.clip(base_threshold, min_threshold, max_threshold))
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._recent_scores: deque[float] = deque(maxlen=max(100, recent_window))
        self._baseline = self._load_baseline_stats(Path(baseline_report_path))

    def _load_baseline_stats(self, path: Path) -> dict:
        if not path.exists():
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload.get("calibration", {})
            stats = calibration.get("reference_score_stats", {})
            if not isinstance(stats, dict):
                return {}
            return stats
        except Exception:
            return {}

    def observe_score(self, *, camera_id: str, score: float) -> None:
        del camera_id
        bounded = float(np.clip(score, 0.0, 1.0))
        with self._lock:
            self._recent_scores.append(bounded)

    def current_threshold(self) -> float:
        with self._lock:
            return float(self._threshold)

    def _recommend_threshold(self, scores: np.ndarray, labels: np.ndarray) -> float | None:
        if len(scores) < 3 or len(np.unique(labels)) < 2:
            return None

        best_threshold = None
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 91):
            y_pred = (scores >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary", zero_division=0)
            if float(f1) > best_f1:
                best_f1 = float(f1)
                best_threshold = float(threshold)

        return best_threshold

    def _feedback_dataset(self, event_store, limit: int = 3000) -> tuple[np.ndarray, np.ndarray, int]:
        items = event_store.list_feedback(limit=limit)
        if not items:
            return np.array([], dtype=float), np.array([], dtype=int), 0

        latest_by_event: dict[str, dict] = {}
        for item in reversed(items):
            event_id = item.get("event_id")
            if event_id and event_id not in latest_by_event:
                latest_by_event[event_id] = item

        scores: list[float] = []
        labels: list[int] = []

        for event_id, fb in latest_by_event.items():
            label = str(fb.get("label", "")).strip().lower()
            if label == "uncertain":
                continue

            event = event_store.get_event(event_id)
            if event is None or event.anomaly_score is None:
                continue

            scores.append(float(np.clip(event.anomaly_score, 0.0, 1.0)))
            labels.append(1 if label == "incident" else 0)

        return np.array(scores, dtype=float), np.array(labels, dtype=int), len(items)

    def calibration_status(self, event_store, min_samples: int = 20) -> dict:
        scores, labels, feedback_samples = self._feedback_dataset(event_store)
        recommended = self._recommend_threshold(scores, labels)

        return {
            "current_threshold": self.current_threshold(),
            "recommended_threshold": recommended,
            "feedback_samples": int(feedback_samples),
            "labeled_event_samples": int(len(labels)),
            "calibration_ready": bool(recommended is not None and len(labels) >= min_samples),
        }

    def recalibrate(self, event_store, *, min_samples: int = 20, apply: bool = True) -> dict:
        previous = self.current_threshold()
        scores, labels, _ = self._feedback_dataset(event_store)
        recommended = self._recommend_threshold(scores, labels)

        applied = False
        current = previous
        if apply and recommended is not None and len(labels) >= min_samples:
            with self._lock:
                self._threshold = float(np.clip(recommended, self._min_threshold, self._max_threshold))
                current = self._threshold
            applied = True

        return {
            "applied": applied,
            "previous_threshold": previous,
            "current_threshold": current,
            "recommended_threshold": recommended,
            "labeled_event_samples": int(len(labels)),
        }

    def drift_status(self) -> dict:
        baseline_mean = self._baseline.get("mean")
        baseline_std = self._baseline.get("std")

        with self._lock:
            recent = np.array(list(self._recent_scores), dtype=float)

        if recent.size < 30:
            return {
                "enabled": baseline_mean is not None,
                "status": "warming_up",
                "baseline_mean": baseline_mean,
                "recent_mean": float(np.mean(recent)) if recent.size else None,
                "mean_shift_sigma": None,
                "recent_count": int(recent.size),
            }

        recent_mean = float(np.mean(recent))
        if baseline_mean is None:
            return {
                "enabled": False,
                "status": "baseline_missing",
                "baseline_mean": None,
                "recent_mean": recent_mean,
                "mean_shift_sigma": None,
                "recent_count": int(recent.size),
            }

        sigma = max(float(baseline_std) if baseline_std is not None else 0.0, 0.05)
        shift_sigma = abs(recent_mean - float(baseline_mean)) / sigma

        if shift_sigma >= 2.5:
            status = "high"
        elif shift_sigma >= 1.5:
            status = "moderate"
        else:
            status = "low"

        return {
            "enabled": True,
            "status": status,
            "baseline_mean": float(baseline_mean),
            "recent_mean": recent_mean,
            "mean_shift_sigma": float(shift_sigma),
            "recent_count": int(recent.size),
        }
