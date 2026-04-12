from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ml.anomaly.autoencoder import load_autoencoder, reconstruction_error
from ml.anomaly.feature_anomaly import anomaly_scores, load_iforest


FEATURE_ORDER = [
    "object_count",
    "mean_conf",
    "person_count",
    "vehicle_count",
    "total_area_ratio",
    "fall_flag",
    "wrong_way_flag",
    "action_count",
    "motion_score",
    "yolo_object_count",
    "yolo_mean_conf",
    "yolo_person_count",
    "yolo_vehicle_count",
    "yolo_risk_count",
    "yolo_area_ratio",
    "yolo_available",
]

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
RISK_LABELS = {"knife", "gun", "weapon", "fire", "smoke", "explosion"}


class HybridAnomalyScorer:
    def __init__(
        self,
        autoencoder_path: str = "models/autoencoder.pt",
        iforest_path: str = "models/isolation_forest.joblib",
        device: str = "cpu",
    ):
        self.device = device
        self.autoencoder = None
        self.iforest = None
        self._prev_gray = None

        if Path(autoencoder_path).exists():
            self.autoencoder = load_autoencoder(autoencoder_path, device=device)

        if Path(iforest_path).exists():
            self.iforest = load_iforest(iforest_path)

    def score(self, frame, detections: list[dict], actions: list[str]) -> float:
        return float(self.score_with_breakdown(frame, detections, actions)["score"])

    def score_with_breakdown(self, frame, detections: list[dict], actions: list[str]) -> dict:
        features = extract_realtime_features(frame, detections, actions, prev_gray=self._prev_gray)
        self._prev_gray = features["_next_prev_gray"]

        ordered_features = np.array([features[name] for name in FEATURE_ORDER], dtype=np.float32)

        heuristic = self._heuristic_score(detections, actions, features)
        yolo_semantic = self._yolo_semantic_score(features)

        ae_input = self._align_features(ordered_features, target_dim=self._autoencoder_input_dim())
        iforest_input = self._align_features(ordered_features, target_dim=self._iforest_input_dim())

        ae_score = self._autoencoder_score(ae_input)
        iforest_score = self._iforest_score(iforest_input)

        blended = 0.20 * heuristic + 0.20 * yolo_semantic + 0.30 * ae_score + 0.30 * iforest_score
        final_score = float(np.clip(blended, 0.0, 1.0))

        return {
            "score": final_score,
            "components": {
                "heuristic": float(heuristic),
                "yolo_semantic": float(yolo_semantic),
                "autoencoder": float(ae_score),
                "isolation_forest": float(iforest_score),
            },
            "features": {name: float(features[name]) for name in FEATURE_ORDER},
        }

    def _motion_score(self, frame) -> tuple[float, np.ndarray | None]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        except Exception:
            return 0.0, self._prev_gray

        if self._prev_gray is None:
            return 0.0, gray

        delta = cv2.absdiff(self._prev_gray, gray)
        motion_score = float(np.mean(delta) / 255.0)
        return float(np.clip(motion_score, 0.0, 1.0)), gray

    def _autoencoder_input_dim(self) -> int | None:
        if self.autoencoder is None:
            return None
        try:
            return int(self.autoencoder.encoder[0].in_features)
        except Exception:
            return None

    def _iforest_input_dim(self) -> int | None:
        if self.iforest is None:
            return None
        value = getattr(self.iforest, "n_features_in_", None)
        return int(value) if value is not None else None

    def _align_features(self, features: np.ndarray, target_dim: int | None) -> np.ndarray:
        if target_dim is None:
            return features.reshape(1, -1)

        if features.size == target_dim:
            return features.reshape(1, -1)

        if features.size > target_dim:
            return features[:target_dim].reshape(1, -1)

        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: features.size] = features
        return padded.reshape(1, -1)

    def _heuristic_score(self, detections: list[dict], actions: list[str], features: dict) -> float:
        score = 0.04
        score += min(0.25, len(detections) * 0.03)
        score += min(0.2, float(features["motion_score"]) * 0.4)
        if any("fall" in action.lower() for action in actions):
            score += 0.45
        if any("against_traffic" in action.lower() or "wrong_way" in action.lower() for action in actions):
            score += 0.25
        if any(item.get("label", "").lower() in RISK_LABELS for item in detections):
            score += 0.4
        return float(np.clip(score, 0.0, 1.0))

    def _yolo_semantic_score(self, features: dict) -> float:
        raw = (
            0.45 * float(features["yolo_risk_count"])
            + 0.15 * float(features["yolo_object_count"])
            + 0.2 * float(features["yolo_area_ratio"])
            + 0.2 * float(features["motion_score"])
            - 0.85
        )
        return float(np.clip(1.0 / (1.0 + np.exp(-raw)), 0.0, 1.0))

    def _autoencoder_score(self, features_2d: np.ndarray) -> float:
        if self.autoencoder is None:
            return 0.0
        error = reconstruction_error(self.autoencoder, features_2d, device=self.device)[0]
        return float(np.clip(error * 8.0, 0.0, 1.0))

    def _iforest_score(self, features_2d: np.ndarray) -> float:
        if self.iforest is None:
            return 0.0
        return float(anomaly_scores(self.iforest, features_2d)[0])


def extract_realtime_features(
    frame,
    detections: list[dict],
    actions: list[str],
    prev_gray,
) -> dict:
    h, w = frame.shape[:2]
    frame_area = max(1.0, float(h * w))

    labels = [item.get("label", "").lower() for item in detections]
    confidences = [float(item.get("confidence", 0.0)) for item in detections]

    total_area_ratio = float(
        sum((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) for item in detections)
        / frame_area
    ) if detections else 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    motion_score = 0.0
    if prev_gray is not None:
        delta = cv2.absdiff(prev_gray, gray)
        motion_score = float(np.mean(delta) / 255.0)

    object_count = float(len(detections))
    person_count = float(sum(1 for label in labels if label == "person"))
    vehicle_count = float(sum(1 for label in labels if label in VEHICLE_LABELS))
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    fall_flag = 1.0 if any("fall" in action.lower() for action in actions) else 0.0
    wrong_way_flag = 1.0 if any("against_traffic" in action.lower() or "wrong_way" in action.lower() for action in actions) else 0.0
    action_count = float(len(actions))

    yolo_risk_count = float(sum(1 for label in labels if label in RISK_LABELS))

    return {
        "object_count": object_count,
        "mean_conf": mean_conf,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "total_area_ratio": float(np.clip(total_area_ratio, 0.0, 1.0)),
        "fall_flag": fall_flag,
        "wrong_way_flag": wrong_way_flag,
        "action_count": action_count,
        "motion_score": float(np.clip(motion_score, 0.0, 1.0)),
        "yolo_object_count": object_count,
        "yolo_mean_conf": mean_conf,
        "yolo_person_count": person_count,
        "yolo_vehicle_count": vehicle_count,
        "yolo_risk_count": yolo_risk_count,
        "yolo_area_ratio": float(np.clip(total_area_ratio, 0.0, 1.0)),
        "yolo_available": 1.0,
        "_next_prev_gray": gray,
    }
