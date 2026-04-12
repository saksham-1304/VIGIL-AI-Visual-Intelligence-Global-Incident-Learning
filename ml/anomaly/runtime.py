from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.anomaly.autoencoder import load_autoencoder, reconstruction_error
from ml.anomaly.feature_anomaly import anomaly_scores, load_iforest


def extract_realtime_features(frame, detections: list[dict], actions: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    frame_area = max(1.0, float(h * w))

    object_count = float(len(detections))
    mean_conf = float(np.mean([item["confidence"] for item in detections])) if detections else 0.0
    person_count = float(sum(1 for item in detections if item["label"].lower() == "person"))
    vehicle_count = float(
        sum(1 for item in detections if item["label"].lower() in {"car", "bus", "truck", "motorcycle"})
    )
    total_area_ratio = float(
        sum((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) for item in detections)
        / frame_area
    )
    fall_flag = 1.0 if any("fall" in action for action in actions) else 0.0
    wrong_way_flag = 1.0 if any("against_traffic" in action for action in actions) else 0.0
    action_count = float(len(actions))

    return np.array(
        [
            object_count,
            mean_conf,
            person_count,
            vehicle_count,
            total_area_ratio,
            fall_flag,
            wrong_way_flag,
            action_count,
        ],
        dtype=np.float32,
    )


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

        if Path(autoencoder_path).exists():
            self.autoencoder = load_autoencoder(autoencoder_path, device=device)

        if Path(iforest_path).exists():
            self.iforest = load_iforest(iforest_path)

    def score(self, frame, detections: list[dict], actions: list[str]) -> float:
        features = extract_realtime_features(frame, detections, actions)
        features_2d = features.reshape(1, -1)

        heuristic = self._heuristic_score(detections, actions)
        ae_score = self._autoencoder_score(features_2d)
        iforest_score = self._iforest_score(features_2d)

        blended = 0.5 * heuristic + 0.25 * ae_score + 0.25 * iforest_score
        return float(np.clip(blended, 0.0, 1.0))

    def _heuristic_score(self, detections: list[dict], actions: list[str]) -> float:
        score = 0.05
        score += min(0.3, len(detections) * 0.04)
        if any("fall" in action for action in actions):
            score += 0.5
        if any("against_traffic" in action for action in actions):
            score += 0.25
        if any(item["label"].lower() in {"fire", "smoke"} for item in detections):
            score += 0.4
        return float(np.clip(score, 0.0, 1.0))

    def _autoencoder_score(self, features_2d: np.ndarray) -> float:
        if self.autoencoder is None:
            return 0.0
        error = reconstruction_error(self.autoencoder, features_2d, device=self.device)[0]
        return float(np.clip(error * 8.0, 0.0, 1.0))

    def _iforest_score(self, features_2d: np.ndarray) -> float:
        if self.iforest is None:
            return 0.0
        return float(anomaly_scores(self.iforest, features_2d)[0])
