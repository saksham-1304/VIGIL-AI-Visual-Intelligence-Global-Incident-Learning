from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackState:
    centroid: np.ndarray
    missing: int = 0


class CentroidTracker:
    def __init__(self, max_missing: int = 20, distance_threshold: float = 90.0):
        self.max_missing = max_missing
        self.distance_threshold = distance_threshold
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        if not detections:
            self._mark_missing_for_all()
            return []

        centroids = np.array([self._bbox_center(item["bbox"]) for item in detections], dtype=np.float32)

        if not self.tracks:
            for idx, detection in enumerate(detections):
                track_id = self._register(centroids[idx])
                detection["track_id"] = track_id
            return detections

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids], dtype=np.float32)
        distance_matrix = self._pairwise_distance(track_centroids, centroids)

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        while True:
            if distance_matrix.size == 0:
                break
            row, col = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            distance = distance_matrix[row, col]
            if np.isinf(distance) or distance > self.distance_threshold:
                break

            track_id = track_ids[row]
            if track_id in assigned_tracks or col in assigned_detections:
                distance_matrix[row, col] = np.inf
                continue

            self.tracks[track_id].centroid = centroids[col]
            self.tracks[track_id].missing = 0
            detections[col]["track_id"] = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(col)

            distance_matrix[row, :] = np.inf
            distance_matrix[:, col] = np.inf

        for track_id in track_ids:
            if track_id not in assigned_tracks:
                self.tracks[track_id].missing += 1

        for idx, detection in enumerate(detections):
            if idx in assigned_detections:
                continue
            track_id = self._register(centroids[idx])
            detection["track_id"] = track_id

        self._cleanup_missing()
        return detections

    def _mark_missing_for_all(self) -> None:
        for state in self.tracks.values():
            state.missing += 1
        self._cleanup_missing()

    def _cleanup_missing(self) -> None:
        stale = [tid for tid, state in self.tracks.items() if state.missing > self.max_missing]
        for tid in stale:
            self.tracks.pop(tid, None)

    def _register(self, centroid: np.ndarray) -> int:
        track_id = self.next_id
        self.tracks[track_id] = TrackState(centroid=centroid, missing=0)
        self.next_id += 1
        return track_id

    @staticmethod
    def _bbox_center(bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @staticmethod
    def _pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.array([])
        diff = a[:, None, :] - b[None, :, :]
        return np.linalg.norm(diff, axis=-1)
