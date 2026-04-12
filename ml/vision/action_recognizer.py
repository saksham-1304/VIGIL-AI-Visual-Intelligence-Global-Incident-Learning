from __future__ import annotations

from collections import defaultdict, deque


class ActionRecognizer:
    def __init__(self, history_size: int = 8):
        self.history_size = history_size
        self.track_history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    def predict(self, _frame, detections: list[dict]) -> list[str]:
        actions: set[str] = set()

        for item in detections:
            label = item["label"].lower()
            x1, y1, x2, y2 = item["bbox"]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            track_id = item.get("track_id")
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if track_id is not None:
                self.track_history[track_id].append(centroid)

            if label == "person":
                ratio = height / width
                if ratio < 0.75:
                    actions.add("person_falling")
                elif ratio > 1.6:
                    actions.add("person_walking")
                else:
                    actions.add("person_standing")

            if label in {"car", "bus", "truck", "motorcycle"}:
                direction = self._direction_delta(track_id)
                if direction is not None and direction < -25:
                    actions.add("vehicle_against_traffic")
                elif direction is not None and abs(direction) > 15:
                    actions.add("vehicle_moving")

            if label in {"fire", "smoke"}:
                actions.add("fire_hazard")

        return sorted(actions)

    def _direction_delta(self, track_id: int | None) -> float | None:
        if track_id is None:
            return None
        history = self.track_history.get(track_id)
        if not history or len(history) < 2:
            return None
        return history[-1][0] - history[0][0]
