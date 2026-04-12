from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "cpu", confidence: float = 0.35):
        self.weights = weights
        self.device = device
        self.confidence = confidence
        self.model = None
        self._prev_gray = None

        try:
            from ultralytics import YOLO

            self.model = YOLO(weights)
            logger.info("Loaded YOLO model from %s on %s", weights, device)
        except Exception as exc:
            logger.warning("YOLO unavailable, using motion fallback detector: %s", exc)

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self.model is not None:
            return self._detect_with_yolo(frame)
        return self._detect_with_motion(frame)

    def _detect_with_yolo(self, frame: np.ndarray) -> list[dict]:
        output: list[dict] = []
        try:
            result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
            names = result.names
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue

                cls_idx = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                output.append(
                    {
                        "label": names.get(cls_idx, str(cls_idx)),
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        except Exception as exc:
            logger.debug("YOLO inference failed, returning empty detections: %s", exc)
        return output

    def _detect_with_motion(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        frame_delta = cv2.absdiff(self._prev_gray, gray)
        threshold = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output: list[dict] = []
        for contour in contours:
            if cv2.contourArea(contour) < 1800:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            output.append(
                {
                    "label": "motion",
                    "confidence": 0.51,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                }
            )

        self._prev_gray = gray
        return output
