from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from app.core.schemas import Detection, IncidentEvent, Severity
from app.services.metrics import incidents_total, inference_latency_seconds, processed_frames_total

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


logger = logging.getLogger(__name__)


class _FallbackDetector:
    def detect(self, _frame) -> list[dict]:
        return []


class _FallbackTracker:
    def update(self, detections: list[dict]) -> list[dict]:
        return detections


class _FallbackActionRecognizer:
    def predict(self, _frame, _detections: list[dict]) -> list[str]:
        return []


class _FallbackAnomaly:
    def score(self, _frame, _detections: list[dict], _actions: list[str]) -> float:
        return 0.0


class _FallbackExplainer:
    def describe_scene(self, _frame, detections: list[dict], actions: list[str], anomaly_score: float) -> str:
        return (
            f"Fallback description: detections={len(detections)}, "
            f"actions={len(actions)}, anomaly_score={anomaly_score:.2f}"
        )


class StreamProcessor:
    def __init__(self, event_store, alert_engine, ws_manager, settings, event_loop=None):
        self.event_store = event_store
        self.alert_engine = alert_engine
        self.ws_manager = ws_manager
        self.settings = settings
        self.event_loop = event_loop

        self.detector = _FallbackDetector()
        self.tracker = _FallbackTracker()
        self.action_recognizer = _FallbackActionRecognizer()
        self.anomaly = _FallbackAnomaly()
        self.explainer = _FallbackExplainer()

        try:
            from ml.anomaly.runtime import HybridAnomalyScorer
            from ml.multimodal.explainer import MultimodalExplainer
            from ml.vision.action_recognizer import ActionRecognizer
            from ml.vision.detector import YoloDetector
            from ml.vision.tracker import CentroidTracker

            self.detector = YoloDetector(weights=settings.yolo_weights, device=settings.model_device)
            self.tracker = CentroidTracker(max_missing=24)
            self.action_recognizer = ActionRecognizer()
            self.anomaly = HybridAnomalyScorer()
            self.explainer = MultimodalExplainer(model_name=settings.blip_model)
            logger.info("Loaded full CV and multimodal stack")
        except Exception as exc:
            logger.warning("Using lightweight fallback inference modules: %s", exc)

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def start_stream(self, source: str, camera_id: str) -> bool:
        if self._running:
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, args=(source, camera_id), daemon=True)
        self._running = True
        self._thread.start()
        logger.info("Started stream processing | source=%s | camera_id=%s", source, camera_id)
        return True

    def stop_stream(self) -> bool:
        if not self._running:
            return False
        self._stop_event.set()
        self._running = False
        logger.info("Stopping stream processing")
        return True

    def process_uploaded_video(self, file_path: str, camera_id: str) -> None:
        if cv2 is None:
            logger.error("OpenCV not available. Uploaded video processing is disabled.")
            return

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            logger.error("Unable to open uploaded video: %s", file_path)
            return

        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % 4 != 0:
                    continue
                event = self._analyze_frame(frame, camera_id=camera_id)
                if event is not None:
                    self._persist_and_alert(event)
        finally:
            capture.release()

    def _capture_loop(self, source: str, camera_id: str) -> None:
        if cv2 is None:
            logger.error("OpenCV not available. Live stream processing is disabled.")
            self._running = False
            return

        capture_source: int | str = 0 if source == "webcam" else source
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            logger.error("Failed to open source: %s", source)
            self._running = False
            return

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                frame_idx += 1
                if frame_idx % 3 != 0:
                    continue

                event = self._analyze_frame(frame, camera_id=camera_id)
                if event is not None:
                    self._persist_and_alert(event)
        finally:
            capture.release()
            self._running = False

    def _analyze_frame(self, frame, camera_id: str) -> IncidentEvent | None:
        start = time.perf_counter()

        detections_raw = self.detector.detect(frame)
        tracked = self.tracker.update(detections_raw)
        actions = self.action_recognizer.predict(frame, tracked)
        anomaly_score = self.anomaly.score(frame, tracked, actions)
        description = self.explainer.describe_scene(frame, tracked, actions, anomaly_score)

        elapsed = time.perf_counter() - start
        inference_latency_seconds.observe(elapsed)
        processed_frames_total.labels(camera_id=camera_id).inc()

        if not tracked and anomaly_score < self.settings.anomaly_threshold:
            return None

        event_type = "anomaly" if anomaly_score >= self.settings.anomaly_threshold else "behavior"
        severity = Severity.high if anomaly_score >= self.settings.anomaly_threshold else Severity.medium

        if any("fall" in action.lower() for action in actions):
            event_type = "fall_detection"
            severity = Severity.critical

        if any(item["label"].lower() in {"fire", "smoke"} for item in tracked):
            event_type = "hazard"
            severity = Severity.critical

        event = IncidentEvent(
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
            description=description,
            detections=[
                Detection(
                    label=item["label"],
                    confidence=float(item["confidence"]),
                    bbox=[float(v) for v in item["bbox"]],
                    track_id=item.get("track_id"),
                )
                for item in tracked
            ],
            actions=actions,
            anomaly_score=round(float(anomaly_score), 4),
            metadata={"latency_ms": round(elapsed * 1000, 2)},
        )
        return event

    def _persist_and_alert(self, event: IncidentEvent) -> None:
        self.event_store.create_event(event)
        incidents_total.labels(
            camera_id=event.camera_id,
            severity=event.severity.value,
            event_type=event.event_type,
        ).inc()

        alerts = self.alert_engine.evaluate(event)
        payload = {"type": "event", "payload": event.model_dump(mode="json")}

        for alert in alerts:
            self.event_store.create_alert(alert)

        if alerts:
            payload["alerts"] = [alert.model_dump(mode="json") for alert in alerts]

        try:
            if self.event_loop and self.event_loop.is_running():
                asyncio.run_coroutine_threadsafe(self.ws_manager.broadcast_json(payload), self.event_loop)
        except Exception as exc:
            logger.debug("Websocket broadcast skipped: %s", exc)
