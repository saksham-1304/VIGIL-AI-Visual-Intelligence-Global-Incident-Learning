from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from ml.anomaly.runtime import HybridAnomalyScorer
from ml.vision.action_recognizer import ActionRecognizer
from ml.vision.detector import YoloDetector
from ml.vision.tracker import CentroidTracker


class MotionOnlyDetector:
    def __init__(self, confidence: float = 0.45) -> None:
        self.confidence = confidence
        self._prev_gray = None

    def detect(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        delta = cv2.absdiff(self._prev_gray, gray)
        threshold = cv2.threshold(delta, 22, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            label = "person" if h > w else "car"
            detections.append(
                {
                    "label": label,
                    "confidence": float(self.confidence),
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                }
            )

        self._prev_gray = gray
        return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-camera load benchmark for incident intelligence pipeline")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--frames-per-camera", type=int, default=75)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-device", type=str, default="cpu")
    parser.add_argument("--detector-mode", choices=["auto", "yolo", "fallback"], default="fallback")
    parser.add_argument("--target-p95-ms", type=float, default=180.0)
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


def synthetic_frame(step: int, camera_idx: int, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    x = int((step * 7 + camera_idx * 29) % max(40, width - 120))
    y = int((step * 5 + camera_idx * 17) % max(40, height - 120))

    cv2.rectangle(frame, (x, y), (x + 90, y + 140), (220, 220, 220), -1)
    cv2.rectangle(frame, (max(0, x - 80), max(0, y + 25)), (max(0, x - 30), max(0, y + 70)), (140, 140, 255), -1)

    noise = rng.integers(0, 35, size=frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def build_detector(mode: str, device: str) -> tuple[object, str]:
    if mode == "fallback":
        return MotionOnlyDetector(), "fallback"

    detector = YoloDetector(weights="yolov8n.pt", device=device, confidence=0.3)
    if mode == "yolo" and detector.model is None:
        raise RuntimeError("YOLO mode requested but ultralytics/weights are unavailable")

    if detector.model is None:
        return MotionOnlyDetector(), "fallback"

    return detector, "yolo"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), p))


def run_single_camera(
    camera_idx: int,
    frames: int,
    width: int,
    height: int,
    seed: int,
    detector_mode: str,
    model_device: str,
) -> dict:
    rng = np.random.default_rng(seed + camera_idx * 1009)

    detector, resolved_mode = build_detector(detector_mode, model_device)
    tracker = CentroidTracker(max_missing=20)
    action = ActionRecognizer(history_size=8)
    scorer = HybridAnomalyScorer(device=model_device)

    latencies_ms: list[float] = []
    anomaly_frames = 0

    start = time.perf_counter()
    for step in range(frames):
        frame = synthetic_frame(step, camera_idx, width, height, rng)

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        actions = action.predict(frame, tracked)
        details = scorer.score_with_breakdown(frame, tracked, actions)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(latency_ms)
        if float(details["score"]) >= 0.72:
            anomaly_frames += 1

    wall = max(1e-6, time.perf_counter() - start)

    return {
        "camera_id": f"cam-{camera_idx:02d}",
        "frames": int(frames),
        "detector_mode": resolved_mode,
        "fps": float(frames / wall),
        "anomaly_frames": int(anomaly_frames),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms) if latencies_ms else 0.0),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "max": float(max(latencies_ms) if latencies_ms else 0.0),
        },
        "raw_latencies_ms": latencies_ms,
    }


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.model_device)

    cameras = max(1, args.cameras)
    frames_per_camera = max(1, args.frames_per_camera)

    wall_start = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=cameras) as executor:
        futures = [
            executor.submit(
                run_single_camera,
                camera_idx=i,
                frames=frames_per_camera,
                width=args.width,
                height=args.height,
                seed=args.seed,
                detector_mode=args.detector_mode,
                model_device=resolved_device,
            )
            for i in range(cameras)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_wall = max(1e-6, time.perf_counter() - wall_start)
    all_latencies = [lat for item in results for lat in item.pop("raw_latencies_ms", [])]
    total_frames = int(cameras * frames_per_camera)

    summary = {
        "cameras": cameras,
        "frames_per_camera": frames_per_camera,
        "total_frames": total_frames,
        "wall_seconds": float(total_wall),
        "throughput_fps": float(total_frames / total_wall),
        "latency_ms": {
            "mean": float(np.mean(all_latencies) if all_latencies else 0.0),
            "p50": percentile(all_latencies, 50),
            "p95": percentile(all_latencies, 95),
            "p99": percentile(all_latencies, 99),
            "max": float(max(all_latencies) if all_latencies else 0.0),
        },
        "slo": {
            "target_p95_ms": float(args.target_p95_ms),
            "pass": bool(percentile(all_latencies, 95) <= float(args.target_p95_ms)),
        },
    }

    report = {
        "summary": summary,
        "per_camera": sorted(results, key=lambda item: item["camera_id"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Load test complete | cameras={cameras} frames={total_frames} "
        f"p95={summary['latency_ms']['p95']:.2f} ms throughput={summary['throughput_fps']:.2f} fps",
        flush=True,
    )
    print(f"Saved load report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
