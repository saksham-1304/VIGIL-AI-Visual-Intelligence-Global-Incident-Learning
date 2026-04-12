from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ml.anomaly.runtime import HybridAnomalyScorer
from ml.multimodal.explainer import MultimodalExplainer
from ml.vision.action_recognizer import ActionRecognizer
from ml.vision.detector import YoloDetector
from ml.vision.tracker import CentroidTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latency and throughput")
    parser.add_argument("--input", type=str, default="webcam")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    return parser.parse_args()


def run_mode(mode: str, capture, max_frames: int) -> dict:
    detector = YoloDetector()
    tracker = CentroidTracker()
    action = ActionRecognizer()
    anomaly = HybridAnomalyScorer()
    explainer = MultimodalExplainer()

    latencies_ms: list[float] = []
    events = 0
    frames = 0

    while frames < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1

        start = time.perf_counter()
        detections = detector.detect(frame)

        tracked = detections
        actions: list[str] = []
        score = 0.0

        if mode in {"detection_tracking", "full"}:
            tracked = tracker.update(detections)

        if mode == "full":
            actions = action.predict(frame, tracked)
            score = anomaly.score(frame, tracked, actions)
            _ = explainer.describe_scene(frame, tracked, actions, score)

        if tracked or score > 0.6:
            events += 1

        latencies_ms.append((time.perf_counter() - start) * 1000)

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0

    return {
        "mode": mode,
        "frames": frames,
        "events": events,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "fps": fps,
    }


def main() -> None:
    args = parse_args()
    capture_source = 0 if args.input == "webcam" else args.input

    results = []
    for mode in ["detection_only", "detection_tracking", "full"]:
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source {args.input}")
        results.append(run_mode(mode, capture, max_frames=args.max_frames))
        capture.release()

    report = {
        "benchmark": results,
        "tradeoff_summary": {
            "fastest": min(results, key=lambda r: r["avg_latency_ms"])["mode"],
            "highest_event_sensitivity": max(results, key=lambda r: r["events"])["mode"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Benchmark report saved to {output_path}")


if __name__ == "__main__":
    main()
