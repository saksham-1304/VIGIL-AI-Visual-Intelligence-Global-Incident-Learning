from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
RISK_LABELS = {"knife", "gun", "weapon", "fire", "smoke", "explosion"}
NORMAL_CLASS_TOKENS = {"normal", "nonviolence", "neutral", "safe", "background"}
SKIP_CLASS_TOKENS = {
    "train",
    "test",
    "frames",
    "images",
    "dataset",
    "datasets",
    "ucf",
    "crime",
    "ucfcrime",
    "raw",
}


class OptionalYoloFeatureExtractor:
    def __init__(
        self,
        enabled: bool,
        weights: str,
        device: str,
        confidence: float,
    ) -> None:
        self.enabled = enabled
        self.weights = weights
        self.device = device
        self.confidence = confidence
        self.model = None
        self.names: dict[int, str] = {}

        if not enabled:
            print("[extract] YOLO disabled by flag", flush=True)
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(weights)
            print(f"[extract] YOLO enabled with weights={weights} device={device}", flush=True)
        except Exception as exc:
            self.model = None
            print(f"[extract] YOLO unavailable; fallback semantic features active: {exc}", flush=True)

    @property
    def available(self) -> bool:
        return self.model is not None

    def extract(self, frame: np.ndarray) -> dict[str, float]:
        if self.model is None:
            return {
                "yolo_object_count": 0.0,
                "yolo_mean_conf": 0.0,
                "yolo_person_count": 0.0,
                "yolo_vehicle_count": 0.0,
                "yolo_risk_count": 0.0,
                "yolo_area_ratio": 0.0,
                "yolo_available": 0.0,
            }

        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1.0, float(frame_h * frame_w))

        try:
            result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
            names = result.names
            if isinstance(names, dict):
                self.names = {int(k): str(v) for k, v in names.items()}

            object_count = 0
            conf_values: list[float] = []
            person_count = 0
            vehicle_count = 0
            risk_count = 0
            total_area = 0.0

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue

                cls_idx = int(box.cls[0])
                label = self.names.get(cls_idx, str(cls_idx)).lower()
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                area = max(0.0, (x2 - x1) * (y2 - y1))

                object_count += 1
                conf_values.append(conf)
                total_area += area
                if label == "person":
                    person_count += 1
                if label in VEHICLE_LABELS:
                    vehicle_count += 1
                if label in RISK_LABELS:
                    risk_count += 1

            return {
                "yolo_object_count": float(object_count),
                "yolo_mean_conf": float(np.mean(conf_values)) if conf_values else 0.0,
                "yolo_person_count": float(person_count),
                "yolo_vehicle_count": float(vehicle_count),
                "yolo_risk_count": float(risk_count),
                "yolo_area_ratio": float(np.clip(total_area / frame_area, 0.0, 1.0)),
                "yolo_available": 1.0,
            }
        except Exception:
            return {
                "yolo_object_count": 0.0,
                "yolo_mean_conf": 0.0,
                "yolo_person_count": 0.0,
                "yolo_vehicle_count": 0.0,
                "yolo_risk_count": 0.0,
                "yolo_area_ratio": 0.0,
                "yolo_available": 0.0,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level features for anomaly models")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing videos or image frames")
    parser.add_argument("--output", type=str, required=True, help="Output feature CSV path")
    parser.add_argument("--frame-step", type=int, default=6, help="Process every N-th frame/image")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to process when using image datasets (0 means all)",
    )
    parser.add_argument("--progress-every", type=int, default=5000, help="Log every N processed samples")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Log heartbeat every N seconds")
    parser.add_argument("--disable-yolo", action="store_true", help="Disable YOLO semantic feature extraction")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--yolo-device", type=str, default="cpu", help="YOLO device: cpu or cuda")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="YOLO confidence threshold")
    return parser.parse_args()


def infer_incident_class(path: Path) -> str:
    components = [part.lower() for part in path.parts]
    for part in reversed(components):
        clean = re.sub(r"[^a-z0-9_\-]", "", part)
        if not clean or clean in SKIP_CLASS_TOKENS:
            continue
        if clean in NORMAL_CLASS_TOKENS:
            return "normal"
        if clean in {"abnormal", "anomaly", "incident"}:
            return "abnormal"
        return clean
    return "unknown"


def is_anomaly_label(incident_class: str, heuristic_score: float) -> int:
    if incident_class == "unknown":
        return int(heuristic_score >= 0.55)
    return 0 if incident_class in NORMAL_CLASS_TOKENS else 1


def compute_handcrafted_features(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[dict[str, float], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)
    edge_density = float(np.mean(edges > 0))

    motion_score = 0.0
    if prev_gray is not None:
        delta = cv2.absdiff(prev_gray, gray)
        motion_score = float(np.mean(delta) / 255.0)

    object_count = float(np.clip(edge_density * 25 + motion_score * 6, 0, 12))
    person_count = float(np.clip(object_count * 0.5, 0, 8))
    vehicle_count = float(np.clip(object_count * 0.3, 0, 6))
    mean_conf = float(np.clip(0.35 + motion_score * 0.6, 0.3, 0.99))
    total_area_ratio = float(np.clip(edge_density * 2.5, 0, 1.0))
    fall_flag = float(1.0 if motion_score > 0.4 and object_count > 3 else 0.0)
    wrong_way_flag = float(1.0 if motion_score > 0.45 and vehicle_count > 1 else 0.0)
    action_count = float(np.clip(1 + motion_score * 5 + object_count * 0.2, 0, 10))

    return (
        {
            "object_count": object_count,
            "mean_conf": mean_conf,
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "total_area_ratio": total_area_ratio,
            "fall_flag": fall_flag,
            "wrong_way_flag": wrong_way_flag,
            "action_count": action_count,
            "motion_score": motion_score,
        },
        gray,
    )


def combine_feature_record(
    handcrafted: dict[str, float],
    yolo_features: dict[str, float],
    source: str,
    incident_class: str,
) -> dict[str, float | int | str]:
    # If YOLO is unavailable, backfill semantic channels from handcrafted approximations.
    if yolo_features["yolo_available"] < 0.5:
        yolo_features = {
            "yolo_object_count": handcrafted["object_count"],
            "yolo_mean_conf": handcrafted["mean_conf"],
            "yolo_person_count": handcrafted["person_count"],
            "yolo_vehicle_count": handcrafted["vehicle_count"],
            "yolo_risk_count": handcrafted["fall_flag"] + handcrafted["wrong_way_flag"],
            "yolo_area_ratio": handcrafted["total_area_ratio"],
            "yolo_available": 0.0,
        }

    weak_score = max(
        handcrafted["fall_flag"],
        handcrafted["wrong_way_flag"],
        handcrafted["motion_score"],
        yolo_features["yolo_risk_count"] * 0.35,
    )
    label = is_anomaly_label(incident_class, heuristic_score=weak_score)

    return {
        **handcrafted,
        **yolo_features,
        "is_anomaly": label,
        "incident_class": incident_class,
        "source": source,
    }


def generate_synthetic_features(rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    anomalies = rng.choice([0, 1], size=rows, p=[0.9, 0.1])

    object_count = rng.normal(loc=3.8, scale=1.2, size=rows).clip(0, 12)
    motion_score = rng.normal(loc=0.22, scale=0.12, size=rows).clip(0, 1)
    person_count = (object_count * rng.uniform(0.35, 0.7, size=rows)).clip(0, 8)
    vehicle_count = (object_count * rng.uniform(0.15, 0.45, size=rows)).clip(0, 6)
    mean_conf = rng.normal(loc=0.56, scale=0.08, size=rows).clip(0.3, 0.99)
    total_area_ratio = rng.normal(loc=0.25, scale=0.09, size=rows).clip(0, 1)
    fall_flag = (anomalies & (rng.random(rows) > 0.4)).astype(float)
    wrong_way_flag = (anomalies & (rng.random(rows) > 0.5)).astype(float)
    action_count = (1 + motion_score * 4 + object_count * 0.25).clip(0, 10)

    yolo_object_count = (object_count + rng.normal(0, 0.8, size=rows)).clip(0, 16)
    yolo_person_count = (person_count + rng.normal(0, 0.5, size=rows)).clip(0, 10)
    yolo_vehicle_count = (vehicle_count + rng.normal(0, 0.4, size=rows)).clip(0, 8)
    yolo_risk_count = (anomalies * rng.integers(0, 3, size=rows)).astype(float)
    yolo_mean_conf = (mean_conf + rng.normal(0, 0.05, size=rows)).clip(0.3, 0.99)
    yolo_area_ratio = (total_area_ratio + rng.normal(0, 0.06, size=rows)).clip(0, 1)

    df = pd.DataFrame(
        {
            "object_count": object_count,
            "mean_conf": mean_conf,
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "total_area_ratio": total_area_ratio,
            "fall_flag": fall_flag,
            "wrong_way_flag": wrong_way_flag,
            "action_count": action_count,
            "motion_score": motion_score,
            "yolo_object_count": yolo_object_count,
            "yolo_mean_conf": yolo_mean_conf,
            "yolo_person_count": yolo_person_count,
            "yolo_vehicle_count": yolo_vehicle_count,
            "yolo_risk_count": yolo_risk_count,
            "yolo_area_ratio": yolo_area_ratio,
            "yolo_available": np.ones(rows, dtype=float),
            "is_anomaly": anomalies.astype(int),
            "incident_class": np.where(anomalies == 1, "abnormal", "normal"),
            "source": "synthetic",
        }
    )
    return df


def collect_videos(input_path: Path) -> list[Path]:
    return [p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]


def collect_images(input_path: Path) -> list[Path]:
    return [p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def main() -> None:
    args = parse_args()
    progress_every = max(1, args.progress_every)
    heartbeat_seconds = max(0, args.heartbeat_seconds)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yolo = OptionalYoloFeatureExtractor(
        enabled=not args.disable_yolo,
        weights=args.yolo_weights,
        device=args.yolo_device,
        confidence=float(np.clip(args.yolo_conf, 0.01, 0.99)),
    )

    video_files = collect_videos(input_path)
    image_files = collect_images(input_path)

    records: list[dict[str, float | int | str]] = []
    if not video_files and not image_files:
        df = generate_synthetic_features()
        df.to_csv(output_path, index=False)
        print(f"No videos/images found in {input_path}. Generated synthetic features at {output_path}", flush=True)
        return

    if video_files:
        total_videos = len(video_files)
        for video_idx, video in enumerate(video_files, start=1):
            print(f"[extract] Processing video {video_idx}/{total_videos}: {video}", flush=True)
            capture = cv2.VideoCapture(str(video))
            prev_gray = None
            frame_idx = 0
            sampled_frames = 0
            incident_class = infer_incident_class(video.parent)
            last_log = time.monotonic()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_idx += 1
                if frame_idx % args.frame_step != 0:
                    continue

                handcrafted, prev_gray = compute_handcrafted_features(frame, prev_gray)
                yolo_features = yolo.extract(frame)
                records.append(
                    combine_feature_record(
                        handcrafted=handcrafted,
                        yolo_features=yolo_features,
                        source=video.name,
                        incident_class=incident_class,
                    )
                )
                sampled_frames += 1

                now = time.monotonic()
                should_log = sampled_frames % progress_every == 0
                if heartbeat_seconds > 0 and now - last_log >= heartbeat_seconds:
                    should_log = True
                if should_log:
                    print(
                        f"[extract] video={video.name} sampled_frames={sampled_frames} "
                        f"frames_read={frame_idx} total_rows={len(records)}",
                        flush=True,
                    )
                    last_log = now

            capture.release()
            print(
                f"[extract] Completed {video.name}: sampled_frames={sampled_frames} rows_total={len(records)}",
                flush=True,
            )
    else:
        sorted_images = sorted(image_files, key=lambda p: (str(p.parent), p.name))
        total_images = len(sorted_images)
        processed_images = 0
        current_group: str | None = None
        current_class = "unknown"
        prev_gray = None
        last_log = time.monotonic()

        for index, image_path in enumerate(sorted_images):
            if args.max_images > 0 and processed_images >= args.max_images:
                break
            if index % args.frame_step != 0:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            group = str(image_path.parent)
            if group != current_group:
                current_group = group
                current_class = infer_incident_class(image_path.parent)
                prev_gray = None

            handcrafted, prev_gray = compute_handcrafted_features(frame, prev_gray)
            yolo_features = yolo.extract(frame)
            records.append(
                combine_feature_record(
                    handcrafted=handcrafted,
                    yolo_features=yolo_features,
                    source=image_path.parent.name,
                    incident_class=current_class,
                )
            )
            processed_images += 1

            now = time.monotonic()
            should_log = processed_images % progress_every == 0
            if heartbeat_seconds > 0 and now - last_log >= heartbeat_seconds:
                should_log = True
            if should_log:
                print(
                    f"[extract] images_processed={processed_images}/{total_images} "
                    f"total_rows={len(records)} current={image_path.name}",
                    flush=True,
                )
                last_log = now

        print(f"Processed {processed_images} images from {input_path}", flush=True)

    if not records:
        df = generate_synthetic_features()
    else:
        df = pd.DataFrame.from_records(records)

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"source", "incident_class"} and df[col].dtype != object
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df.to_csv(output_path, index=False)
    print(
        f"Saved {len(df)} feature rows to {output_path} | feature_columns={len([c for c in df.columns if c not in {'source', 'incident_class', 'is_anomaly'}])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
