from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level features for anomaly models")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing videos")
    parser.add_argument("--output", type=str, required=True, help="Output feature CSV path")
    parser.add_argument("--frame-step", type=int, default=6, help="Process every N-th frame")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to process when using image datasets (0 means all)",
    )
    return parser.parse_args()


def compute_feature_vector(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
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

    is_anomaly = int(fall_flag or wrong_way_flag or (motion_score > 0.55 and object_count > 5))

    features = np.array(
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
    return features, gray, is_anomaly


def generate_synthetic_features(rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = rng.normal(loc=0.2, scale=0.08, size=(rows, 8)).clip(0, 1)
    base[:, 0] *= 9
    base[:, 2] *= 6
    base[:, 3] *= 4
    anomalies = rng.choice([0, 1], size=rows, p=[0.92, 0.08])
    base[anomalies == 1, 5] = 1.0
    base[anomalies == 1, 6] = 1.0
    base[anomalies == 1, 0] += rng.uniform(1.5, 4.0, size=(anomalies == 1).sum())

    df = pd.DataFrame(
        base,
        columns=[
            "object_count",
            "mean_conf",
            "person_count",
            "vehicle_count",
            "total_area_ratio",
            "fall_flag",
            "wrong_way_flag",
            "action_count",
        ],
    )
    df["is_anomaly"] = anomalies
    df["source"] = "synthetic"
    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_files = [
        p
        for p in input_path.rglob("*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ]

    image_files = [
        p
        for p in input_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    records: list[dict] = []
    if not video_files and not image_files:
        df = generate_synthetic_features()
        df.to_csv(output_path, index=False)
        print(f"No videos found in {input_path}. Generated synthetic features at {output_path}")
        return

    if video_files:
        for video in video_files:
            capture = cv2.VideoCapture(str(video))
            prev_gray = None
            frame_idx = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % args.frame_step != 0:
                    continue

                features, prev_gray, is_anomaly = compute_feature_vector(frame, prev_gray)
                records.append(
                    {
                        "object_count": float(features[0]),
                        "mean_conf": float(features[1]),
                        "person_count": float(features[2]),
                        "vehicle_count": float(features[3]),
                        "total_area_ratio": float(features[4]),
                        "fall_flag": float(features[5]),
                        "wrong_way_flag": float(features[6]),
                        "action_count": float(features[7]),
                        "is_anomaly": is_anomaly,
                        "source": video.name,
                    }
                )
            capture.release()
    else:
        sorted_images = sorted(image_files, key=lambda p: (str(p.parent), p.name))
        processed_images = 0
        current_group: str | None = None
        prev_gray = None

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
                prev_gray = None

            features, prev_gray, is_anomaly = compute_feature_vector(frame, prev_gray)
            records.append(
                {
                    "object_count": float(features[0]),
                    "mean_conf": float(features[1]),
                    "person_count": float(features[2]),
                    "vehicle_count": float(features[3]),
                    "total_area_ratio": float(features[4]),
                    "fall_flag": float(features[5]),
                    "wrong_way_flag": float(features[6]),
                    "action_count": float(features[7]),
                    "is_anomaly": is_anomaly,
                    "source": image_path.parent.name,
                }
            )
            processed_images += 1

        print(f"Processed {processed_images} images from {input_path}")

    if not records:
        df = generate_synthetic_features()
    else:
        df = pd.DataFrame.from_records(records)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
