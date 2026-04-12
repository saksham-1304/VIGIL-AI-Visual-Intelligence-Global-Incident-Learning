from __future__ import annotations

import subprocess
from pathlib import Path

from prefect import flow, task


def run_cmd(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stderr}")


@task
def extract_features_task(data_dir: str, feature_path: str) -> None:
    run_cmd([
        "python",
        "ml/scripts/extract_features.py",
        "--input",
        data_dir,
        "--output",
        feature_path,
    ])


@task
def train_autoencoder_task(feature_path: str, model_path: str) -> None:
    run_cmd([
        "python",
        "ml/scripts/train_autoencoder.py",
        "--features",
        feature_path,
        "--output",
        model_path,
    ])


@task
def train_iforest_task(feature_path: str, model_path: str) -> None:
    run_cmd([
        "python",
        "ml/scripts/train_feature_anomaly.py",
        "--features",
        feature_path,
        "--output",
        model_path,
    ])


@task
def evaluate_task(feature_path: str, ae_path: str, iforest_path: str, report_path: str) -> None:
    run_cmd([
        "python",
        "ml/scripts/evaluate_anomaly.py",
        "--features",
        feature_path,
        "--ae",
        ae_path,
        "--iforest",
        iforest_path,
        "--report",
        report_path,
    ])


@task
def benchmark_task(report_path: str) -> None:
    run_cmd([
        "python",
        "ml/scripts/benchmark_pipeline.py",
        "--input",
        "webcam",
        "--output",
        report_path,
        "--max-frames",
        "180",
    ])


@flow(name="incident-intelligence-training-flow", log_prints=True)
def incident_training_flow(data_dir: str = "data/raw") -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    feature_path = "data/processed/features.csv"
    ae_path = "models/autoencoder.pt"
    iforest_path = "models/isolation_forest.joblib"
    eval_report = "artifacts/eval_report.json"
    benchmark_report = "artifacts/latency_benchmark.json"

    extract_features_task(data_dir, feature_path)
    train_autoencoder_task(feature_path, ae_path)
    train_iforest_task(feature_path, iforest_path)
    evaluate_task(feature_path, ae_path, iforest_path, eval_report)
    benchmark_task(benchmark_report)


if __name__ == "__main__":
    incident_training_flow()
