from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle GPU training runner for anomaly pipeline")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/incident-intel-output")
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument(
        "--max-images",
        type=int,
        default=300000,
        help="Maximum number of images to process for frame datasets (0 means all)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--max-benchmark-frames", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_processed = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    artifacts_dir = repo_root / "artifacts"

    data_processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features_path = data_processed / "features.csv"
    autoencoder_path = models_dir / "autoencoder.pt"
    iforest_path = models_dir / "isolation_forest.joblib"
    eval_report_path = artifacts_dir / "eval_report.json"
    benchmark_path = artifacts_dir / "latency_benchmark.json"

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_path else f"{str(repo_root)}{os.pathsep}{existing_path}"

    python_exec = sys.executable

    run_command(
        [
            python_exec,
            "ml/scripts/extract_features.py",
            "--input",
            args.input_dir,
            "--output",
            str(features_path),
            "--frame-step",
            str(args.frame_step),
            "--max-images",
            str(args.max_images),
        ],
        cwd=repo_root,
        env=env,
    )

    run_command(
        [
            python_exec,
            "ml/scripts/train_autoencoder.py",
            "--features",
            str(features_path),
            "--output",
            str(autoencoder_path),
            "--latent-dim",
            str(args.latent_dim),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ],
        cwd=repo_root,
        env=env,
    )

    run_command(
        [
            python_exec,
            "ml/scripts/train_feature_anomaly.py",
            "--features",
            str(features_path),
            "--output",
            str(iforest_path),
            "--n-estimators",
            str(args.n_estimators),
            "--contamination",
            str(args.contamination),
        ],
        cwd=repo_root,
        env=env,
    )

    run_command(
        [
            python_exec,
            "ml/scripts/evaluate_anomaly.py",
            "--features",
            str(features_path),
            "--ae",
            str(autoencoder_path),
            "--iforest",
            str(iforest_path),
            "--report",
            str(eval_report_path),
            "--device",
            args.device,
        ],
        cwd=repo_root,
        env=env,
    )

    if args.run_benchmark:
        run_command(
            [
                python_exec,
                "ml/scripts/benchmark_pipeline.py",
                "--input",
                "webcam",
                "--output",
                str(benchmark_path),
                "--max-frames",
                str(args.max_benchmark_frames),
            ],
            cwd=repo_root,
            env=env,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in [features_path, autoencoder_path, iforest_path, eval_report_path]:
        destination = output_dir / source.name
        shutil.copy2(source, destination)

    benchmark_destination = output_dir / benchmark_path.name
    if benchmark_path.exists():
        shutil.copy2(benchmark_path, benchmark_destination)

    with eval_report_path.open("r", encoding="utf-8") as infile:
        report = json.load(infile)

    summary = {
        "best_model": report.get("best_model"),
        "runtime": report.get("runtime"),
        "output_dir": str(output_dir),
        "files": sorted([path.name for path in output_dir.iterdir() if path.is_file()]),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    archive_base = output_dir / "incident_intel_training_outputs"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(output_dir))

    print("Training completed successfully")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Archive: {archive_path}")


if __name__ == "__main__":
    main()
