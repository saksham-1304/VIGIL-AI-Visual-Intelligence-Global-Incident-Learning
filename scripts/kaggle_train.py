from __future__ import annotations

import argparse
import errno
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import TextIO


def _format_duration(seconds: float) -> str:
    total = int(max(0, seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _stream_pipe(pipe: TextIO | None, output_queue: queue.Queue[str | None]) -> None:
    if pipe is None:
        output_queue.put(None)
        return
    try:
        for line in iter(pipe.readline, ""):
            output_queue.put(line.rstrip("\n"))
    finally:
        pipe.close()
        output_queue.put(None)


def run_command(
    stage_name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    heartbeat_seconds: int,
) -> None:
    print(f"\n=== Stage: {stage_name} ===", flush=True)
    print("$", " ".join(command), flush=True)

    start_time = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_queue: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_stream_pipe, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    stream_closed = False
    last_heartbeat = start_time

    while True:
        try:
            item = output_queue.get(timeout=1.0)
            if item is None:
                stream_closed = True
            else:
                print(item, flush=True)
        except queue.Empty:
            pass

        now = time.monotonic()
        if heartbeat_seconds > 0 and process.poll() is None and now - last_heartbeat >= heartbeat_seconds:
            print(
                f"[heartbeat] {stage_name} still running | elapsed={_format_duration(now - start_time)}",
                flush=True,
            )
            last_heartbeat = now

        if stream_closed and process.poll() is not None:
            break

    return_code = process.wait()
    reader.join(timeout=1.0)
    elapsed = _format_duration(time.monotonic() - start_time)
    print(f"=== Stage complete: {stage_name} | duration={elapsed} ===", flush=True)

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _build_lightweight_archive(output_dir: Path, destination: Path) -> Path:
    if destination.exists():
        destination.unlink()

    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in sorted(output_dir.iterdir()):
            if not candidate.is_file():
                continue
            # Avoid creating nested archives of previous zip outputs.
            if candidate.suffix == ".zip":
                continue
            archive.write(candidate, arcname=candidate.name)

    return destination


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
    parser.add_argument("--disable-yolo", action="store_true", help="Disable YOLO semantic feature extraction")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-device", type=str, default="cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision for autoencoder training")
    parser.add_argument("--multi-gpu", action="store_true", help="Use all visible GPUs for autoencoder training")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers for autoencoder")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints/autoencoder",
        help="Checkpoint directory for autoencoder",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--max-benchmark-frames", type=int, default=240)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--cross-scene-min-rows", type=int, default=25)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--baseline-max-train", type=int, default=20000)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")
    parser.add_argument("--skip-iforest", action="store_true")
    parser.add_argument("--skip-load-test", action="store_true")
    parser.add_argument("--load-cameras", type=int, default=4)
    parser.add_argument("--load-frames-per-camera", type=int, default=75)
    parser.add_argument("--load-detector-mode", choices=["auto", "yolo", "fallback"], default="fallback")
    parser.add_argument("--skip-feedback-simulation", action="store_true")
    parser.add_argument("--feedback-samples", type=int, default=800)
    parser.add_argument("--feedback-label-noise", type=float, default=0.03)
    parser.add_argument("--skip-quality-gate", action="store_true")
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
    load_test_path = artifacts_dir / "multi_camera_load_test.json"
    feedback_report_path = artifacts_dir / "feedback_simulation.json"
    quality_gate_path = artifacts_dir / "project_readiness.json"

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_path else f"{str(repo_root)}{os.pathsep}{existing_path}"
    env["PYTHONUNBUFFERED"] = "1"

    python_exec = sys.executable

    extract_cmd = [
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
        "--progress-every",
        str(args.progress_every),
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--yolo-weights",
        args.yolo_weights,
        "--yolo-device",
        args.yolo_device,
        "--yolo-conf",
        str(args.yolo_conf),
    ]
    if args.disable_yolo:
        print("[kaggle_train] YOLO semantic extraction disabled", flush=True)
        extract_cmd.append("--disable-yolo")

    if args.skip_extraction:
        if not features_path.exists():
            raise FileNotFoundError(f"--skip-extraction requested but features file is missing: {features_path}")
        print(f"[kaggle_train] Skipping feature extraction and reusing: {features_path}", flush=True)
    else:
        run_command(
            "feature extraction",
            extract_cmd,
            cwd=repo_root,
            env=env,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    autoencoder_cmd = [
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
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
    ]
    if args.amp:
        autoencoder_cmd.append("--amp")
    if args.multi_gpu:
        autoencoder_cmd.append("--multi-gpu")
    if args.resume:
        autoencoder_cmd.append("--resume")

    if args.skip_autoencoder:
        if not autoencoder_path.exists():
            raise FileNotFoundError(f"--skip-autoencoder requested but model is missing: {autoencoder_path}")
        print(f"[kaggle_train] Skipping autoencoder training and reusing: {autoencoder_path}", flush=True)
    else:
        run_command(
            "autoencoder training",
            autoencoder_cmd,
            cwd=repo_root,
            env=env,
            heartbeat_seconds=args.heartbeat_seconds,
        )


    if args.skip_iforest:
        if not iforest_path.exists():
            raise FileNotFoundError(f"--skip-iforest requested but model is missing: {iforest_path}")
        print(f"[kaggle_train] Skipping Isolation Forest training and reusing: {iforest_path}", flush=True)
    else:
        run_command(
            "feature anomaly model training",
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
            heartbeat_seconds=args.heartbeat_seconds,
        )

    run_command(
        "evaluation",
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
            "--cross-scene-min-rows",
            str(args.cross_scene_min_rows),
            "--holdout-ratio",
            str(args.holdout_ratio),
            "--random-seed",
            str(args.random_seed),
            "--baseline-max-train",
            str(args.baseline_max_train),
        ],
        cwd=repo_root,
        env=env,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    if args.run_benchmark:
        run_command(
            "benchmark",
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
            heartbeat_seconds=args.heartbeat_seconds,
        )

    if not args.skip_load_test:
        run_command(
            "multi-camera load validation",
            [
                python_exec,
                "ml/scripts/multi_camera_load_test.py",
                "--output",
                str(load_test_path),
                "--cameras",
                str(args.load_cameras),
                "--frames-per-camera",
                str(args.load_frames_per_camera),
                "--model-device",
                args.device,
                "--detector-mode",
                args.load_detector_mode,
            ],
            cwd=repo_root,
            env=env,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    if not args.skip_feedback_simulation:
        run_command(
            "feedback-loop simulation",
            [
                python_exec,
                "ml/scripts/simulate_feedback_loop.py",
                "--features",
                str(features_path),
                "--ae",
                str(autoencoder_path),
                "--iforest",
                str(iforest_path),
                "--eval-report",
                str(eval_report_path),
                "--output",
                str(feedback_report_path),
                "--device",
                args.device,
                "--feedback-samples",
                str(args.feedback_samples),
                "--label-noise",
                str(args.feedback_label_noise),
            ],
            cwd=repo_root,
            env=env,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    if not args.skip_quality_gate:
        quality_cmd = [
            python_exec,
            "ml/scripts/quality_gate.py",
            "--eval-report",
            str(eval_report_path),
            "--output",
            str(quality_gate_path),
        ]
        if load_test_path.exists():
            quality_cmd.extend(["--load-report", str(load_test_path)])
        if feedback_report_path.exists():
            quality_cmd.extend(["--feedback-report", str(feedback_report_path)])

        run_command(
            "quality gate scoring",
            quality_cmd,
            cwd=repo_root,
            env=env,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_candidates = [
        features_path,
        autoencoder_path,
        iforest_path,
        eval_report_path,
        load_test_path,
        feedback_report_path,
        quality_gate_path,
    ]

    for source in export_candidates:
        if not source.exists():
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)

    benchmark_destination = output_dir / benchmark_path.name
    if benchmark_path.exists():
        shutil.copy2(benchmark_path, benchmark_destination)

    checkpoint_export_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_src = checkpoint_dir.resolve()
        checkpoint_dst = checkpoint_export_dir.resolve()

        # Avoid deleting the source when checkpoints are already under output/checkpoints.
        if checkpoint_src == checkpoint_dst or checkpoint_src.is_relative_to(checkpoint_dst):
            print(
                f"[kaggle_train] Checkpoints already inside output directory, skipping copy: {checkpoint_dir}",
                flush=True,
            )
        else:
            if checkpoint_export_dir.exists():
                shutil.rmtree(checkpoint_export_dir)
            shutil.copytree(checkpoint_dir, checkpoint_export_dir)
    else:
        print(
            f"[kaggle_train] Checkpoint directory not found, skipping export: {checkpoint_dir}",
            flush=True,
        )

    with eval_report_path.open("r", encoding="utf-8") as infile:
        report = json.load(infile)

    summary = {
        "best_model": report.get("best_model"),
        "runtime": report.get("runtime"),
        "training": {
            "amp": args.amp,
            "multi_gpu": args.multi_gpu,
            "num_workers": args.num_workers,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_interval": args.checkpoint_interval,
            "resume": args.resume,
            "heartbeat_seconds": args.heartbeat_seconds,
            "progress_every": args.progress_every,
            "yolo_enabled": not args.disable_yolo,
            "yolo_weights": args.yolo_weights,
            "yolo_device": args.yolo_device,
            "yolo_conf": args.yolo_conf,
            "cross_scene_min_rows": args.cross_scene_min_rows,
            "holdout_ratio": args.holdout_ratio,
            "random_seed": args.random_seed,
            "baseline_max_train": args.baseline_max_train,
            "skip_extraction": args.skip_extraction,
            "skip_autoencoder": args.skip_autoencoder,
            "skip_iforest": args.skip_iforest,
            "load_test_enabled": not args.skip_load_test,
            "load_cameras": args.load_cameras,
            "load_frames_per_camera": args.load_frames_per_camera,
            "load_detector_mode": args.load_detector_mode,
            "feedback_simulation_enabled": not args.skip_feedback_simulation,
            "feedback_samples": args.feedback_samples,
            "feedback_label_noise": args.feedback_label_noise,
            "quality_gate_enabled": not args.skip_quality_gate,
            "checkpoint_files": sorted([p.name for p in checkpoint_dir.glob("*.pt")]) if checkpoint_dir.exists() else [],
        },
        "output_dir": str(output_dir),
        "files": sorted([path.name for path in output_dir.iterdir() if path.is_file()]),
    }

    archive_base = output_dir / "incident_intel_training_outputs"
    archive_zip_path = Path(f"{archive_base}.zip")
    archive_path: str | None = None
    archive_status = "created"

    try:
        if archive_zip_path.exists():
            archive_zip_path.unlink()
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(output_dir))
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise

        archive_status = "skipped_no_space"
        if archive_zip_path.exists():
            archive_zip_path.unlink()

        print(
            "[kaggle_train] Full archive skipped due to limited disk space. "
            "Attempting lightweight archive.",
            flush=True,
        )

        lightweight_archive = output_dir / "incident_intel_training_outputs_light.zip"
        try:
            archive_path = str(_build_lightweight_archive(output_dir=output_dir, destination=lightweight_archive))
            archive_status = "lightweight_created"
        except OSError as fallback_exc:
            if fallback_exc.errno != errno.ENOSPC:
                raise
            archive_status = "skipped_no_space"
            archive_path = None
            if lightweight_archive.exists():
                lightweight_archive.unlink()
            print("[kaggle_train] Archive skipped: no space left on device.", flush=True)

    summary["archive"] = {"path": archive_path, "status": archive_status}
    summary["files"] = sorted([path.name for path in output_dir.iterdir() if path.is_file()])
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training completed successfully")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    if archive_path:
        print(f"Archive: {archive_path}")
    else:
        print("Archive: skipped (insufficient disk space)")


if __name__ == "__main__":
    main()
