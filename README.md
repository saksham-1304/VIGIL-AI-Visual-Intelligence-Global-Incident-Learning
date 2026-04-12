# Real-Time Multimodal Incident Intelligence System

Production-grade, research-oriented platform for live incident understanding from video streams.

## What This System Does

1. Ingests live video from webcam, RTSP-like feeds, or uploaded files.
2. Detects and tracks objects in real time.
3. Infers action cues and computes hybrid anomaly scores.
4. Generates natural language incident narratives.
5. Stores structured events and alerts.
6. Serves a modern operations dashboard with overlays and timeline.
7. Ships with Docker, MLOps, observability, and CI/CD.

## Core Stack

- Computer Vision: YOLOv8 (with motion fallback), centroid tracker, rule-based action recognizer.
- Anomaly: Autoencoder + Isolation Forest + hybrid runtime scorer.
- Multimodal: BLIP captioning (fallback template mode) + prompt-structured explanation generation.
- Backend: FastAPI, SQLAlchemy, WebSocket, Prometheus metrics.
- Frontend: Next.js + TypeScript dashboard.
- MLOps: DVC, MLflow, Prefect.
- Monitoring: Prometheus + Grafana.
- Deployment: Docker Compose + Kubernetes manifests.

## Architecture (Text)

```text
Video Source -> Ingestion API -> Detection -> Tracking -> Action Understanding
-> Hybrid Anomaly Scoring -> Multimodal Explanation -> Event/Alert Store
-> WebSocket + REST -> Dashboard + Monitoring + Research Artifacts
```

Detailed architecture: [docs/architecture.md](docs/architecture.md)

## Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── db/
│   │   ├── services/
│   │   └── ws/
│   └── tests/
├── ml/
│   ├── anomaly/
│   ├── multimodal/
│   ├── vision/
│   ├── scripts/
│   └── orchestration/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── docker/dockerfiles/
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── infra/k8s/
├── scripts/
├── docs/
├── dvc.yaml
├── params.yaml
└── docker-compose.yml
```

## Local Setup

### Prerequisites

1. Python 3.10+
2. Node.js 20+
3. Docker + Docker Compose
4. Optional GPU + CUDA for faster inference

### Option A: PowerShell Bootstrap

```powershell
./scripts/bootstrap.ps1 -WithDocker
```

### Option B: Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Run Services

```bash
docker compose up -d redis postgres mlflow prefect-server prometheus grafana
```

```bash
PYTHONPATH="$PWD:$PWD/backend" uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

```bash
cd frontend && npm run dev
```

Dashboard: [http://localhost:3000](http://localhost:3000)
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
MLflow: [http://localhost:5000](http://localhost:5000)
Prefect: [http://localhost:4200](http://localhost:4200)
Prometheus: [http://localhost:9090](http://localhost:9090)
Grafana: [http://localhost:3001](http://localhost:3001)

## API Endpoints

- GET /api/v1/health
- POST /api/v1/ingest/upload
- POST /api/v1/stream/start
- POST /api/v1/stream/stop
- GET /api/v1/events
- GET /api/v1/alerts
- GET /api/v1/model/status
- GET /api/v1/model/calibration
- GET /api/v1/model/drift
- POST /api/v1/model/feedback
- POST /api/v1/model/recalibrate
- GET /metrics
- WS /ws/events

## Research Pipeline

### 1. Feature Extraction

```bash
python ml/scripts/extract_features.py --input data/raw --output data/processed/features.csv
```

This stage now supports YOLO semantic feature fusion (object/person/vehicle/risk channels) with fallback behavior when YOLO is unavailable.

### 2. Train Two Anomaly Models

```bash
python ml/scripts/train_autoencoder.py --features data/processed/features.csv --output models/autoencoder.pt
python ml/scripts/train_feature_anomaly.py --features data/processed/features.csv --output models/isolation_forest.joblib
```

### 3. Evaluate + Compare

```bash
python ml/scripts/evaluate_anomaly.py --features data/processed/features.csv --ae models/autoencoder.pt --iforest models/isolation_forest.joblib --report artifacts/eval_report.json
```

Evaluation report now includes:

1. PR-AUC and ROC-AUC for each model.
2. Automatic threshold calibration by maximizing F1.
3. Ablation table: `autoencoder`, `isolation_forest`, `hybrid`, `yolo_semantic`, `hybrid_yolo_fusion`.
4. Per-class incident recall and cross-scene diagnostics.
5. Reference score distribution for production drift monitoring.

### 4. Latency Benchmark + Ablation Inputs

```bash
python ml/scripts/benchmark_pipeline.py --input webcam --output artifacts/latency_benchmark.json --max-frames 180
```

## DVC Pipeline

```bash
dvc repro
```

This executes extraction, training, and evaluation stages declared in [dvc.yaml](dvc.yaml).

## Prefect Orchestration

```bash
python ml/orchestration/prefect_flow.py
```

This automates ingestion, training, evaluation, and benchmarking.

## Kaggle GPU Training

If you want to train on Kaggle GPU and then continue development in GitHub, use the dedicated guide:

- [docs/kaggle_gpu_training.md](docs/kaggle_gpu_training.md)
- [notebooks/kaggle_gpu_training.ipynb](notebooks/kaggle_gpu_training.ipynb)

Quick command once inside Kaggle notebook and repo root:

python scripts/kaggle_train.py --input-dir /kaggle/input/ucf-crime-dataset --output-dir /kaggle/working/incident-intel-output --device auto --epochs 40 --latent-dim 64 --batch-size 128 --max-images 300000 --load-cameras 4 --load-frames-per-camera 60 --feedback-samples 1000

Optional benchmark in Kaggle (disabled by default): add --run-benchmark

Additional quality-evidence stages now run by default (unless skipped):

1. Multi-camera load validation (`multi_camera_load_test.json`)
2. Feedback-loop threshold simulation (`feedback_simulation.json`)
3. Readiness scoring (`project_readiness.json`)

## MLOps and Monitoring

- MLOps architecture + DAG: [docs/mlops.md](docs/mlops.md)
- Prometheus scrape config: [monitoring/prometheus.yml](monitoring/prometheus.yml)
- Grafana dashboard: [monitoring/grafana/dashboards/incident_intelligence.json](monitoring/grafana/dashboards/incident_intelligence.json)
- Experiment design: [docs/research_plan.md](docs/research_plan.md)

### Human Feedback and Recalibration Loop

1. Submit event-level labels through `POST /api/v1/model/feedback`.
2. Monitor readiness via `GET /api/v1/model/calibration`.
3. Apply threshold updates via `POST /api/v1/model/recalibrate`.
4. Watch score-distribution drift with `GET /api/v1/model/drift`.

### Quality Evidence Reports

Training/evaluation exports now include:

1. `artifacts/eval_report.json`: split-safe ablations, calibrated thresholds, per-class and cross-scene diagnostics.
2. `artifacts/multi_camera_load_test.json`: multi-camera throughput and latency SLO checks.
3. `artifacts/feedback_simulation.json`: before/after threshold recalibration impact.
4. `artifacts/project_readiness.json`: consolidated research/practical scoring gates.

## Deployment

### Docker Compose Full Stack

```bash
docker compose up --build
```

### Kubernetes

```bash
kubectl apply -f infra/k8s/data-services.yaml
kubectl apply -f infra/k8s/api-deployment.yaml
kubectl apply -f infra/k8s/frontend-deployment.yaml
```

### AWS Script (ECR + K8s apply)

```bash
bash scripts/deploy_aws.sh
```

## CI/CD

GitHub Actions workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml):

1. Backend tests.
2. Frontend typecheck/lint/build.

## Presentation Assets

- Demo flow: [docs/demo_script.md](docs/demo_script.md)
- Interview prep: [docs/interview_qa.md](docs/interview_qa.md)

## Why This Project Is Strong

1. Combines CV + temporal cues + multimodal language generation.
2. Includes both product and research tracks.
3. Provides reproducible MLOps lifecycle and production observability.
4. Has deployment-ready infrastructure and quality gates.
