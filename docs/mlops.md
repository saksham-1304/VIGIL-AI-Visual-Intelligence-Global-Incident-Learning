# MLOps Architecture

## MLOps Architecture Diagram

```text
[Raw Video Data]
  -> [Feature Extraction Script]
  -> [DVC-tracked Features]
  -> [Autoencoder Training] -----> [MLflow Run: metrics + artifacts]
  -> [Isolation Forest Training] -> [MLflow Run: metrics + artifacts]
  -> [Evaluation + Benchmark] ----> [artifacts/eval_report.json, latency_benchmark.json]
  -> [Model Registry Folder: /models]
  -> [Runtime API Inference Service]
  -> [Prometheus Metrics]
  -> [Grafana Dashboards]
```

## Pipeline DAG

```text
extract_features
   |\
   | \ 
   |  -> train_iforest
   -> train_autoencoder
          \ 
           -> evaluate
```

Equivalent orchestrated flow is implemented in [ml/orchestration/prefect_flow.py](../ml/orchestration/prefect_flow.py).

## Experiment Tracking (MLflow)

1. Start MLflow: `make mlflow` or Docker Compose service.
2. Set `MLFLOW_TRACKING_URI=http://localhost:5000`.
3. Run training scripts:
   - `python ml/scripts/train_autoencoder.py ...`
   - `python ml/scripts/train_feature_anomaly.py ...`
4. View runs in MLflow UI with logged parameters and metrics JSON artifacts.

## Monitoring Setup

1. Start infra with Docker Compose (`prometheus`, `grafana`, `api`).
2. Prometheus scrapes [monitoring/prometheus.yml](../monitoring/prometheus.yml).
3. Grafana auto-loads [monitoring/grafana/dashboards/incident_intelligence.json](../monitoring/grafana/dashboards/incident_intelligence.json).
4. Open Grafana at `http://localhost:3001` and inspect:
   - Processed frame throughput
   - Incident rate by severity
   - P95 inference latency

## Continuous Training Strategy

1. Trigger retraining based on drift or periodic schedule.
2. Refresh feature dataset using `extract_features.py`.
3. Execute Prefect flow for retraining and evaluation.
4. Promote model if hybrid F1 and latency constraints meet threshold gates.
