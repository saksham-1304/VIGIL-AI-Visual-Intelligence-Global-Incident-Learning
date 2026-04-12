# Interview Q&A

## Q1. Why combine tracking with anomaly detection?

Tracking stabilizes identity over time, reducing duplicated detections and improving temporal context for anomaly scoring.

## Q2. Why two anomaly models?

Reconstruction captures distribution shift; Isolation Forest captures outliers in tabular feature space. Hybrid fusion increases robustness.

## Q3. How does multimodal reasoning help operations?

Natural language summaries convert raw visual signals into actionable narratives for SOC teams and dispatch workflows.

## Q4. How do you handle latency constraints?

Use frame skipping, lightweight detector variants, asynchronous processing, and model-specific benchmarking with p95 targets.

## Q5. How do you ensure production observability?

Prometheus exposes throughput/latency/incident metrics and Grafana dashboards provide real-time operations visibility.

## Q6. How is retraining handled?

Feature extraction + training + evaluation are orchestrated with Prefect; DVC and MLflow provide reproducibility and traceability.

## Q7. What is your fallback strategy if YOLO is unavailable?

Motion-based fallback maintains degraded but functional alerting for operational continuity and demos.

## Q8. How would you scale to 500 cameras?

Split into ingestion workers, batched GPU inference, stream partitioning (Kafka/Redis Streams), and horizontal auto-scaling with Kubernetes.
