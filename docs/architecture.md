# System Architecture

## High-Level Architecture Diagram

```text
[Video Sources]
  |  (RTSP/Webcam/File Upload)
  v
[Ingestion Layer: FastAPI Upload + Stream Control]
  |
  v
[Vision Pipeline]
  |- YOLOv8 Detection
  |- Centroid Tracking (ByteTrack-like logic)
  |- Action Recognition Heuristics
  v
[Anomaly Intelligence]
  |- Autoencoder Reconstruction Score
  |- Isolation Forest Feature Score
  |- Hybrid Score Fusion
  v
[Multimodal Reasoning]
  |- BLIP Caption (if available)
  |- Prompt-Template Incident Narration
  v
[Incident + Alert Engine]
  |- Rule-based criticality logic
  |- Alert generation and severity tagging
  v
[Persistence + Stream]
  |- SQL (SQLite/PostgreSQL)
  |- WebSocket broadcast
  |- Metrics (/metrics)
  v
[Next.js Dashboard]
  |- Live feed with overlays
  |- Incident timeline
  |- Alert rail
  |- Operations KPIs
```

## Core Design Decisions

1. A modular inference stack keeps each capability independently replaceable (detector, tracker, action, anomaly, explainer).
2. Runtime anomaly scoring combines learned and heuristic signals, improving robustness before full dataset maturity.
3. FastAPI + WebSocket supports low-latency streaming and easy integration with production API gateways.
4. SQL event storage keeps incident audit trails deterministic and query-friendly.
5. Frontend is decoupled from inference runtime via API + WS contracts for scalability.

## Trade-offs

1. Local fallback detector enables portability but has lower semantic accuracy than YOLO.
2. Heuristic action recognition is lightweight but should be replaced with SlowFast/Video Swin for production accuracy.
3. Hybrid anomaly score improves practical reliability but introduces calibration overhead.
4. Single-service API is faster to iterate; microservice decomposition is better for large-scale multi-camera deployments.
