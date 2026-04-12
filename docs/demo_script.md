# Demo Script (10-12 Minutes)

## 1. Problem Framing (1 min)

- Explain need for real-time incident intelligence in smart city and industrial safety environments.

## 2. Architecture Walkthrough (2 min)

- Show flow from live video to detection/tracking/anomaly scoring to natural language explanation.
- Highlight modular design and deployment readiness.

## 3. Live System Demo (4 min)

1. Start stack:
   - `docker compose up -d`
   - `make api`
   - `make frontend`
2. Open dashboard at `http://localhost:3000`.
3. Start stream and trigger sample events (fall motion / wrong-way motion).
4. Show timeline + alert rail + overlay boxes.

## 4. Research Results (2 min)

- Present evaluation and latency benchmark artifacts.
- Discuss ablation impact of tracking and multimodal narration.

## 5. MLOps Story (2 min)

- Show MLflow experiment runs.
- Show Prefect flow and DVC stages.
- Show Grafana operational metrics.

## 6. Closing (1 min)

- Summarize production-readiness and extension paths (true action models, active learning loop, edge deployment).
