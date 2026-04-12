# Research and Evaluation Plan

## Research Questions

1. Which anomaly approach performs better under sparse incident labels: reconstruction or feature isolation?
2. How much does tracking improve event consistency and false positive reduction?
3. What is the speed-accuracy frontier for detection-only vs full multimodal pipeline?

## Experiments

1. Model Comparison
   - Autoencoder vs Isolation Forest vs Hybrid
   - Metrics: Precision, Recall, F1, ROC-AUC

2. Latency Benchmark
   - Modes: detection_only, detection_tracking, full
   - Metrics: avg latency, P95 latency, FPS

3. Ablation Study
   - Full pipeline
   - Without tracking
   - Without multimodal explanation
   - Without anomaly branch

## Evaluation Outputs

1. `artifacts/eval_report.json`
2. `artifacts/latency_benchmark.json`
3. Optional confusion matrices and per-event qualitative review log

## Suggested Acceptance Gates

1. Hybrid F1 >= 0.72 on validation split
2. Full pipeline P95 latency <= 180ms on target hardware
3. Critical alert precision >= 0.80 on curated incident set
