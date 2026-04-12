# Research and Evaluation Plan

## Research Questions

1. Which anomaly approach performs better under sparse incident labels: reconstruction or feature isolation?
2. How much lift is obtained from YOLO semantic feature fusion over handcrafted-only signals?
3. How stable are thresholds and performance across different scenes/classes?
4. What is the speed-accuracy frontier for detection-only vs full multimodal pipeline?

## Experiments

1. Model Comparison
   - Autoencoder vs Isolation Forest vs Hybrid vs YOLO Semantic vs Hybrid+YOLO Fusion
   - Classical unsupervised baselines: One-Class SVM and Local Outlier Factor
   - Metrics: Precision, Recall, F1, PR-AUC, ROC-AUC

   Protocol:
   - Use source-aware holdout split when source IDs are present.
   - Calibrate threshold on train split only.
   - Report all primary metrics on holdout test split.

2. Threshold Calibration
   - Grid-search threshold maximizing F1 on labeled validation samples
   - Export calibrated threshold and reference score statistics

3. Cross-Scene Diagnostics
   - Source-wise performance slices using `source` as scene identity
   - Metrics: F1 mean/std, recall mean/std across eligible scenes

4. Latency Benchmark
   - Modes: detection_only, detection_tracking, full
   - Metrics: avg latency, P95 latency, FPS

5. Ablation Study
   - Full pipeline
   - Without tracking
   - Without multimodal explanation
   - Without anomaly branch

## Evaluation Outputs

1. `artifacts/eval_report.json`
2. `artifacts/latency_benchmark.json`
3. Per-class incident recall table
4. Drift baseline statistics for deployment monitoring
5. Optional confusion matrices and per-event qualitative review log

## Suggested Acceptance Gates

1. Best ablation F1 >= 0.75 with PR-AUC >= 0.78 on validation split
2. Full pipeline P95 latency <= 180ms on target hardware
3. Critical alert precision >= 0.80 on curated incident set
4. Cross-scene F1 std <= 0.12 across eligible scene groups
