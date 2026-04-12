from prometheus_client import Counter, Histogram

processed_frames_total = Counter(
    "processed_frames_total",
    "Total number of processed video frames",
    ["camera_id"],
)

incidents_total = Counter(
    "incidents_total",
    "Total number of generated incidents",
    ["camera_id", "severity", "event_type"],
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    buckets=(0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2),
)

feedback_submissions_total = Counter(
    "feedback_submissions_total",
    "Total number of human feedback labels submitted",
    ["label"],
)

model_recalibrations_total = Counter(
    "model_recalibrations_total",
    "Total number of model recalibration requests",
    ["applied"],
)
