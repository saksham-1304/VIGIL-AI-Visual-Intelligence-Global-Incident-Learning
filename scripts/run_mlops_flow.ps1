$env:PYTHONPATH = "$PWD"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

python ml/orchestration/prefect_flow.py
