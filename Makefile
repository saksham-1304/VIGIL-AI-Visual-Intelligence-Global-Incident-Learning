PYTHONPATH := $(PWD):$(PWD)/backend

.PHONY: install test api frontend mlflow flow benchmark dvc

install:
	pip install -r requirements.txt

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev

test:
	PYTHONPATH=$(PYTHONPATH) pytest backend/tests -q

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

flow:
	PYTHONPATH=$(PWD) python ml/orchestration/prefect_flow.py

benchmark:
	PYTHONPATH=$(PWD) python ml/scripts/benchmark_pipeline.py --input webcam --output artifacts/latency_benchmark.json

dvc:
	dvc repro
