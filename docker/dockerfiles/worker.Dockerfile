FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY ml/requirements.txt /tmp/ml-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ml-requirements.txt

COPY ml /workspace/ml
COPY data /workspace/data
COPY models /workspace/models
COPY artifacts /workspace/artifacts

ENV PYTHONPATH=/workspace

CMD ["python", "ml/orchestration/prefect_flow.py"]
