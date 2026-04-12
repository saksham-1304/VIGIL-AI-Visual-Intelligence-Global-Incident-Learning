FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY ml/requirements.txt /tmp/ml-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt -r /tmp/ml-requirements.txt

COPY backend /workspace/backend
COPY ml /workspace/ml
COPY models /workspace/models

ENV PYTHONPATH=/workspace:/workspace/backend
WORKDIR /workspace/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
