#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:$PWD/backend"

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

docker compose up -d redis postgres mlflow prefect-server prometheus grafana

uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

pushd frontend >/dev/null
npm install
npm run dev &
FRONT_PID=$!
popd >/dev/null

trap "kill $API_PID $FRONT_PID" EXIT
wait
