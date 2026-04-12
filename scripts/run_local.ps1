$env:PYTHONPATH = "$PWD;$PWD\backend"

Write-Host "Starting backend on http://localhost:8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; . .\.venv\Scripts\Activate.ps1; `$env:PYTHONPATH='$env:PYTHONPATH'; uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000"

Write-Host "Starting frontend on http://localhost:3000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm install; npm run dev"

Write-Host "Launching shared infrastructure"
docker compose up -d redis postgres mlflow prefect-server prometheus grafana
