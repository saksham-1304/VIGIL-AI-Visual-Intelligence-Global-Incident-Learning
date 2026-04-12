param(
    [switch]$WithDocker
)

python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]

if ($WithDocker) {
    docker compose up -d postgres redis mlflow prefect-server prometheus grafana
}

Write-Host "Environment bootstrap completed."
