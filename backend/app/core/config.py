from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Real-Time Multimodal Incident Intelligence API"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "sqlite:///./incident_intel.db"
    redis_url: str = "redis://localhost:6379/0"

    model_device: str = "cpu"
    yolo_weights: str = "yolov8n.pt"
    blip_model: str = "Salesforce/blip-image-captioning-base"

    anomaly_threshold: float = 0.72
    min_anomaly_threshold: float = 0.35
    max_anomaly_threshold: float = 0.95
    drift_recent_window: int = 4000
    calibration_report_path: str = "artifacts/eval_report.json"


settings = Settings()
