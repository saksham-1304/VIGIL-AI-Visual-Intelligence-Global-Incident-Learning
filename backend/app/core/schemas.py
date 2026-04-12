from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float] = Field(default_factory=list)
    track_id: int | None = None


class IncidentEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    camera_id: str
    event_type: str
    severity: Severity = Severity.medium
    description: str
    detections: list[Detection] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    anomaly_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    title: str
    severity: Severity
    message: str


class StreamStartRequest(BaseModel):
    source: str = "webcam"
    camera_id: str = "cam-01"


class UploadResponse(BaseModel):
    accepted: bool
    filename: str
    detail: str
