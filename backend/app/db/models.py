from sqlalchemy import JSON, Column, DateTime, Float, String, Text
from sqlalchemy.sql import func

from app.db.database import Base


class IncidentEventModel(Base):
    __tablename__ = "incident_events"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    camera_id = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)
    severity = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=False)
    detections = Column(JSON, nullable=False, default=list)
    actions = Column(JSON, nullable=False, default=list)
    anomaly_score = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)


class AlertModel(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    event_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    severity = Column(String, index=True, nullable=False)
    message = Column(Text, nullable=False)


class EventFeedbackModel(Base):
    __tablename__ = "event_feedback"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    event_id = Column(String, index=True, nullable=False)
    camera_id = Column(String, index=True, nullable=False)
    label = Column(String, index=True, nullable=False)
    reviewer = Column(String, nullable=False, default="operator")
    note = Column(Text, nullable=False, default="")
    metadata_json = Column(JSON, nullable=False, default=dict)
