from sqlalchemy import desc, select

from app.core.schemas import Alert, IncidentEvent
from app.db.models import AlertModel, IncidentEventModel


class EventStore:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def create_event(self, event: IncidentEvent) -> IncidentEvent:
        with self._session_factory() as db:
            model = IncidentEventModel(
                id=event.id,
                camera_id=event.camera_id,
                event_type=event.event_type,
                severity=event.severity.value,
                description=event.description,
                detections=[item.model_dump() for item in event.detections],
                actions=event.actions,
                anomaly_score=event.anomaly_score,
                metadata_json=event.metadata,
            )
            db.add(model)
            db.commit()
            db.refresh(model)
        return event

    def list_events(self, limit: int = 100) -> list[IncidentEvent]:
        with self._session_factory() as db:
            rows = db.scalars(
                select(IncidentEventModel)
                .order_by(desc(IncidentEventModel.timestamp))
                .limit(limit)
            ).all()

        return [
            IncidentEvent(
                id=row.id,
                timestamp=row.timestamp,
                camera_id=row.camera_id,
                event_type=row.event_type,
                severity=row.severity,
                description=row.description,
                detections=row.detections,
                actions=row.actions,
                anomaly_score=row.anomaly_score,
                metadata=row.metadata_json,
            )
            for row in rows
        ]

    def create_alert(self, alert: Alert) -> Alert:
        with self._session_factory() as db:
            model = AlertModel(
                id=alert.id,
                event_id=alert.event_id,
                title=alert.title,
                severity=alert.severity.value,
                message=alert.message,
            )
            db.add(model)
            db.commit()
            db.refresh(model)
        return alert

    def list_alerts(self, limit: int = 100) -> list[Alert]:
        with self._session_factory() as db:
            rows = db.scalars(select(AlertModel).order_by(desc(AlertModel.created_at)).limit(limit)).all()

        return [
            Alert(
                id=row.id,
                created_at=row.created_at,
                event_id=row.event_id,
                title=row.title,
                severity=row.severity,
                message=row.message,
            )
            for row in rows
        ]
