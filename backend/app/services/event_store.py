from sqlalchemy import desc, select
from uuid import uuid4

from app.core.schemas import Alert, IncidentEvent
from app.db.models import AlertModel, EventFeedbackModel, IncidentEventModel


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

    def get_event(self, event_id: str) -> IncidentEvent | None:
        with self._session_factory() as db:
            row = db.scalar(select(IncidentEventModel).where(IncidentEventModel.id == event_id))

        if row is None:
            return None

        return IncidentEvent(
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

    def create_feedback(
        self,
        *,
        event_id: str,
        camera_id: str,
        label: str,
        reviewer: str,
        note: str,
        metadata: dict | None = None,
    ) -> dict:
        feedback_id = str(uuid4())
        payload = metadata or {}

        with self._session_factory() as db:
            model = EventFeedbackModel(
                id=feedback_id,
                event_id=event_id,
                camera_id=camera_id,
                label=label,
                reviewer=reviewer,
                note=note,
                metadata_json=payload,
            )
            db.add(model)
            db.commit()
            db.refresh(model)

        return {
            "id": model.id,
            "event_id": model.event_id,
            "camera_id": model.camera_id,
            "label": model.label,
            "reviewer": model.reviewer,
            "note": model.note,
        }

    def list_feedback(self, limit: int = 1000) -> list[dict]:
        with self._session_factory() as db:
            rows = db.scalars(
                select(EventFeedbackModel).order_by(desc(EventFeedbackModel.created_at)).limit(limit)
            ).all()

        return [
            {
                "id": row.id,
                "event_id": row.event_id,
                "camera_id": row.camera_id,
                "label": row.label,
                "reviewer": row.reviewer,
                "note": row.note,
                "created_at": row.created_at,
                "metadata": row.metadata_json,
            }
            for row in rows
        ]
