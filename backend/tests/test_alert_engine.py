from app.core.schemas import IncidentEvent, Severity
from app.services.alert_engine import AlertEngine


def test_alert_engine_creates_critical_alert_for_fall() -> None:
    engine = AlertEngine(anomaly_threshold=0.7)
    event = IncidentEvent(
        camera_id="cam-test",
        event_type="behavior",
        severity=Severity.high,
        description="A person appears to have fallen near the entrance",
        actions=["person_falling"],
        anomaly_score=0.83,
    )

    alerts = engine.evaluate(event)
    assert alerts
    assert any(alert.severity in {Severity.high, Severity.critical} for alert in alerts)
