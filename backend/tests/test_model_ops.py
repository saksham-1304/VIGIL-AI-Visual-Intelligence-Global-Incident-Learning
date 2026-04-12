from fastapi.testclient import TestClient

from app.core.schemas import IncidentEvent, Severity
from app.main import app


def test_model_feedback_calibration_and_drift_endpoints() -> None:
    with TestClient(app) as client:
        store = client.app.state.event_store

        high_event = IncidentEvent(
            camera_id="cam-model-ops",
            event_type="anomaly",
            severity=Severity.high,
            description="Potential robbery-like behavior near exit",
            anomaly_score=0.91,
        )
        low_event = IncidentEvent(
            camera_id="cam-model-ops",
            event_type="behavior",
            severity=Severity.medium,
            description="Normal movement in hallway",
            anomaly_score=0.18,
        )

        store.create_event(high_event)
        store.create_event(low_event)

        feedback_high = client.post(
            "/api/v1/model/feedback",
            json={
                "event_id": high_event.id,
                "label": "incident",
                "reviewer": "qa-reviewer",
                "note": "Confirmed incident",
            },
        )
        feedback_low = client.post(
            "/api/v1/model/feedback",
            json={
                "event_id": low_event.id,
                "label": "not_incident",
                "reviewer": "qa-reviewer",
                "note": "False alarm",
            },
        )

        assert feedback_high.status_code == 200
        assert feedback_low.status_code == 200

        calibration = client.get("/api/v1/model/calibration", params={"min_samples": 5})
        assert calibration.status_code == 200
        calibration_payload = calibration.json()
        assert calibration_payload["feedback_samples"] >= 2
        assert calibration_payload["labeled_event_samples"] >= 2

        recalibrate = client.post(
            "/api/v1/model/recalibrate",
            json={"min_samples": 2, "apply": True},
        )
        assert recalibrate.status_code == 200
        recalibrate_payload = recalibrate.json()
        assert "current_threshold" in recalibrate_payload
        assert recalibrate_payload["labeled_event_samples"] >= 2

        drift = client.get("/api/v1/model/drift")
        assert drift.status_code == 200
        drift_payload = drift.json()
        assert "status" in drift_payload
        assert "recent_count" in drift_payload
