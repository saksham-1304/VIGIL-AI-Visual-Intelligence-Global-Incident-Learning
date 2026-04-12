from app.core.schemas import Alert, IncidentEvent, Severity


class AlertEngine:
    def __init__(self, anomaly_threshold: float = 0.75) -> None:
        self.anomaly_threshold = anomaly_threshold

    def evaluate(self, event: IncidentEvent) -> list[Alert]:
        alerts: list[Alert] = []

        if event.anomaly_score is not None and event.anomaly_score >= self.anomaly_threshold:
            alerts.append(
                Alert(
                    event_id=event.id,
                    title="High Anomaly Score",
                    severity=Severity.high,
                    message=f"Anomaly score reached {event.anomaly_score:.2f} on {event.camera_id}",
                )
            )

        fall_detected = any("fall" in action.lower() for action in event.actions)
        wrong_way = "wrong_way" in event.event_type.lower() or "against traffic" in event.description.lower()

        if fall_detected:
            alerts.append(
                Alert(
                    event_id=event.id,
                    title="Potential Fall Detected",
                    severity=Severity.critical,
                    message=event.description,
                )
            )

        if wrong_way:
            alerts.append(
                Alert(
                    event_id=event.id,
                    title="Wrong-Way Movement",
                    severity=Severity.high,
                    message=event.description,
                )
            )

        if event.severity in {Severity.high, Severity.critical} and not alerts:
            alerts.append(
                Alert(
                    event_id=event.id,
                    title="High Priority Incident",
                    severity=event.severity,
                    message=event.description,
                )
            )

        return alerts
