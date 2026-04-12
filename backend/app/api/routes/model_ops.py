from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.schemas import (
    DriftStatus,
    EventFeedbackRequest,
    EventFeedbackResponse,
    ModelCalibrationStatus,
    RecalibrateRequest,
    RecalibrateResponse,
)
from app.services.metrics import feedback_submissions_total, model_recalibrations_total

router = APIRouter(prefix="/api/v1/model", tags=["model-ops"])


@router.get("/calibration", response_model=ModelCalibrationStatus)
def calibration_status(
    request: Request,
    min_samples: int = Query(default=20, ge=5, le=500),
) -> ModelCalibrationStatus:
    status = request.app.state.model_ops.calibration_status(request.app.state.event_store, min_samples=min_samples)
    return ModelCalibrationStatus(**status)


@router.get("/drift", response_model=DriftStatus)
def drift_status(request: Request) -> DriftStatus:
    payload = request.app.state.model_ops.drift_status()
    return DriftStatus(**payload)


@router.get("/status")
def model_status(
    request: Request,
    min_samples: int = Query(default=20, ge=5, le=500),
) -> dict:
    calibration = request.app.state.model_ops.calibration_status(request.app.state.event_store, min_samples=min_samples)
    drift = request.app.state.model_ops.drift_status()
    return {
        "calibration": calibration,
        "drift": drift,
    }


@router.post("/feedback", response_model=EventFeedbackResponse)
def submit_feedback(request: Request, payload: EventFeedbackRequest) -> EventFeedbackResponse:
    event = request.app.state.event_store.get_event(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {payload.event_id}")

    request.app.state.event_store.create_feedback(
        event_id=payload.event_id,
        camera_id=event.camera_id,
        label=payload.label.value,
        reviewer=payload.reviewer,
        note=payload.note,
        metadata={
            "event_type": event.event_type,
            "anomaly_score": event.anomaly_score,
        },
    )
    feedback_submissions_total.labels(label=payload.label.value).inc()

    calibration = request.app.state.model_ops.calibration_status(request.app.state.event_store, min_samples=20)
    return EventFeedbackResponse(
        accepted=True,
        event_id=payload.event_id,
        label=payload.label,
        reviewer=payload.reviewer,
        recalibration_recommended=bool(calibration["calibration_ready"]),
    )


@router.post("/recalibrate", response_model=RecalibrateResponse)
def recalibrate(request: Request, payload: RecalibrateRequest) -> RecalibrateResponse:
    result = request.app.state.model_ops.recalibrate(
        request.app.state.event_store,
        min_samples=payload.min_samples,
        apply=payload.apply,
    )
    model_recalibrations_total.labels(applied=str(bool(result.get("applied"))).lower()).inc()
    return RecalibrateResponse(**result)
