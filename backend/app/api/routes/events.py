from fastapi import APIRouter, Query, Request

from app.core.schemas import IncidentEvent

router = APIRouter(prefix="/api/v1/events", tags=["events"])


@router.get("", response_model=list[IncidentEvent])
def list_events(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> list[IncidentEvent]:
    return request.app.state.event_store.list_events(limit=limit)
