from fastapi import APIRouter, Query, Request

from app.core.schemas import Alert

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=list[Alert])
def list_alerts(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> list[Alert]:
    return request.app.state.event_store.list_alerts(limit=limit)
