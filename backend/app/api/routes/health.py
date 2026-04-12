from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "app": request.app.title,
        "stream_running": request.app.state.stream_processor.running,
        "websocket_clients": request.app.state.ws_manager.active_count,
    }
