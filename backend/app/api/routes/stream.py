from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.core.schemas import StreamStartRequest

router = APIRouter(tags=["stream"])


@router.post("/api/v1/stream/start")
def start_stream(request: Request, payload: StreamStartRequest) -> dict:
    started = request.app.state.stream_processor.start_stream(payload.source, payload.camera_id)
    return {"started": started, "source": payload.source, "camera_id": payload.camera_id}


@router.post("/api/v1/stream/stop")
def stop_stream(request: Request) -> dict:
    stopped = request.app.state.stream_processor.stop_stream()
    return {"stopped": stopped}


@router.websocket("/ws/events")
async def events_ws(websocket: WebSocket) -> None:
    manager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "status", "payload": {"connected": True}})
        while True:
            message = await websocket.receive_text()
            if message.lower() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
