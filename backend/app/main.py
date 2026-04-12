import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import alerts, events, health, ingest, metrics, stream
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import Base, SessionLocal, engine
from app.services.alert_engine import AlertEngine
from app.services.event_store import EventStore
from app.services.stream_processor import StreamProcessor
from app.ws.manager import WebSocketManager

configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    main_loop = asyncio.get_running_loop()

    ws_manager = WebSocketManager()
    event_store = EventStore(SessionLocal)
    alert_engine = AlertEngine(anomaly_threshold=settings.anomaly_threshold)
    stream_processor = StreamProcessor(
        event_store=event_store,
        alert_engine=alert_engine,
        ws_manager=ws_manager,
        settings=settings,
        event_loop=main_loop,
    )

    app.state.main_loop = main_loop
    app.state.ws_manager = ws_manager
    app.state.event_store = event_store
    app.state.alert_engine = alert_engine
    app.state.stream_processor = stream_processor


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "api_version": "v1",
    }


app.include_router(health.router)
app.include_router(events.router)
app.include_router(alerts.router)
app.include_router(ingest.router)
app.include_router(stream.router)
app.include_router(metrics.router)
