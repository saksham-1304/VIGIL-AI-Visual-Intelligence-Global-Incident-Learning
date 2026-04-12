from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile, status

from app.core.schemas import UploadResponse

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])
UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = "upload-cam",
) -> UploadResponse:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    destination = UPLOAD_DIR / f"{uuid4()}{suffix}"

    with destination.open("wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    background_tasks.add_task(
        request.app.state.stream_processor.process_uploaded_video,
        str(destination),
        camera_id,
    )

    return UploadResponse(
        accepted=True,
        filename=destination.name,
        detail="Video accepted for async processing",
    )
