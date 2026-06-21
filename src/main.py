from pathlib import Path
import asyncio
import json

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import config
from src.map_builder import build_map, render_map_html
from src.models import MissingApiKeyError, NoImagesError, NoMetadataError
from src.score import run_scoring

app = FastAPI(title="ShadeScapes")
templates = Jinja2Templates(directory=str(config.PROJECT_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(config.PROJECT_ROOT / "static")), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    folium_map = build_map()
    map_html = render_map_html(folium_map)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"map_html": map_html},
    )


@app.post("/api/score")
async def score_images(force: bool = Query(default=False)):
    async def stream_scoring():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(message: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", message))

        def worker() -> None:
            try:
                summary = run_scoring(force=force, on_progress=on_progress)
                loop.call_soon_threadsafe(queue.put_nowait, ("complete", summary))
            except MissingApiKeyError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", 503, str(exc))
                )
            except (NoImagesError, NoMetadataError) as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", 400, str(exc))
                )

        loop.run_in_executor(None, worker)

        while True:
            event = await queue.get()
            if event[0] == "progress":
                yield json.dumps({"type": "progress", "message": event[1]}) + "\n"
            elif event[0] == "complete":
                summary = event[1]
                yield json.dumps({"type": "complete", **summary.model_dump()}) + "\n"
                break
            elif event[0] == "error":
                _, status_code, detail = event
                yield json.dumps(
                    {"type": "error", "status": status_code, "detail": detail}
                ) + "\n"
                break

    return StreamingResponse(stream_scoring(), media_type="application/x-ndjson")


@app.get("/images/{filename}")
def get_image(filename: str):
    if not filename.endswith(".jpeg"):
        raise HTTPException(status_code=404, detail="Image not found")
    image_path = (config.IMAGES_DIR / filename).resolve()
    if not image_path.is_relative_to(config.IMAGES_DIR.resolve()):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")
