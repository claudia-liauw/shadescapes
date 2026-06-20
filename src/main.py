from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import config
from src.map_builder import build_map
from src.models import MissingApiKeyError, NoImagesError
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
    map_html = folium_map.get_root().render()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"map_html": map_html},
    )


@app.post("/api/score")
def score_images(force: bool = Query(default=False)):
    try:
        summary = run_scoring(force=force)
        return summary.model_dump()
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NoImagesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/images/{filename}")
def get_image(filename: str):
    if not filename.endswith(".jpeg"):
        raise HTTPException(status_code=404, detail="Image not found")
    image_path = config.IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")
