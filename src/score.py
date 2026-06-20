import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from google import genai
from PIL import Image
from pydantic import ValidationError

from src import config
from src.models import MissingApiKeyError, NoImagesError, ScoreSummary, ShadeScore

IMAGES_DIR = config.IMAGES_DIR
METADATA_CSV = config.METADATA_CSV
SCORES_CSV = config.SCORES_CSV


def discover_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    return [path.name for path in sorted(IMAGES_DIR.glob("*.jpeg"))]


def build_prompt(heading: float | None = None) -> str:
    heading_line = (
        f"The camera heading is {heading:.0f} degrees. Assume tropical afternoon sun from the west/southwest relative to this view.\n"
        if heading is not None and not pd.isna(heading)
        else "Assume tropical afternoon sun from the west/southwest.\n"
    )
    return (
        "You are assessing pedestrian shade on the sidewalk in this street-level photo.\n"
        "Context: Singapore, 2–4pm, hot afternoon. Score how shaded the walkable pedestrian path is, not overall scene aesthetics.\n"
        f"{heading_line}"
        "Respond with JSON only, no markdown, using exactly this schema:\n"
        "{\n"
        '  "pedestrian_shade_score": 0.72,\n'
        '  "shade_sources": ["street_trees", "building_overhang"],\n'
        '  "confidence": "high",\n'
        '  "reasoning": "One or two sentences."\n'
        "}\n"
        "pedestrian_shade_score must be a float from 0.0 (fully exposed) to 1.0 (fully shaded).\n"
        'confidence must be one of: "low", "medium", "high".'
    )


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_vlm_response(raw: str) -> ShadeScore:
    payload = json.loads(_strip_markdown_fence(raw))
    return ShadeScore.model_validate(payload)


def _call_gemini(image_path: Path, prompt: str) -> str:
    client = genai.Client()
    image = Image.open(image_path)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[image, prompt],
    )
    return response.text or ""


def score_image(image_path: Path, metadata_row: pd.Series, retry: bool = True) -> ShadeScore:
    prompt = build_prompt(metadata_row.get("heading"))
    raw = _call_gemini(image_path, prompt)
    try:
        return parse_vlm_response(raw)
    except (json.JSONDecodeError, ValidationError):
        if not retry:
            raise
        retry_prompt = prompt + "\nReturn valid JSON only. No prose outside the JSON object."
        raw = _call_gemini(image_path, retry_prompt)
        return parse_vlm_response(raw)


def load_existing_scores() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "uuid",
                "pedestrian_shade_score",
                "shade_sources",
                "confidence",
                "reasoning",
                "scored_at",
            ]
        )
    try:
        return pd.read_csv(SCORES_CSV)
    except EmptyDataError:
        return pd.DataFrame(
            columns=[
                "uuid",
                "pedestrian_shade_score",
                "shade_sources",
                "confidence",
                "reasoning",
                "scored_at",
            ]
        )


def _write_scores(df: pd.DataFrame) -> None:
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)


def run_scoring(force: bool = False) -> ScoreSummary:
    api_key = config.get_google_api_key()
    if not api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not configured")

    images = discover_images()
    if not images:
        raise NoImagesError(f"No images found in {IMAGES_DIR}")

    metadata = pd.read_csv(METADATA_CSV)
    metadata_by_uuid = metadata.set_index("uuid")
    existing = load_existing_scores()
    existing_uuids = set(existing["uuid"].astype(str)) if not existing.empty else set()

    scored_count = 0
    skipped_count = 0
    errors: list[str] = []
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    rows_by_uuid = {str(row["uuid"]): row for row in rows}

    image_names = images
    image_paths = [IMAGES_DIR / name for name in image_names]

    for image_path in image_paths:
        uuid = image_path.stem
        if uuid not in metadata_by_uuid.index:
            skipped_count += 1
            errors.append(f"{uuid}: no metadata row in filtered_streetscapes.csv")
            continue

        if not force and uuid in existing_uuids:
            skipped_count += 1
            continue

        try:
            result = score_image(image_path, metadata_by_uuid.loc[uuid])
            rows_by_uuid[uuid] = {
                "uuid": uuid,
                "pedestrian_shade_score": result.pedestrian_shade_score,
                "shade_sources": json.dumps(result.shade_sources),
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            scored_count += 1
        except Exception as exc:
            errors.append(f"{uuid}: {exc}")

    _write_scores(pd.DataFrame(rows_by_uuid.values()))
    return ScoreSummary(scored=scored_count, skipped=skipped_count, errors=errors)
