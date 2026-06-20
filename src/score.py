import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError
from google import genai
from PIL import Image
from pydantic import ValidationError

from src import config
from src.models import MissingApiKeyError, NoImagesError, NoMetadataError, ScoreSummary, ShadeScore

IMAGES_DIR = config.IMAGES_DIR
METADATA_CSV = config.METADATA_CSV
SCORES_CSV = config.SCORES_CSV
BATCH_SIZE = 15
SCORE_COLUMNS = [
    "uuid",
    "pedestrian_shade_score",
    "shade_sources",
    "confidence",
    "reasoning",
    "scored_at",
]


def discover_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    return [path.name for path in sorted(IMAGES_DIR.glob("*.jpeg"))]


def _format_context_line(heading: float | None = None, hour: float | None = None) -> str:
    if heading is not None and not pd.isna(heading):
        heading_text = f"{heading:.0f} degrees"
    else:
        heading_text = "not provided"
    if hour is not None and not pd.isna(hour):
        try:
            hour_int = int(float(hour))
        except (ValueError, TypeError):
            hour_int = None
    else:
        hour_int = None
    hour_text = f"{hour_int}:00 local time" if hour_int is not None else "not provided"
    return (
        "Context: tropical Singapore; "
        f"heading {heading_text}; "
        f"hour {hour_text}.\n"
    )


def build_prompt(heading: float | None = None, hour: float | None = None) -> str:
    context_line = _format_context_line(heading, hour)
    return (
        "You are assessing pedestrian shade on the sidewalk in this street-level photo.\n"
        f"{context_line}"
        "Score how shaded the walkable pedestrian path is, not overall scene aesthetics.\n"
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
    prompt = build_prompt(metadata_row.get("heading"), metadata_row.get("hour"))
    raw = _call_gemini(image_path, prompt)
    try:
        return parse_vlm_response(raw)
    except (json.JSONDecodeError, ValidationError):
        if not retry:
            raise
        retry_prompt = prompt + "\nReturn valid JSON only. No prose outside the JSON object."
        raw = _call_gemini(image_path, retry_prompt)
        return parse_vlm_response(raw)


def load_metadata(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or METADATA_CSV
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def load_existing_scores(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or SCORES_CSV
    if not path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=SCORE_COLUMNS)


def _write_scores(df: pd.DataFrame) -> None:
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)


def _score_task(
    key: Any, image_path: Path, metadata_row: pd.Series
) -> tuple[Any, ShadeScore | None, str | None]:
    try:
        return key, score_image(image_path, metadata_row, retry=True), None
    except Exception as exc:
        return key, None, str(exc)


def _format_batch_error(key: Any, error: str) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        uuid, run_id = key
        return f"{uuid} run {run_id}: {error}"
    return f"{key}: {error}"


def run_scoring_batch(
    to_score: list[tuple[Any, Path, pd.Series]],
    *,
    build_row: Callable[[Any, ShadeScore], dict],
    require_api_key: bool = True,
    skipped_count: int = 0,
    skip_reasons: dict[str, int] | None = None,
    skips: list[str] | None = None,
    pre_errors: list[str] | None = None,
    started_at: float | None = None,
    on_scored: Callable[[Any, ShadeScore], None] | None = None,
) -> tuple[list[dict], ScoreSummary]:
    started = started_at if started_at is not None else time.perf_counter()
    if require_api_key and not config.get_google_api_key():
        raise MissingApiKeyError("GOOGLE_API_KEY is not configured")

    rows: list[dict] = []
    errors: list[str] = list(pre_errors or [])

    if to_score:
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [
                executor.submit(_score_task, key, image_path, metadata_row)
                for key, image_path, metadata_row in to_score
            ]
            for future in as_completed(futures):
                key, result, error = future.result()
                if error is not None:
                    errors.append(_format_batch_error(key, error))
                    continue
                rows.append(build_row(key, result))
                if on_scored is not None:
                    on_scored(key, result)

    elapsed_seconds = round(time.perf_counter() - started, 1)
    summary = ScoreSummary(
        scored=len(rows),
        skipped=skipped_count,
        skip_reasons={key: count for key, count in (skip_reasons or {}).items() if count},
        skips=skips or [],
        errors=errors,
        elapsed_seconds=elapsed_seconds,
    )
    return rows, summary


def build_score_row(uuid: str, result: ShadeScore) -> dict:
    return {
        "uuid": uuid,
        "pedestrian_shade_score": result.pedestrian_shade_score,
        "shade_sources": json.dumps(result.shade_sources),
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def run_scoring(force: bool = False) -> ScoreSummary:
    started_at = time.perf_counter()
    if not config.get_google_api_key():
        raise MissingApiKeyError("GOOGLE_API_KEY is not configured")

    images = discover_images()
    if not images:
        raise NoImagesError(f"No images found in {IMAGES_DIR}")

    if not METADATA_CSV.exists():
        raise NoMetadataError(f"{METADATA_CSV} not found")

    metadata_rows = {
        str(row["uuid"]): row for _, row in load_metadata().iterrows()
    }
    existing = load_existing_scores()
    existing_uuids = set(existing["uuid"].astype(str)) if not existing.empty else set()

    skipped_count = 0
    skip_reasons: dict[str, int] = {"already_scored": 0, "missing_metadata": 0}
    skips: list[str] = []
    rows_by_uuid = {str(row["uuid"]): row for row in existing.to_dict("records")} if not existing.empty else {}
    to_score: list[tuple[str, Path, pd.Series]] = []

    for image_path in [IMAGES_DIR / name for name in images]:
        uuid = image_path.stem
        if uuid not in metadata_rows:
            skipped_count += 1
            skip_reasons["missing_metadata"] += 1
            skips.append(f"{uuid}: no metadata row in filtered_streetscapes.csv")
            continue
        if not force and uuid in existing_uuids:
            skipped_count += 1
            skip_reasons["already_scored"] += 1
            skips.append(f"{uuid}: already scored")
            continue
        to_score.append((uuid, image_path, metadata_rows[uuid]))

    new_rows, summary = run_scoring_batch(
        to_score,
        build_row=build_score_row,
        require_api_key=False,
        skipped_count=skipped_count,
        skip_reasons=skip_reasons,
        skips=skips,
        started_at=started_at,
    )
    for row in new_rows:
        rows_by_uuid[row["uuid"]] = row
    _write_scores(pd.DataFrame(rows_by_uuid.values(), columns=SCORE_COLUMNS))
    return summary
