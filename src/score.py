import json
import math
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
METADATA_REQUIRED_COLUMNS = ["uuid", "lat", "lon"]
RATE_LIMIT_MAX_REQUESTS_PER_MINUTE = 15
RATE_LIMIT_PERIOD_SECONDS = 60.0


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

def find_missing_metadata_columns(
    metadata: pd.DataFrame | None = None,
    csv_path: Path | None = None,
) -> list[str]:
    if metadata is None:
        path = csv_path or METADATA_CSV
        try:
            metadata = pd.read_csv(path, nrows=0)
        except FileNotFoundError:
            return []
        except EmptyDataError:
            return list(METADATA_REQUIRED_COLUMNS)
    return [
        column for column in METADATA_REQUIRED_COLUMNS if column not in metadata.columns
    ]


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


def _emit_progress(
    message: str,
    progress_log: list[str],
    on_progress: Callable[[str], None] | None = None,
    *,
    record: bool = True,
) -> None:
    print(message)
    if record:
        progress_log.append(message)
    if on_progress is not None:
        on_progress(message)


def _format_batch_complete_message(
    batch_number: int,
    completed_requests: int,
    total_to_score: int,
    *,
    waiting_seconds: int | None = None,
) -> str:
    message = (
        f"Batch {batch_number} complete "
        f"({completed_requests}/{total_to_score} total processed)"
    )
    if waiting_seconds is not None:
        message += (
            f", rate limit: waiting {waiting_seconds}s before next batch "
            f"({RATE_LIMIT_MAX_REQUESTS_PER_MINUTE}/min max)"
        )
    return message


def _wait_with_countdown(
    batch_number: int,
    completed_requests: int,
    total_to_score: int,
    sleep_time: float,
    progress_log: list[str],
    on_progress: Callable[[str], None] | None,
) -> None:
    end_time = time.perf_counter() + sleep_time
    recorded = False
    while True:
        remaining = max(0, math.ceil(end_time - time.perf_counter()))
        if remaining <= 0:
            break
        _emit_progress(
            _format_batch_complete_message(
                batch_number,
                completed_requests,
                total_to_score,
                waiting_seconds=remaining,
            ),
            progress_log,
            on_progress,
            record=not recorded,
        )
        recorded = True
        time.sleep(min(1.0, end_time - time.perf_counter()))


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
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], ScoreSummary]:
    started = started_at if started_at is not None else time.perf_counter()
    if require_api_key and not config.get_google_api_key():
        raise MissingApiKeyError("GOOGLE_API_KEY is not configured")

    rows: list[dict] = []
    errors: list[str] = list(pre_errors or [])
    progress_log: list[str] = []
    total_to_score = len(to_score)

    if to_score:
        batch_size = RATE_LIMIT_MAX_REQUESTS_PER_MINUTE
        total_batches = (total_to_score + batch_size - 1) // batch_size
        completed_requests = 0
        for batch_index in range(total_batches):
            batch = to_score[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            batch_start = time.perf_counter()
            _emit_progress(
                f"Batch {batch_index + 1}/{total_batches}: scoring {len(batch)} requests "
                f"({completed_requests}/{total_to_score} processed so far)",
                progress_log,
                on_progress,
            )
            with ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(batch))) as executor:
                futures = [
                    executor.submit(_score_task, key, image_path, metadata_row)
                    for key, image_path, metadata_row in batch
                ]
                for future in as_completed(futures):
                    key, result, error = future.result()
                    if error is not None:
                        errors.append(_format_batch_error(key, error))
                    else:
                        rows.append(build_row(key, result))
                        if on_scored is not None:
                            on_scored(key, result)
                    completed_requests += 1
            if batch_index < total_batches - 1:
                elapsed = time.perf_counter() - batch_start
                sleep_time = RATE_LIMIT_PERIOD_SECONDS - elapsed
                if sleep_time > 0:
                    _wait_with_countdown(
                        batch_index + 1,
                        completed_requests,
                        total_to_score,
                        sleep_time,
                        progress_log,
                        on_progress,
                    )
                else:
                    _emit_progress(
                        _format_batch_complete_message(
                            batch_index + 1,
                            completed_requests,
                            total_to_score,
                        ),
                        progress_log,
                        on_progress,
                    )
            else:
                _emit_progress(
                    _format_batch_complete_message(
                        batch_index + 1,
                        completed_requests,
                        total_to_score,
                    ),
                    progress_log,
                    on_progress,
                )

    elapsed_seconds = round(time.perf_counter() - started, 1)
    summary = ScoreSummary(
        scored=len(rows),
        skipped=skipped_count,
        skip_reasons={key: count for key, count in (skip_reasons or {}).items() if count},
        skips=skips or [],
        errors=errors,
        progress=progress_log,
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


def run_scoring(
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> ScoreSummary:
    started_at = time.perf_counter()
    if not config.get_google_api_key():
        raise MissingApiKeyError("GOOGLE_API_KEY is not configured")

    if not METADATA_CSV.exists():
        raise NoMetadataError(f"{config.relative_path(METADATA_CSV)} not found")

    images = discover_images()
    if not images:
        raise NoImagesError(f"No images found in {config.relative_path(IMAGES_DIR)}")

    metadata = load_metadata()
    metadata_path = config.relative_path(METADATA_CSV)
    missing_columns = find_missing_metadata_columns(metadata)
    if missing_columns:
        missing_columns_list = ", ".join(missing_columns)
        raise NoMetadataError(
            f"{metadata_path} missing columns: {missing_columns_list}"
        )
    metadata_rows = {
        str(row["uuid"]): row for _, row in metadata.iterrows()
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
            skips.append(f"{uuid}: no metadata row in {metadata_path}")
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
        on_progress=on_progress,
    )
    for row in new_rows:
        rows_by_uuid[row["uuid"]] = row
    _write_scores(pd.DataFrame(rows_by_uuid.values(), columns=SCORE_COLUMNS))
    return summary
