"""Score evaluation images into eval/data/scores.csv. Does not touch data/scores.csv."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from pandas.errors import EmptyDataError

from eval.metrics import resolve_image_path
from eval.paths import (
    EVAL_SCORES_CSV,
    FILTERED_METADATA_CSV,
    HUMAN_LABELS_CSV,
    SYNTHETIC_METADATA_CSV,
)
from src.models import ScoreSummary
from src.score import score_image

SCORE_COLUMNS = [
    "uuid",
    "pedestrian_shade_score",
    "shade_sources",
    "confidence",
    "reasoning",
    "scored_at",
]


def load_eval_scores() -> pd.DataFrame:
    if not EVAL_SCORES_CSV.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    try:
        return pd.read_csv(EVAL_SCORES_CSV)
    except EmptyDataError:
        return pd.DataFrame(columns=SCORE_COLUMNS)


def _load_metadata_index() -> dict[str, pd.Series]:
    frames: list[pd.DataFrame] = []
    if FILTERED_METADATA_CSV.exists():
        frames.append(pd.read_csv(FILTERED_METADATA_CSV))
    if SYNTHETIC_METADATA_CSV.exists():
        frames.append(pd.read_csv(SYNTHETIC_METADATA_CSV))
    if not frames:
        return {}
    metadata = pd.concat(frames, ignore_index=True)
    return {str(row["uuid"]): row for _, row in metadata.iterrows()}


def score_eval_uuids(force: bool = False) -> ScoreSummary:
    labels = pd.read_csv(HUMAN_LABELS_CSV)
    metadata_rows = _load_metadata_index()
    existing = load_eval_scores()
    existing_uuids = set(existing["uuid"].astype(str)) if not existing.empty else set()

    scored_count = 0
    skipped_count = 0
    errors: list[str] = []
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    rows_by_uuid = {str(row["uuid"]): row for row in rows}

    for uuid in labels["uuid"].astype(str):
        if not force and uuid in existing_uuids:
            skipped_count += 1
            continue
        if uuid not in metadata_rows:
            errors.append(f"{uuid}: no metadata row")
            continue
        image_path = resolve_image_path(uuid)
        if image_path is None:
            errors.append(f"{uuid}: missing image in sample/ or synthetic/")
            continue
        try:
            result = score_image(image_path, metadata_rows[uuid], retry=True)
        except Exception as exc:
            errors.append(f"{uuid}: {exc}")
            continue
        rows_by_uuid[uuid] = {
            "uuid": uuid,
            "pedestrian_shade_score": result.pedestrian_shade_score,
            "shade_sources": json.dumps(result.shade_sources),
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        scored_count += 1
        print(f"{uuid}: {result.pedestrian_shade_score:.2f}")

    EVAL_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_by_uuid.values(), columns=SCORE_COLUMNS).to_csv(EVAL_SCORES_CSV, index=False)
    print(f"Wrote {len(rows_by_uuid)} rows to {EVAL_SCORES_CSV}")
    return ScoreSummary(scored=scored_count, skipped=skipped_count, errors=errors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score eval images into eval/data/scores.csv")
    parser.add_argument("--force", action="store_true", help="Re-score all uuids in human_labels.csv")
    args = parser.parse_args()
    score_eval_uuids(force=args.force)
