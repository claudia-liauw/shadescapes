"""Score evaluation images into eval/data/scores.csv. Does not touch data/scores.csv."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from eval.metrics import resolve_image_path
from eval.paths import (
    EVAL_SCORES_CSV,
    FILTERED_METADATA_CSV,
    HUMAN_LABELS_CSV,
    SYNTHETIC_METADATA_CSV,
)
from src.models import ScoreSummary
from src.score import SCORE_COLUMNS, build_score_row, load_existing_scores, load_metadata, run_scoring_batch

def _eval_metadata_rows() -> dict[str, pd.Series]:
    frames: list[pd.DataFrame] = []
    for path in (FILTERED_METADATA_CSV, SYNTHETIC_METADATA_CSV):
        metadata = load_metadata(path)
        if not metadata.empty:
            frames.append(metadata)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return {str(row["uuid"]): row for _, row in combined.iterrows()}


def score_eval_uuids(force: bool = False) -> ScoreSummary:
    started_at = time.perf_counter()
    labels = pd.read_csv(HUMAN_LABELS_CSV)
    metadata_rows = _eval_metadata_rows()
    existing = load_existing_scores(EVAL_SCORES_CSV)
    existing_uuids = set(existing["uuid"].astype(str)) if not existing.empty else set()

    skipped_count = 0
    pre_errors: list[str] = []
    rows_by_uuid = {str(row["uuid"]): row for row in existing.to_dict("records")} if not existing.empty else {}
    to_score: list[tuple[str, Path, pd.Series]] = []

    for uuid in labels["uuid"].astype(str):
        if not force and uuid in existing_uuids:
            skipped_count += 1
            continue
        if uuid not in metadata_rows:
            pre_errors.append(f"{uuid}: no metadata row")
            continue
        image_path = resolve_image_path(uuid)
        if image_path is None:
            pre_errors.append(f"{uuid}: missing image in sample/ or synthetic/")
            continue
        to_score.append((uuid, image_path, metadata_rows[uuid]))

    def write_results(rows_by_uuid_out: dict[str, dict]) -> None:
        EVAL_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_by_uuid_out.values(), columns=SCORE_COLUMNS).to_csv(EVAL_SCORES_CSV, index=False)
        print(f"Wrote {len(rows_by_uuid_out)} rows to {EVAL_SCORES_CSV}")

    new_rows, summary = run_scoring_batch(
        to_score,
        build_row=build_score_row,
        skipped_count=skipped_count,
        pre_errors=pre_errors,
        started_at=started_at,
        on_scored=lambda uuid, result: print(f"{uuid}: {result.pedestrian_shade_score:.2f}"),
    )
    for row in new_rows:
        rows_by_uuid[row["uuid"]] = row
    write_results(rows_by_uuid)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score eval images into eval/data/scores.csv")
    parser.add_argument("--force", action="store_true", help="Re-score all uuids in human_labels.csv")
    args = parser.parse_args()
    score_eval_uuids(force=args.force)
