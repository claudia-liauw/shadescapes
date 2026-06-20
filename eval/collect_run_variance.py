"""Score stability sample images multiple times for Tier B. Requires GOOGLE_API_KEY."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.metrics import RUNS_PER_IMAGE
from eval.paths import FILTERED_METADATA_CSV, RUN_VARIANCE_CSV, SAMPLE_IMAGES_DIR, STABILITY_SAMPLE_CSV
from src.models import ShadeScore
from src.score import load_metadata, run_scoring_batch

VARIANCE_COLUMNS = [
    "uuid",
    "run_id",
    "pedestrian_shade_score",
    "confidence",
    "scored_at",
]


def _build_variance_row(key: tuple[str, int], result: ShadeScore) -> dict:
    uuid, run_id = key
    return {
        "uuid": uuid,
        "run_id": run_id,
        "pedestrian_shade_score": result.pedestrian_shade_score,
        "confidence": result.confidence,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_run_variance(runs_per_image: int = RUNS_PER_IMAGE) -> pd.DataFrame:
    started_at = time.perf_counter()
    sample = pd.read_csv(STABILITY_SAMPLE_CSV)
    metadata_rows = {
        str(row["uuid"]): row for _, row in load_metadata(FILTERED_METADATA_CSV).iterrows()
    }
    to_score: list[tuple[tuple[str, int], Path, pd.Series]] = []

    for uuid in sample["uuid"].astype(str):
        image_path = SAMPLE_IMAGES_DIR / f"{uuid}.jpeg"
        if uuid not in metadata_rows:
            print(f"WARNING: no metadata for {uuid}")
            continue
        if not image_path.exists():
            print(f"WARNING: missing image {image_path}")
            continue
        meta_row = metadata_rows[uuid]
        for run_id in range(1, runs_per_image + 1):
            to_score.append(((uuid, run_id), image_path, meta_row))

    new_rows, summary = run_scoring_batch(
        to_score,
        build_row=_build_variance_row,
        started_at=started_at,
        on_scored=lambda key, result: print(
            f"{key[0]} run {key[1]}: {result.pedestrian_shade_score:.2f}"
        ),
    )
    for error in summary.errors:
        print(f"WARNING: {error}")

    df = pd.DataFrame(sorted(new_rows, key=lambda row: (row["uuid"], row["run_id"])), columns=VARIANCE_COLUMNS)
    RUN_VARIANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUN_VARIANCE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {RUN_VARIANCE_CSV} in {summary.elapsed_seconds:.1f}s")
    return df


if __name__ == "__main__":
    collect_run_variance()
