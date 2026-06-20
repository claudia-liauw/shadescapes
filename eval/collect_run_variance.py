"""Score stability sample images multiple times for Tier B. Requires GOOGLE_API_KEY."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from eval.metrics import RUNS_PER_IMAGE
from eval.paths import (
    FILTERED_METADATA_CSV,
    RUN_VARIANCE_CSV,
    SAMPLE_IMAGES_DIR,
    STABILITY_SAMPLE_CSV,
    SYNTHETIC_METADATA_CSV,
)
from src.score import score_image


def _load_metadata_index() -> dict[str, pd.Series]:
    frames: list[pd.DataFrame] = []
    if FILTERED_METADATA_CSV.exists():
        frames.append(pd.read_csv(FILTERED_METADATA_CSV))
    if SYNTHETIC_METADATA_CSV.exists():
        frames.append(pd.read_csv(SYNTHETIC_METADATA_CSV))
    metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {str(row["uuid"]): row for _, row in metadata.iterrows()}


def collect_run_variance(runs_per_image: int = RUNS_PER_IMAGE) -> pd.DataFrame:
    sample = pd.read_csv(STABILITY_SAMPLE_CSV)
    metadata = _load_metadata_index()
    rows: list[dict] = []

    for uuid in sample["uuid"].astype(str):
        image_path = SAMPLE_IMAGES_DIR / f"{uuid}.jpeg"
        if uuid not in metadata:
            print(f"WARNING: no metadata for {uuid}")
            continue
        if not image_path.exists():
            print(f"WARNING: missing image {image_path}")
            continue
        meta_row = metadata[uuid]
        for run_id in range(1, runs_per_image + 1):
            start = time.perf_counter()
            result = score_image(image_path, meta_row, retry=True)
            duration = time.perf_counter() - start
            rows.append(
                {
                    "uuid": uuid,
                    "run_id": run_id,
                    "pedestrian_shade_score": result.pedestrian_shade_score,
                    "confidence": result.confidence,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"{uuid} run {run_id}: {result.pedestrian_shade_score:.2f} ({duration:.1f}s)")

    df = pd.DataFrame(rows)
    RUN_VARIANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUN_VARIANCE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {RUN_VARIANCE_CSV}")
    return df


if __name__ == "__main__":
    collect_run_variance()
