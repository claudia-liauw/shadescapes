# Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-based evaluation (`eval/evaluation.ipynb`) that scores a curated eval set via `eval/score_eval.py` into `eval/data/scores.csv`, reports Tier A (VLM vs human on real images), Tier B (run stability), and Tier C (mismatch gallery grouped by `scene_category`), with optional synthetic gap-fill images.

**Architecture:** Pure join/metric logic lives in `eval/metrics.py` (unit-tested). Eval scoring uses `src.score.score_image` but reads images from `data/images/sample/` and `data/images/synthetic/` — not `config.IMAGES_DIR` (`exploration/`). Production demo scores stay in `data/scores.csv`. Multi-run stability is collected by `eval/collect_run_variance.py` into `eval/data/run_variance.csv`. Optional `eval/generate_images.py` fills synthetic gap cases from `eval/data/synthetic_prompts.csv`.

**Tech Stack:** Python 3.10, pandas, matplotlib, Pillow, google-genai, IPython/Jupyter (all already in `pyproject.toml`)

**Spec:** [2026-06-19-evaluation-design.md](./2026-06-19-evaluation-design.md)

> **Note:** This plan replaces the previous version entirely. The old plan assumed composite GVI/SVI baseline, `data/scores.csv` for eval, and `exploration/` images — all dropped or changed in the approved design.

---

## File map

| File | Responsibility |
|------|----------------|
| `eval/__init__.py` | Package marker |
| `eval/paths.py` | Eval-specific paths (`eval/data/*.csv`, `sample/`, `synthetic/`) |
| `eval/metrics.py` | Join eval data, Spearman/MAE, mismatch flags, run-variance summaries |
| `eval/score_eval.py` | Score uuids in `eval/data/human_labels.csv` → `eval/data/scores.csv` |
| `eval/collect_run_variance.py` | Score 10 images 3× each from `sample/` → `eval/data/run_variance.csv` |
| `eval/generate_images.py` | Optional: Imagen API → `synthetic/` + `synthetic_streetscapes.csv` |
| `tests/test_eval_metrics.py` | Unit tests for metrics module |
| `tests/test_score_eval.py` | Unit tests for score_eval (mocked API) |
| `eval/data/` | Eval CSVs (labels, scores, variance, prompts) |
| `eval/data/human_labels.csv` | ~30 uuids + `scene_category` + labels (author fills scores) |
| `eval/data/synthetic_prompts.csv` | Optional gap-fill prompts (6–8 rows) |
| `eval/data/stability_sample.csv` | 10 uuids for Tier B |
| `eval/data/scores.csv` | VLM outputs for eval set (separate from production) |
| `eval/data/run_variance.csv` | Raw multi-run VLM scores (30 rows) |
| `data/synthetic_streetscapes.csv` | Metadata for synthetic images |
| `eval/evaluation.ipynb` | Full eval report with saved outputs |
| `README.md` | Eval summary section + link to notebook |

---

### Task 1: Eval paths and metrics module

**Files:**
- Create: `eval/__init__.py` (empty)
- Create: `eval/paths.py`
- Create: `eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `eval/__init__.py` (empty file).

Create `tests/test_eval_metrics.py`:

```python
import pandas as pd
import pytest

from eval.metrics import (
    HIGH_VARIANCE_RANGE,
    MISMATCH_THRESHOLD,
    build_eval_frame,
    compute_method_metrics,
    filter_real_images,
    flag_mismatches,
    load_eval_metadata,
    merge_run_stability,
    normalize_human,
    per_image_run_stats,
    resolve_image_path,
    stability_summary,
    summarize_mismatches_by_scene_category,
)
from eval.paths import SAMPLE_IMAGES_DIR, SYNTHETIC_IMAGES_DIR


def test_normalize_human():
    s = pd.Series([1, 3, 5])
    result = normalize_human(s)
    pd.testing.assert_series_equal(result, pd.Series([0.0, 0.5, 1.0]), check_names=False)


def test_load_eval_metadata_real_only(tmp_path, monkeypatch):
    real_csv = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "uuid": ["a"],
            "source": ["Mapillary"],
            "lat": [1.3],
            "lon": [103.8],
            "hour": [14],
            "heading": [90.0],
            "place": ["campus"],
            "sidewalk_pct": [0.02],
        }
    ).to_csv(real_csv, index=False)
    monkeypatch.setattr("eval.metrics.FILTERED_METADATA_CSV", real_csv)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_METADATA_CSV", tmp_path / "missing.csv")

    meta = load_eval_metadata()
    assert len(meta) == 1
    assert meta.loc[0, "source"] == "mapillary"


def test_load_eval_metadata_concat_real_and_synthetic(tmp_path, monkeypatch):
    real_csv = tmp_path / "filtered.csv"
    syn_csv = tmp_path / "synthetic.csv"
    pd.DataFrame({"uuid": ["a"], "source": ["Mapillary"], "hour": [14], "heading": [90.0], "sidewalk_pct": [0.02], "lat": [1.3], "lon": [103.8], "place": ["campus"]}).to_csv(real_csv, index=False)
    pd.DataFrame({"uuid": ["syn-1"], "source": ["synthetic"], "hour": [14], "heading": [180.0], "sidewalk_pct": [0.35], "lat": [1.31], "lon": [103.81], "place": ["synthetic_linkway"]}).to_csv(syn_csv, index=False)
    monkeypatch.setattr("eval.metrics.FILTERED_METADATA_CSV", real_csv)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_METADATA_CSV", syn_csv)

    meta = load_eval_metadata()
    assert set(meta["uuid"]) == {"a", "syn-1"}
    assert set(meta["source"]) == {"mapillary", "synthetic"}


def test_resolve_image_path_prefers_sample(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    synthetic_dir = tmp_path / "synthetic"
    sample_dir.mkdir()
    synthetic_dir.mkdir()
    (sample_dir / "u1.jpeg").write_bytes(b"x")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", synthetic_dir)

    assert resolve_image_path("u1") == sample_dir / "u1.jpeg"


def test_resolve_image_path_falls_back_to_synthetic(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    synthetic_dir = tmp_path / "synthetic"
    sample_dir.mkdir()
    synthetic_dir.mkdir()
    (synthetic_dir / "syn-1.jpeg").write_bytes(b"x")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", synthetic_dir)

    assert resolve_image_path("syn-1") == synthetic_dir / "syn-1.jpeg"


def test_build_eval_frame_inner_join_and_warnings():
    labels = pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "shade_1to5": [3, 4, 5],
            "scene_category": ["tree_canopy", "open_exposure", "building_shadow"],
            "notes": ["", "", ""],
        }
    )
    scores = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "pedestrian_shade_score": [0.5, 0.8],
            "shade_sources": ['["street_trees"]', '["street_trees"]'],
            "confidence": ["high", "high"],
            "reasoning": ["r1", "r2"],
        }
    )
    metadata = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "source": ["mapillary", "mapillary"],
            "hour": [14, 15],
            "sidewalk_pct": [0.02, 0.05],
            "heading": [90.0, 180.0],
            "lat": [1.3, 1.31],
            "lon": [103.8, 103.81],
            "place": ["campus", "hospital"],
        }
    )
    frame, warnings = build_eval_frame(labels, scores, metadata)
    assert len(frame) == 2
    assert "human_norm" in frame.columns
    assert "err_vlm" in frame.columns
    assert frame.loc[frame["uuid"] == "a", "scene_category"].iloc[0] == "tree_canopy"
    assert any("c" in w for w in warnings)


def test_filter_real_images():
    frame = pd.DataFrame(
        {
            "uuid": ["a", "syn-1"],
            "source": ["mapillary", "synthetic"],
            "human_norm": [0.5, 0.5],
            "pedestrian_shade_score": [0.5, 0.9],
        }
    )
    real = filter_real_images(frame)
    assert list(real["uuid"]) == ["a"]


def test_compute_method_metrics():
    frame = pd.DataFrame(
        {
            "human_norm": [0.0, 0.5, 1.0],
            "pedestrian_shade_score": [0.1, 0.5, 0.9],
        }
    )
    metrics = compute_method_metrics(frame, "pedestrian_shade_score")
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_flag_mismatches_vlm_only():
    frame = pd.DataFrame(
        {
            "uuid": ["x", "y", "z"],
            "human_norm": [0.5, 0.5, 0.5],
            "pedestrian_shade_score": [0.5, 0.9, 0.2],
            "scene_category": ["tree_canopy", "open_exposure", "building_shadow"],
        }
    )
    frame["err_vlm"] = (frame["pedestrian_shade_score"] - frame["human_norm"]).abs()
    mismatches = flag_mismatches(frame, threshold=0.25)
    assert set(mismatches["uuid"]) == {"y", "z"}


def test_summarize_mismatches_by_scene_category():
    mismatches = pd.DataFrame(
        {
            "scene_category": ["tree_canopy", "tree_canopy", "open_exposure"],
        }
    )
    summary = summarize_mismatches_by_scene_category(mismatches)
    assert summary.loc[summary["scene_category"] == "tree_canopy", "count"].iloc[0] == 2
    assert summary.loc[summary["scene_category"] == "open_exposure", "count"].iloc[0] == 1


def test_per_image_run_stats():
    runs = pd.DataFrame(
        {
            "uuid": ["a", "a", "a", "b", "b", "b"],
            "run_id": [1, 2, 3, 1, 2, 3],
            "pedestrian_shade_score": [0.5, 0.6, 0.55, 0.2, 0.8, 0.3],
        }
    )
    stats = per_image_run_stats(runs)
    assert len(stats) == 2
    row_a = stats.loc[stats["uuid"] == "a"].iloc[0]
    assert row_a["score_std"] == pytest.approx(0.05, abs=0.01)
    assert row_a["score_range"] == pytest.approx(0.1, abs=0.01)


def test_stability_summary():
    stats = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "score_std": [0.05, 0.12],
            "score_range": [0.10, 0.20],
        }
    )
    runs = pd.DataFrame(
        {
            "uuid": ["a", "a", "b", "b"],
            "run_id": [1, 2, 1, 2],
            "pedestrian_shade_score": [0.5, 0.6, 0.2, 0.3],
        }
    )
    summary = stability_summary(stats, runs)
    assert summary["n_images"] == 2
    assert summary["median_std"] == pytest.approx(0.085, abs=0.01)
    assert summary["pct_high_variance"] == pytest.approx(50.0)
    assert summary["test_retest_spearman"] == pytest.approx(1.0)


def test_merge_run_stability():
    frame = pd.DataFrame({"uuid": ["a", "b"], "scene_category": ["tree_canopy", "open_exposure"]})
    stats = pd.DataFrame(
        {
            "uuid": ["a"],
            "score_std": [0.05],
            "score_range": [0.10],
        }
    )
    merged = merge_run_stability(frame, stats)
    assert merged.loc[merged["uuid"] == "a", "run_std"].iloc[0] == pytest.approx(0.05)
    assert pd.isna(merged.loc[merged["uuid"] == "b", "run_std"].iloc[0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Users/liauw/Desktop/Sputnik/2025-26/Job/GovTech/shadescapes
uv run pytest tests/test_eval_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Implement `eval/paths.py`**

```python
from pathlib import Path

from src import config

PROJECT_ROOT = config.PROJECT_ROOT
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_DATA_DIR = EVAL_DIR / "data"
HUMAN_LABELS_CSV = EVAL_DATA_DIR / "human_labels.csv"
EVAL_SCORES_CSV = EVAL_DATA_DIR / "scores.csv"
STABILITY_SAMPLE_CSV = EVAL_DATA_DIR / "stability_sample.csv"
RUN_VARIANCE_CSV = EVAL_DATA_DIR / "run_variance.csv"
SYNTHETIC_PROMPTS_CSV = EVAL_DATA_DIR / "synthetic_prompts.csv"

FILTERED_METADATA_CSV = config.METADATA_CSV
SYNTHETIC_METADATA_CSV = config.DATA_DIR / "synthetic_streetscapes.csv"

SAMPLE_IMAGES_DIR = config.DATA_DIR / "images" / "sample"
SYNTHETIC_IMAGES_DIR = config.DATA_DIR / "images" / "synthetic"
```

- [ ] **Step 4: Implement `eval/metrics.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.paths import FILTERED_METADATA_CSV, SAMPLE_IMAGES_DIR, SYNTHETIC_IMAGES_DIR, SYNTHETIC_METADATA_CSV

MISMATCH_THRESHOLD = 0.25
HIGH_VARIANCE_RANGE = 0.15
RUNS_PER_IMAGE = 3
STABILITY_SAMPLE_SIZE = 10


def normalize_human(shade_1to5: pd.Series) -> pd.Series:
    return (shade_1to5.astype(float) - 1) / 4


def load_eval_metadata() -> pd.DataFrame:
    real = pd.read_csv(FILTERED_METADATA_CSV)
    real = real.copy()
    real["source"] = "mapillary"

    if SYNTHETIC_METADATA_CSV.exists():
        synthetic = pd.read_csv(SYNTHETIC_METADATA_CSV)
        synthetic = synthetic.copy()
        synthetic["source"] = synthetic.get("source", "synthetic").fillna("synthetic")
        return pd.concat([real, synthetic], ignore_index=True)

    return real


def resolve_image_path(uuid: str) -> Path | None:
    sample_path = SAMPLE_IMAGES_DIR / f"{uuid}.jpeg"
    if sample_path.exists():
        return sample_path
    synthetic_path = SYNTHETIC_IMAGES_DIR / f"{uuid}.jpeg"
    if synthetic_path.exists():
        return synthetic_path
    return None


def build_eval_frame(
    labels: pd.DataFrame,
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    labeled = labels.dropna(subset=["shade_1to5"]).copy()
    labeled_uuids = set(labeled["uuid"])
    scored_uuids = set(scores["uuid"])
    meta_uuids = set(metadata["uuid"])

    missing_scores = sorted(labeled_uuids - scored_uuids)
    missing_meta = sorted(labeled_uuids - meta_uuids)
    if missing_scores:
        warnings.append(f"Excluded uuids missing from eval/data/scores.csv: {missing_scores}")
    if missing_meta:
        warnings.append(f"Excluded uuids missing from metadata: {missing_meta}")

    frame = (
        labeled.merge(scores, on="uuid", how="inner")
        .merge(metadata, on="uuid", how="inner", suffixes=("", "_meta"))
    )
    frame["human_norm"] = normalize_human(frame["shade_1to5"])
    frame["err_vlm"] = (frame["pedestrian_shade_score"] - frame["human_norm"]).abs()
    return frame, warnings


def filter_real_images(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["source"] == "mapillary"].copy()


def compute_method_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float | int]:
    n = len(frame)
    if n == 0:
        return {"n": 0, "spearman": float("nan"), "mae": float("nan")}
    spearman = frame["human_norm"].corr(frame[score_col], method="spearman")
    mae = (frame[score_col] - frame["human_norm"]).abs().mean()
    return {"n": n, "spearman": float(spearman), "mae": float(mae)}


def flag_mismatches(frame: pd.DataFrame, threshold: float = MISMATCH_THRESHOLD) -> pd.DataFrame:
    flagged = frame[frame["err_vlm"] > threshold].copy()
    if flagged.empty:
        return pd.DataFrame(columns=list(frame.columns))
    return flagged


def summarize_mismatches_by_scene_category(mismatches: pd.DataFrame) -> pd.DataFrame:
    if mismatches.empty:
        return pd.DataFrame(columns=["scene_category", "count"])
    return (
        mismatches.groupby("scene_category")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def per_image_run_stats(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby("uuid")["pedestrian_shade_score"]
    return pd.DataFrame(
        {
            "uuid": grouped.mean().index,
            "score_mean": grouped.mean().values,
            "score_std": grouped.std(ddof=0).values,
            "score_range": (grouped.max() - grouped.min()).values,
            "n_runs": grouped.count().values,
        }
    )


def stability_summary(stats: pd.DataFrame, runs: pd.DataFrame) -> dict[str, float | int]:
    if stats.empty:
        return {
            "n_images": 0,
            "median_std": float("nan"),
            "max_range": float("nan"),
            "pct_high_variance": float("nan"),
            "test_retest_spearman": float("nan"),
        }
    run1 = runs[runs["run_id"] == 1].set_index("uuid")["pedestrian_shade_score"]
    run2 = runs[runs["run_id"] == 2].set_index("uuid")["pedestrian_shade_score"]
    common = run1.index.intersection(run2.index)
    test_retest = (
        float(run1.loc[common].corr(run2.loc[common], method="spearman"))
        if len(common) >= 2
        else float("nan")
    )
    return {
        "n_images": len(stats),
        "median_std": float(stats["score_std"].median()),
        "max_range": float(stats["score_range"].max()),
        "pct_high_variance": float((stats["score_range"] > HIGH_VARIANCE_RANGE).mean() * 100),
        "test_retest_spearman": test_retest,
    }


def merge_run_stability(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        out = frame.copy()
        out["run_std"] = float("nan")
        out["run_range"] = float("nan")
        return out
    merged = frame.merge(stats[["uuid", "score_std", "score_range"]], on="uuid", how="left")
    return merged.rename(columns={"score_std": "run_std", "score_range": "run_range"})
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_eval_metrics.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/paths.py eval/metrics.py tests/test_eval_metrics.py
git commit -m "$(cat <<'EOF'
feat(eval): add metrics module for VLM vs human evaluation

Pure join/metric helpers for Tier A/B/C without composite baseline.
EOF
)"
```

---

### Task 2: Eval scoring script (`eval/score_eval.py`)

**Files:**
- Create: `eval/score_eval.py`
- Test: `tests/test_score_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_score_eval.py`:

```python
import pandas as pd
from unittest.mock import patch

from eval.score_eval import load_eval_scores, score_eval_uuids
from src.models import ShadeScore


def test_load_eval_scores_empty(tmp_path, monkeypatch):
    scores_csv = tmp_path / "scores.csv"
    monkeypatch.setattr("eval.score_eval.EVAL_SCORES_CSV", scores_csv)
    df = load_eval_scores()
    assert list(df.columns) == [
        "uuid",
        "pedestrian_shade_score",
        "shade_sources",
        "confidence",
        "reasoning",
        "scored_at",
    ]
    assert df.empty


@patch("eval.score_eval.score_image")
def test_score_eval_uuids_writes_eval_scores(mock_score_image, tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    data_dir = tmp_path / "data"
    sample_dir = data_dir / "images" / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "u1.jpeg").write_bytes(b"fake")

    labels_csv = eval_dir / "human_labels.csv"
    scores_csv = eval_dir / "scores.csv"
    meta_csv = data_dir / "filtered_streetscapes.csv"
    eval_dir.mkdir()
    pd.DataFrame({"uuid": ["u1"], "shade_1to5": [3], "scene_category": ["tree_canopy"], "notes": [""]}).to_csv(labels_csv, index=False)
    pd.DataFrame(
        {
            "uuid": ["u1"],
            "source": ["Mapillary"],
            "orig_id": [""],
            "lat": [1.3],
            "lon": [103.8],
            "hour": [14],
            "heading": [90.0],
            "place": ["campus"],
            "sidewalk_pct": [0.02],
        }
    ).to_csv(meta_csv, index=False)

    monkeypatch.setattr("eval.score_eval.HUMAN_LABELS_CSV", labels_csv)
    monkeypatch.setattr("eval.score_eval.EVAL_SCORES_CSV", scores_csv)
    monkeypatch.setattr("eval.score_eval.FILTERED_METADATA_CSV", meta_csv)
    monkeypatch.setattr("eval.score_eval.SYNTHETIC_METADATA_CSV", data_dir / "synthetic_streetscapes.csv")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", data_dir / "images" / "synthetic")

    mock_score_image.return_value = ShadeScore(
        pedestrian_shade_score=0.6,
        shade_sources=["street_trees"],
        confidence="high",
        reasoning="test",
    )

    summary = score_eval_uuids(force=True)
    assert summary.scored == 1
    out = pd.read_csv(scores_csv)
    assert len(out) == 1
    assert out.loc[0, "uuid"] == "u1"
    assert out.loc[0, "pedestrian_shade_score"] == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_score_eval.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'eval.score_eval'`

- [ ] **Step 3: Implement `eval/score_eval.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_score_eval.py tests/test_eval_metrics.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add eval/score_eval.py tests/test_score_eval.py
git commit -m "$(cat <<'EOF'
feat(eval): add score_eval script writing to eval/data/scores.csv

Scores human_labels uuids from sample/ and synthetic/ without touching production scores.
EOF
)"
```

---

### Task 3: Human labels scaffold

**Files:**
- Create: `eval/data/` (directory for eval CSVs)
- Create: `eval/data/human_labels.csv`

- [ ] **Step 1: Generate ~30 diverse uuids from `sample/` pool**

Only include uuids that have a `.jpeg` in `data/images/sample/` and a row in `filtered_streetscapes.csv`. Spread across `place` values (max 2 per place):

```bash
uv run python -c "
import pandas as pd
from pathlib import Path

meta = pd.read_csv('data/filtered_streetscapes.csv')
sample_dir = Path('data/images/sample')
available = {p.stem for p in sample_dir.glob('*.jpeg')}
meta = meta[meta['uuid'].isin(available)].copy()

rows = []
for place, group in meta.groupby('place'):
    for uuid in group['uuid'].head(2):
        rows.append({'uuid': uuid, 'shade_1to5': '', 'scene_category': '', 'notes': ''})
        if len(rows) >= 30:
            break
    if len(rows) >= 30:
        break

if len(rows) < 30:
    seen = {r['uuid'] for r in rows}
    for uuid in meta['uuid']:
        if uuid not in seen:
            rows.append({'uuid': uuid, 'shade_1to5': '', 'scene_category': '', 'notes': ''})
        if len(rows) >= 30:
            break

out = Path('eval/data/human_labels.csv')
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f'Wrote {len(rows)} rows to {out}')
print(meta[meta['uuid'].isin([r[\"uuid\"] for r in rows])]['place'].value_counts().head(10))
"
```

- [ ] **Step 2: Hand-label `shade_1to5` and `scene_category` (author task)**

Open images in `data/images/sample/{uuid}.jpeg` (not `exploration/`).

For each row fill:
- `shade_1to5` (1–5) per rubric in design spec
- `scene_category` — one of: `tree_canopy`, `building_shadow`, `covered_walkway`, `open_exposure`, `mixed_sources`, `ambiguous_path`

Rubric reminder:
- Anchor to 2–4pm tropical sun, sidewalk-level
- Score shade on the **walkable path**, not scene aesthetics
- Ignore clouds unless they materially block sun on the path

- [ ] **Step 3: Score eval images**

```bash
uv run python -m eval.score_eval
```

Expected: `eval/data/scores.csv` with one row per labeled uuid. Requires `GOOGLE_API_KEY` in `.env`.

- [ ] **Step 4: Commit scaffold**

```bash
git add eval/data/human_labels.csv
git commit -m "$(cat <<'EOF'
chore(eval): scaffold human_labels.csv from sample image pool

Defines eval set uuids; author fills shade_1to5 and scene_category.
EOF
)"
```

---

### Task 4: Run variance collection (Tier B)

**Files:**
- Create: `eval/data/stability_sample.csv`
- Create: `eval/collect_run_variance.py`
- Create: `eval/data/run_variance.csv` (generated by script)

- [ ] **Step 1: Pick 10 stability sample uuids spanning `scene_category`**

Run after `human_labels.csv` has `scene_category` filled (or pick by uuid diversity if categories not yet assigned):

```bash
uv run python -c "
import pandas as pd
from pathlib import Path

labels = pd.read_csv('eval/data/human_labels.csv')
picked = []
for cat, group in labels.dropna(subset=['scene_category']).groupby('scene_category'):
    if str(cat).strip():
        picked.append(group.iloc[0]['uuid'])
    if len(picked) >= 10:
        break
if len(picked) < 10:
    for uuid in labels['uuid']:
        if uuid not in picked:
            picked.append(uuid)
        if len(picked) >= 10:
            break
out = Path('eval/data/stability_sample.csv')
pd.DataFrame({'uuid': picked[:10]}).to_csv(out, index=False)
print(f'Wrote {len(picked[:10])} uuids to {out}')
"
```

- [ ] **Step 2: Create `eval/collect_run_variance.py`**

```python
"""Score stability sample images multiple times for Tier B. Requires GOOGLE_API_KEY."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from eval.metrics import RUNS_PER_IMAGE
from eval.paths import FILTERED_METADATA_CSV, RUN_VARIANCE_CSV, SAMPLE_IMAGES_DIR, STABILITY_SAMPLE_CSV, SYNTHETIC_METADATA_CSV
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
            result = score_image(image_path, meta_row, retry=True)
            rows.append(
                {
                    "uuid": uuid,
                    "run_id": run_id,
                    "pedestrian_shade_score": result.pedestrian_shade_score,
                    "confidence": result.confidence,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"{uuid} run {run_id}: {result.pedestrian_shade_score:.2f}")

    df = pd.DataFrame(rows)
    RUN_VARIANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUN_VARIANCE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {RUN_VARIANCE_CSV}")
    return df


if __name__ == "__main__":
    collect_run_variance()
```

- [ ] **Step 3: Collect runs (requires API key)**

```bash
uv run python eval/collect_run_variance.py
```

Expected: `eval/data/run_variance.csv` with 30 rows (10 uuids × 3 runs). Commit this file with saved outputs.

- [ ] **Step 4: Commit**

```bash
git add eval/data/stability_sample.csv eval/collect_run_variance.py eval/data/run_variance.csv
git commit -m "$(cat <<'EOF'
feat(eval): add Tier B run variance collection from sample images

Scores 10-image stability subset 3x into eval/data/run_variance.csv.
EOF
)"
```

---

### Task 5: Synthetic gap-fill (optional)

**Files:**
- Create: `eval/data/synthetic_prompts.csv`
- Create: `eval/generate_images.py`
- Create: `data/images/synthetic/` (directory)
- Create: `data/synthetic_streetscapes.csv` (generated)

Skip this task entirely if time-constrained; Tier A/C still run on real images only.

- [ ] **Step 1: Create `eval/data/synthetic_prompts.csv`**

Use the example rows from the design spec (6–8 rows covering `covered_walkway`, `building_shadow`, `tree_canopy`, `open_exposure`, `mixed_sources`, `ambiguous_path`).

- [ ] **Step 2: Implement `eval/generate_images.py`**

```python
"""Generate synthetic streetscape images for Tier C gap-fill. Requires GOOGLE_API_KEY."""

from __future__ import annotations

import argparse

import pandas as pd
from google import genai
from google.genai import types

from eval.paths import SYNTHETIC_IMAGES_DIR, SYNTHETIC_METADATA_CSV, SYNTHETIC_PROMPTS_CSV
from src import config

IMAGEN_MODEL = "imagen-4.0-generate-001"

PROMPT_TEMPLATE = """Photorealistic street-level photograph, eye height ~1.5m, Singapore tropical setting.
{prompt}
Concrete walkable path visible in the lower third of the frame.
Afternoon lighting consistent with {hour}:00.
No text overlays, no watermarks, no people."""


def build_image_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(prompt=row["prompt"], hour=int(row["hour"]))


def generate_one(uuid: str, row: pd.Series, force: bool = False) -> bool:
    out_path = SYNTHETIC_IMAGES_DIR / f"{uuid}.jpeg"
    if out_path.exists() and not force:
        print(f"SKIP {uuid}: already exists")
        return False

    client = genai.Client(api_key=config.get_google_api_key())
    prompt = build_image_prompt(row)
    response = client.models.generate_images(
        model=IMAGEN_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="4:3",
        ),
    )
    image = response.generated_images[0].image
    SYNTHETIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"CREATED {out_path}")
    return True


def append_metadata(uuid: str, row: pd.Series) -> None:
    new_row = {
        "uuid": uuid,
        "source": "synthetic",
        "orig_id": "",
        "lat": 1.30,
        "lon": 103.80,
        "hour": int(row["hour"]),
        "heading": float(row["heading"]),
        "place": f"synthetic_{row['scene_category']}",
        "sidewalk_pct": float(row["sidewalk_pct"]),
    }
    if SYNTHETIC_METADATA_CSV.exists():
        existing = pd.read_csv(SYNTHETIC_METADATA_CSV)
        existing = existing[existing["uuid"] != uuid]
        df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(SYNTHETIC_METADATA_CSV, index=False)


def main(uuid_filter: str | None = None, force: bool = False) -> None:
    prompts = pd.read_csv(SYNTHETIC_PROMPTS_CSV)
    if uuid_filter:
        prompts = prompts[prompts["uuid"] == uuid_filter]
    created = skipped = errors = 0
    for _, row in prompts.iterrows():
        uuid = str(row["uuid"])
        try:
            if generate_one(uuid, row, force=force):
                append_metadata(uuid, row)
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {uuid}: {exc}")
    print(f"Summary: created={created}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", help="Generate a single uuid from synthetic_prompts.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images")
    args = parser.parse_args()
    main(uuid_filter=args.uuid, force=args.force)
```

- [ ] **Step 3: Generate images (requires API key + Imagen access)**

```bash
uv run python -m eval.generate_images
```

Review images manually. Re-roll with `--uuid syn-linkway-01 --force` if needed.

- [ ] **Step 4: Add synthetic uuids to `human_labels.csv`, label, and score**

Add rows for each synthetic uuid with `scene_category` pre-filled from `synthetic_prompts.csv`. Fill `shade_1to5`. Then:

```bash
uv run python -m eval.score_eval
```

- [ ] **Step 5: Commit**

```bash
git add eval/data/synthetic_prompts.csv eval/generate_images.py data/synthetic_streetscapes.csv data/images/synthetic/
git commit -m "$(cat <<'EOF'
feat(eval): add optional synthetic image generation for Tier C gap-fill

Generates stress-test images into synthetic/ with separate metadata CSV.
EOF
)"
```

---

### Task 6: Evaluation notebook

**Files:**
- Create: `eval/evaluation.ipynb`

- [ ] **Step 1: Create notebook with section cells**

Build `eval/evaluation.ipynb` with these cells in order:

**Cell 1 (markdown) — Intro**

```markdown
# Shade Index Evaluation

**Claim:** VLM `pedestrian_shade_score` agrees with human shade judgment on a hand-labeled sample of streetscape images.

**Ground truth:** Hand labels on 1–5 scale (normalized to 0–1), single rater, afternoon pedestrian context.

**Limitations:** N≈30 real images, one Singapore corridor, static snapshots, no solar geometry modelling. Tier A uses single-run scores from `eval/data/scores.csv`; Tier B checks stability on a 10-image subset. Optional synthetic gap-fill images are excluded from Tier A headline metrics.
```

**Cell 2 (code) — Setup**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display

from eval.metrics import (
    HIGH_VARIANCE_RANGE,
    MISMATCH_THRESHOLD,
    build_eval_frame,
    compute_method_metrics,
    filter_real_images,
    flag_mismatches,
    load_eval_metadata,
    merge_run_stability,
    per_image_run_stats,
    resolve_image_path,
    stability_summary,
    summarize_mismatches_by_scene_category,
)
from eval.paths import EVAL_SCORES_CSV, HUMAN_LABELS_CSV, RUN_VARIANCE_CSV

labels = pd.read_csv(HUMAN_LABELS_CSV)
metadata = load_eval_metadata()

if EVAL_SCORES_CSV.exists():
    scores = pd.read_csv(EVAL_SCORES_CSV)
else:
    scores = pd.DataFrame()
    print("WARNING: eval/data/scores.csv missing — run: uv run python -m eval.score_eval")

eval_frame, warnings = build_eval_frame(labels, scores, metadata)
for w in warnings:
    print("WARNING:", w)

if RUN_VARIANCE_CSV.exists():
    run_variance = pd.read_csv(RUN_VARIANCE_CSV)
    run_stats = per_image_run_stats(run_variance)
    eval_frame = merge_run_stability(eval_frame, run_stats)
else:
    run_variance = pd.DataFrame()
    run_stats = pd.DataFrame()
    print("WARNING: eval/data/run_variance.csv not found — Tier B will be skipped.")

print(f"Eval set size: {len(eval_frame)}")
print(eval_frame["source"].value_counts(dropna=False))
eval_frame.head()
```

**Cell 3 (markdown) — Tier A**

```markdown
## Tier A — VLM vs human (primary accuracy)

Headline metrics on **real Mapillary images only** (`source == mapillary`).
```

**Cell 4 (code) — Tier A metrics + scatter**

```python
tier_a = filter_real_images(eval_frame)
vlm_metrics = compute_method_metrics(tier_a, "pedestrian_shade_score")
print("Tier A (real images only):", vlm_metrics)

if tier_a.empty:
    print("No labeled real images with scores — complete human_labels.csv and run score_eval.")
else:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tier_a["human_norm"], tier_a["pedestrian_shade_score"], alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect agreement")
    ax.set_xlabel("Human (normalized)")
    ax.set_ylabel("VLM pedestrian_shade_score")
    ax.set_title(f"Tier A: Human vs VLM (n={len(tier_a)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()
```

**Cell 5 (markdown) — Tier B**

```markdown
## Tier B — Run stability

Same image + prompt scored **3 times** on a 10-image subset from `sample/`.
```

**Cell 6 (code) — Tier B metrics**

```python
if run_variance.empty:
    print("No run variance data — run: uv run python eval/collect_run_variance.py")
else:
    summary = stability_summary(run_stats, run_variance)
    print("Stability summary:", summary)
    display(run_stats.sort_values("score_std", ascending=False))

    high_var = run_stats[run_stats["score_range"] > HIGH_VARIANCE_RANGE]
    if not high_var.empty:
        print(f"High-variance images (range > {HIGH_VARIANCE_RANGE}):")
        display(high_var)
```

**Cell 7 (markdown) — Tier C**

```markdown
## Tier C — Mismatch analysis

Images where VLM error exceeds **0.25** vs human labels. Grouped by `scene_category`. Includes real and synthetic images; check `source` when interpreting.
```

**Cell 8 (code) — Tier C summary tables**

```python
mismatches = flag_mismatches(eval_frame, threshold=MISMATCH_THRESHOLD)
print(f"Mismatches: {len(mismatches)} (threshold={MISMATCH_THRESHOLD})")

if not mismatches.empty:
    display(summarize_mismatches_by_scene_category(mismatches))
    if len(mismatches) >= 3:
        (
            mismatches.groupby("scene_category")["err_vlm"]
            .mean()
            .sort_values(ascending=False)
            .plot(kind="bar", title="Mean VLM error by scene_category")
        )
        plt.ylabel("mean |VLM - human|")
        plt.tight_layout()
        plt.show()
else:
    print("No mismatches at current threshold — consider lowering MISMATCH_THRESHOLD.")
```

**Cell 9 (code) — Tier C image gallery**

```python
from PIL import Image


def show_mismatch_card(row):
    image_path = resolve_image_path(row["uuid"])
    stability_line = ""
    if "run_std" in row and pd.notna(row.get("run_std")):
        stability_line = (
            f"<p>Run stability: std={row['run_std']:.2f} · range={row['run_range']:.2f}</p>"
        )
    hour = row.get("hour", "—")
    sidewalk = row.get("sidewalk_pct", float("nan"))
    sidewalk_text = f"{sidewalk:.1%}" if pd.notna(sidewalk) else "—"
    display(
        HTML(
            f"<h4>{row['scene_category']} · {row.get('source', '—')}</h4>"
            f"<p>"
            f"human={row['shade_1to5']} (norm {row['human_norm']:.2f}) · "
            f"VLM={row['pedestrian_shade_score']:.2f} (err {row['err_vlm']:.2f})"
            f"</p>"
            f"<p>hour={hour} · sidewalk_pct={sidewalk_text} · heading={row.get('heading', '—')}°</p>"
            f"{stability_line}"
            f"<p><i>{row.get('reasoning', '')}</i></p>"
            f"<p><b>Notes:</b> {row.get('notes', '') or '—'}</p>"
        )
    )
    if image_path and image_path.exists():
        display(Image.open(image_path).convert("RGB"))
    else:
        display(HTML(f"<p><i>Missing image for {row['uuid']}</i></p>"))
    display(HTML("<hr>"))


if mismatches.empty:
    print("No mismatch gallery to show.")
else:
    for scene_category, group in mismatches.groupby("scene_category"):
        display(HTML(f"<h3>{scene_category}</h3>"))
        for _, row in group.iterrows():
            show_mismatch_card(row)
```

**Cell 10 (markdown) — Takeaways**

```markdown
## Takeaways

<!-- Fill after reviewing results. Example structure:

- Tier A (real only): VLM Spearman ρ = X.XX, MAE = Y.YY (n=N)
- Tier B: median run std = X.XX; M/M images high-variance
- Tier C: biggest errors in `ambiguous_path` / `open_exposure` (if pattern emerges)
- High-variance mismatches: treat as ambiguous, not definitive failures
- Synthetic gap-fill: supplementary for Tier C only
- Limitation: single rater, static images, N≈30 real images
-->
```

- [ ] **Step 2: Execute notebook and save outputs**

```bash
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Open the notebook, fill in the Takeaways cell with real numbers, re-run if needed, and save with all cell outputs visible.

- [ ] **Step 3: Commit**

```bash
git add eval/evaluation.ipynb
git commit -m "$(cat <<'EOF'
feat(eval): add evaluation notebook with Tier A/B/C report

Committed with saved outputs for offline review.
EOF
)"
```

---

### Task 7: README eval section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add or update Evaluation section**

Replace placeholders with actual numbers from the executed notebook:

```markdown
## Evaluation

We evaluate whether VLM `pedestrian_shade_score` agrees with human shade judgment on a hand-labeled sample of ~30 corridor images (1–5 afternoon pedestrian shade scale, single rater). Eval scores live in `eval/data/scores.csv` (separate from the demo app's `data/scores.csv`).

| Metric | Value |
|--------|-------|
| Spearman ρ (Tier A, real images) | _fill_ |
| MAE (Tier A, real images) | _fill_ |

**Run stability (Tier B):** 10 images scored 3× each; median run-to-run std = _fill_.

Full methodology, mismatch analysis, and image gallery: [`eval/evaluation.ipynb`](eval/evaluation.ipynb) (committed with saved outputs).

**Limitations:** single rater, N≈30 real images, one corridor, static snapshots — shade varies by time of day and season. Optional synthetic gap-fill images support Tier C category coverage only and are excluded from headline Tier A metrics.

To re-run:

```bash
uv run python -m eval.score_eval          # score eval set → eval/data/scores.csv
uv run python eval/collect_run_variance.py  # Tier B (optional)
uv run jupyter notebook eval/evaluation.ipynb
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: add evaluation summary to README

Links to notebook with Tier A/B headline numbers and limitations.
EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| `eval/data/scores.csv` separate from `data/scores.csv` | Task 2, Task 6 |
| Tier A: Spearman + MAE vs human (real only) | Task 1, Task 6 (cells 3–4) |
| Tier B: run stability (10×3) | Task 1, Task 4, Task 6 (cells 5–6) |
| Tier C: mismatch + `scene_category` + hour/sidewalk_pct + images | Task 1, Task 6 (cells 7–9) |
| `eval/data/human_labels.csv` with `scene_category` | Task 3 |
| Images from `sample/` + `synthetic/`, not `exploration/` | Task 2, Task 4, Task 6 |
| Composite baseline dropped | Task 1 (no composite functions) |
| Optional synthetic gap-fill | Task 5 |
| `eval/data/run_variance.csv` schema | Task 4 |
| Notebook Tier A before Tier B | Task 6 |
| README integration | Task 7 |
| Error handling (missing uuids, images, scores, run_variance) | Task 1, Task 6 |
| No new dependencies | ✓ |

---

## Verification

```bash
uv run pytest tests/test_eval_metrics.py tests/test_score_eval.py -v
uv run python -m eval.score_eval                    # requires GOOGLE_API_KEY + labeled human_labels.csv
uv run python eval/collect_run_variance.py          # Tier B; requires API key
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Manual checks:
1. `eval/evaluation.ipynb` opens with tables and images visible (no re-run needed)
2. Tier A uses `eval/data/scores.csv` and filters to `source == mapillary`
3. Tier B shows stability summary when `run_variance.csv` is present
4. Tier C gallery groups by `scene_category` and shows hour + sidewalk_pct
5. `data/scores.csv` unchanged by eval scripts
6. README numbers match notebook Tier A metrics
