# Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-based evaluation (`eval/evaluation.ipynb`) that compares VLM shade scores against human labels and a composite GVI/SVI baseline, assesses run-to-run VLM stability (Tier A½), and includes Tier C mismatch galleries grouped by `place` metadata.

**Architecture:** Pure metric/join logic lives in `eval/metrics.py` (unit-tested), including run-variance summaries. Multi-run scores are collected via `eval/collect_run_variance.py` into `eval/run_variance.csv`. The notebook imports metrics, loads CSVs via `src.config` paths, and renders Tier A½/A/B tables plus Tier C inline images.

**Tech Stack:** Python 3.10, pandas, matplotlib, Pillow, IPython/Jupyter (all already in `pyproject.toml`)

**Spec:** [2026-06-19-evaluation-design.md](./2026-06-19-evaluation-design.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `eval/metrics.py` | Join eval data, composite score, Spearman/MAE, mismatch flags, run-variance summaries |
| `eval/collect_run_variance.py` | Score 10 sample images 3× each; write `run_variance.csv` |
| `tests/test_eval_metrics.py` | Unit tests for metrics module |
| `eval/human_labels.csv` | ~30 uuids + empty label columns for author |
| `eval/stability_sample.csv` | 10 uuids subset for Tier A½ |
| `eval/run_variance.csv` | Raw multi-run VLM scores (30 rows) |
| `eval/evaluation.ipynb` | Full eval report with saved outputs |
| `README.md` | Eval summary section + link to notebook |

---

### Task 1: Eval metrics module

**Files:**
- Create: `eval/metrics.py`
- Create: `eval/__init__.py` (empty)
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `eval/__init__.py` (empty file).

Create `tests/test_eval_metrics.py`:

```python
import pandas as pd
import pytest

from eval.metrics import (
    MISMATCH_THRESHOLD,
    HIGH_VARIANCE_RANGE,
    build_eval_frame,
    classify_miss_type,
    composite_score,
    compute_method_metrics,
    flag_mismatches,
    merge_run_stability,
    normalize_human,
    per_image_run_stats,
    stability_summary,
    summarize_mismatches_by_place,
)


def test_normalize_human():
    s = pd.Series([1, 3, 5])
    result = normalize_human(s)
    pd.testing.assert_series_equal(result, pd.Series([0.0, 0.5, 1.0]), check_names=False)


def test_composite_score():
    row = pd.Series({"green_view_index": 0.6, "sky_view_index": 0.2})
    assert composite_score(row) == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)


def test_classify_miss_type():
    assert classify_miss_type(0.30, 0.10) == "vlm_miss"
    assert classify_miss_type(0.10, 0.30) == "composite_miss"
    assert classify_miss_type(0.30, 0.30) == "both_miss"
    assert classify_miss_type(0.10, 0.10) == "ok"


def test_build_eval_frame_inner_join_and_warnings():
    labels = pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "shade_1to5": [3, 4, 5],
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
            "place": ["campus", "hospital"],
            "green_view_index": [0.4, 0.2],
            "sky_view_index": [0.1, 0.5],
            "heading": [90.0, 180.0],
            "lat": [1.3, 1.31],
            "lon": [103.8, 103.81],
        }
    )
    frame, warnings = build_eval_frame(labels, scores, metadata)
    assert len(frame) == 2
    assert "human_norm" in frame.columns
    assert "composite_score" in frame.columns
    assert any("c" in w for w in warnings)


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


def test_flag_mismatches():
    frame = pd.DataFrame(
        {
            "uuid": ["x", "y", "z"],
            "human_norm": [0.5, 0.5, 0.5],
            "pedestrian_shade_score": [0.5, 0.9, 0.2],
            "composite_score": [0.5, 0.5, 0.2],
            "place": ["campus", "hospital", "park"],
        }
    )
    mismatches = flag_mismatches(frame, threshold=0.25)
    assert set(mismatches["uuid"]) == {"y", "z"}
    assert mismatches.loc[mismatches["uuid"] == "y", "miss_type"].iloc[0] == "vlm_miss"


def test_summarize_mismatches_by_place():
    mismatches = pd.DataFrame(
        {
            "place": ["campus", "campus", "hospital"],
            "miss_type": ["vlm_miss", "both_miss", "composite_miss"],
        }
    )
    summary = summarize_mismatches_by_place(mismatches)
    assert summary.loc["campus", "vlm_miss"] == 1
    assert summary.loc["campus", "both_miss"] == 1
    assert summary.loc["hospital", "composite_miss"] == 1


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
    frame = pd.DataFrame({"uuid": ["a", "b"], "place": ["campus", "park"]})
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

- [ ] **Step 3: Implement `eval/metrics.py`**

```python
from __future__ import annotations

import pandas as pd

MISMATCH_THRESHOLD = 0.25
HIGH_VARIANCE_RANGE = 0.15
RUNS_PER_IMAGE = 3
STABILITY_SAMPLE_SIZE = 10


def normalize_human(shade_1to5: pd.Series) -> pd.Series:
    return (shade_1to5.astype(float) - 1) / 4


def composite_score(row: pd.Series) -> float:
    gvi = float(row["green_view_index"])
    svi = float(row["sky_view_index"])
    return 0.5 * (1 - svi) + 0.5 * gvi


def classify_miss_type(err_vlm: float, err_composite: float, threshold: float = MISMATCH_THRESHOLD) -> str:
    vlm_bad = err_vlm > threshold
    comp_bad = err_composite > threshold
    if vlm_bad and comp_bad:
        return "both_miss"
    if vlm_bad:
        return "vlm_miss"
    if comp_bad:
        return "composite_miss"
    return "ok"


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
        warnings.append(f"Excluded uuids missing from scores.csv: {missing_scores}")
    if missing_meta:
        warnings.append(f"Excluded uuids missing from metadata: {missing_meta}")

    frame = (
        labeled.merge(scores, on="uuid", how="inner")
        .merge(metadata, on="uuid", how="inner", suffixes=("", "_meta"))
    )
    frame["human_norm"] = normalize_human(frame["shade_1to5"])
    frame["composite_score"] = frame.apply(composite_score, axis=1)
    frame["err_vlm"] = (frame["pedestrian_shade_score"] - frame["human_norm"]).abs()
    frame["err_composite"] = (frame["composite_score"] - frame["human_norm"]).abs()
    return frame, warnings


def compute_method_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float | int]:
    n = len(frame)
    if n == 0:
        return {"n": 0, "spearman": float("nan"), "mae": float("nan")}
    spearman = frame["human_norm"].corr(frame[score_col], method="spearman")
    mae = (frame[score_col] - frame["human_norm"]).abs().mean()
    return {"n": n, "spearman": float(spearman), "mae": float(mae)}


def flag_mismatches(frame: pd.DataFrame, threshold: float = MISMATCH_THRESHOLD) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        miss_type = classify_miss_type(row["err_vlm"], row["err_composite"], threshold)
        if miss_type == "ok":
            continue
        rows.append({**row.to_dict(), "miss_type": miss_type})
    if not rows:
        return pd.DataFrame(columns=list(frame.columns) + ["miss_type"])
    return pd.DataFrame(rows)


def summarize_mismatches_by_place(mismatches: pd.DataFrame) -> pd.DataFrame:
    if mismatches.empty:
        return pd.DataFrame()
    return (
        mismatches.groupby(["place", "miss_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    vlm = compute_method_metrics(frame, "pedestrian_shade_score")
    comp = compute_method_metrics(frame, "composite_score")
    return pd.DataFrame(
        [
            {"method": "VLM", "spearman": vlm["spearman"], "mae": vlm["mae"], "n": vlm["n"]},
            {"method": "Composite", "spearman": comp["spearman"], "mae": comp["mae"], "n": comp["n"]},
        ]
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
    test_retest = float(run1.loc[common].corr(run2.loc[common], method="spearman")) if len(common) >= 2 else float("nan")
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_eval_metrics.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

Expected: all non-integration tests PASS

---

### Task 2: Human labels scaffold

**Files:**
- Create: `eval/human_labels.csv`

- [ ] **Step 1: Generate ~30 diverse uuids from corridor metadata**

Run once to pick uuids spanning different `place` values (max 2 per place, cap at 30):

```bash
uv run python -c "
import pandas as pd
from pathlib import Path

meta = pd.read_csv('data/filtered_streetscapes.csv')
rows = []
for place, group in meta.groupby('place'):
    for uuid in group['uuid'].head(2):
        rows.append({'uuid': uuid, 'shade_1to5': '', 'notes': ''})
        if len(rows) >= 30:
            break
    if len(rows) >= 30:
        break
# top up if fewer than 30 places
if len(rows) < 30:
    seen = {r['uuid'] for r in rows}
    for uuid in meta['uuid']:
        if uuid not in seen:
            rows.append({'uuid': uuid, 'shade_1to5': '', 'notes': ''})
        if len(rows) >= 30:
            break
out = Path('eval/human_labels.csv')
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f'Wrote {len(rows)} rows to {out}')
print(meta[meta['uuid'].isin([r[\"uuid\"] for r in rows])]['place'].value_counts())
"
```

- [ ] **Step 2: Hand-label `shade_1to5` (author task, not automatable)**

Open images in `data/images/exploration/{uuid}.jpeg` and fill `shade_1to5` (1–5) per rubric in the design spec. Add optional `notes` for notable cases.

Rubric reminder:
- 1 = fully exposed sidewalk; 5 = fully shaded walkable path
- Anchor to 2–4pm tropical sun
- Ignore clouds unless they block sun on the path

- [ ] **Step 3: Score any unlabeled images via the app**

For uuids in `human_labels.csv` that lack rows in `data/scores.csv`, run scoring through the FastAPI app or `uv run python -c "from src.score import run_scoring; run_scoring()"`.

---

### Task 3: Run variance collection (Tier A½)

**Files:**
- Create: `eval/stability_sample.csv`
- Create: `eval/collect_run_variance.py`
- Create: `eval/run_variance.csv` (generated by script)

- [ ] **Step 1: Pick 10 stability sample uuids**

Run after `human_labels.csv` exists (labeled or unlabeled — uuids only needed):

```bash
uv run python -c "
import pandas as pd
from pathlib import Path

labels = pd.read_csv('eval/human_labels.csv')
meta = pd.read_csv('data/filtered_streetscapes.csv')
merged = labels.merge(meta[['uuid', 'place']], on='uuid')
# spread across places: one per place until 10
picked = []
for place, group in merged.groupby('place'):
    picked.append(group.iloc[0]['uuid'])
    if len(picked) >= 10:
        break
if len(picked) < 10:
    for uuid in merged['uuid']:
        if uuid not in picked:
            picked.append(uuid)
        if len(picked) >= 10:
            break
out = Path('eval/stability_sample.csv')
pd.DataFrame({'uuid': picked[:10]}).to_csv(out, index=False)
print(f'Wrote {len(picked[:10])} uuids to {out}')
"
```

- [ ] **Step 2: Create `eval/collect_run_variance.py`**

```python
"""Score stability sample images multiple times for Tier A½. Requires GOOGLE_API_KEY."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.metrics import RUNS_PER_IMAGE
from src.config import IMAGES_DIR, METADATA_CSV
from src.score import score_image

STABILITY_SAMPLE_CSV = Path("eval/stability_sample.csv")
RUN_VARIANCE_CSV = Path("eval/run_variance.csv")


def collect_run_variance(runs_per_image: int = RUNS_PER_IMAGE) -> pd.DataFrame:
    sample = pd.read_csv(STABILITY_SAMPLE_CSV)
    metadata = pd.read_csv(METADATA_CSV).set_index("uuid")
    rows: list[dict] = []

    for uuid in sample["uuid"]:
        image_path = IMAGES_DIR / f"{uuid}.jpeg"
        if uuid not in metadata.index:
            print(f"WARNING: no metadata for {uuid}")
            continue
        if not image_path.exists():
            print(f"WARNING: missing image {image_path}")
            continue
        meta_row = metadata.loc[uuid]
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
    RUN_VARIANCE_CSV.parent.mkdir(exist_ok=True)
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

Expected: `eval/run_variance.csv` with 30 rows (10 uuids × 3 runs). Commit this file with saved outputs.

---

### Task 4: Evaluation notebook

**Files:**
- Create: `eval/evaluation.ipynb`

- [ ] **Step 1: Create notebook with section cells**

Build `eval/evaluation.ipynb` with these cells in order:

**Cell 1 (markdown) — Intro**

```markdown
# Shade Index Evaluation

**Claim:** VLM `pedestrian_shade_score` agrees with human shade judgment better than a composite of `green_view_index` and `sky_view_index`.

**Ground truth:** Hand labels on 1–5 scale (normalized to 0–1), single rater, afternoon pedestrian context.

**Limitations:** N≈30, one Singapore corridor, static snapshots, no solar geometry modelling. Tier A uses single-run scores; Tier A½ checks stability on a 10-image subset.
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
    comparison_table,
    flag_mismatches,
    merge_run_stability,
    per_image_run_stats,
    stability_summary,
    summarize_mismatches_by_place,
)
from src.config import IMAGES_DIR, METADATA_CSV, SCORES_CSV

LABELS_CSV = Path("eval/human_labels.csv")
RUN_VARIANCE_CSV = Path("eval/run_variance.csv")

labels = pd.read_csv(LABELS_CSV)
scores = pd.read_csv(SCORES_CSV)
metadata = pd.read_csv(METADATA_CSV)

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
    print("WARNING: eval/run_variance.csv not found — Tier A½ will be skipped.")

print(f"Eval set size: {len(eval_frame)}")
eval_frame.head()
```

**Cell 3 (markdown) — Tier A½**

```markdown
## Tier A½ — Run stability

Same image + prompt scored **3 times** on a 10-image subset. Composite baseline has zero run variance by construction.
```

**Cell 4 (code) — Tier A½ metrics**

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

**Cell 5 (markdown) — Tier A**

```markdown
## Tier A — VLM vs human
```

**Cell 6 (code) — Tier A metrics + scatter**

```python
from eval.metrics import compute_method_metrics

vlm_metrics = compute_method_metrics(eval_frame, "pedestrian_shade_score")
print("VLM:", vlm_metrics)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(eval_frame["human_norm"], eval_frame["pedestrian_shade_score"], alpha=0.8)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect agreement")
ax.set_xlabel("Human (normalized)")
ax.set_ylabel("VLM pedestrian_shade_score")
ax.set_title("Tier A: Human vs VLM")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
plt.show()
```

**Cell 7 (markdown) — Tier B**

```markdown
## Tier B — VLM vs composite baseline

Composite = `0.5 * (1 - sky_view_index) + 0.5 * green_view_index`
```

**Cell 8 (code) — Tier B comparison table**

```python
comparison_table(eval_frame)
```

**Cell 9 (markdown) — Tier C**

```markdown
## Tier C — Mismatch analysis

Images where either VLM or composite error exceeds **0.25** vs human labels. Grouped by `place` metadata. Images in the stability sample also show run std/range.
```

**Cell 10 (code) — Tier C summary tables**

```python
mismatches = flag_mismatches(eval_frame, threshold=MISMATCH_THRESHOLD)
print(f"Mismatches: {len(mismatches)} (threshold={MISMATCH_THRESHOLD})")

if not mismatches.empty:
    display(summarize_mismatches_by_place(mismatches))
    display(mismatches.groupby("miss_type").size().to_frame("count"))
else:
    print("No mismatches at current threshold — consider lowering MISMATCH_THRESHOLD.")
```

**Cell 11 (code) — Tier C image gallery**

```python
from PIL import Image

def show_mismatch_card(row):
    image_path = IMAGES_DIR / f"{row['uuid']}.jpeg"
    stability_line = ""
    if "run_std" in row and pd.notna(row.get("run_std")):
        stability_line = (
            f"<p>Run stability: std={row['run_std']:.2f} · range={row['run_range']:.2f}</p>"
        )
    display(HTML(
        f"<h4>{row['place']} · {row['miss_type']}</h4>"
        f"<p>"
        f"human={row['shade_1to5']} (norm {row['human_norm']:.2f}) · "
        f"VLM={row['pedestrian_shade_score']:.2f} (err {row['err_vlm']:.2f}) · "
        f"composite={row['composite_score']:.2f} (err {row['err_composite']:.2f})"
        f"</p>"
        f"<p>GVI={row['green_view_index']:.2f} · SVI={row['sky_view_index']:.2f} · "
        f"heading={row['heading']}°</p>"
        f"{stability_line}"
        f"<p><i>{row.get('reasoning', '')}</i></p>"
        f"<p><b>Notes:</b> {row.get('notes', '') or '—'}</p>"
    ))
    if image_path.exists():
        display(Image.open(image_path).convert("RGB"))
    else:
        display(HTML(f"<p><i>Missing image: {image_path}</i></p>"))
    display(HTML("<hr>"))

if mismatches.empty:
    print("No mismatch gallery to show.")
else:
    for miss_type, group in mismatches.groupby("miss_type"):
        display(HTML(f"<h3>{miss_type}</h3>"))
        for _, row in group.iterrows():
            show_mismatch_card(row)
```

**Cell 12 (markdown) — Takeaways**

```markdown
## Takeaways

<!-- Fill after reviewing results. Example structure:

- VLM Spearman ρ = X.XX vs composite ρ = Y.YY
- Tier A½: median run std = X.XX; N/M images high-variance
- Biggest composite misses: `parking_lot` (high GVI, no canopy over path)
- High-variance mismatches: treat as ambiguous, not definitive failures
- Limitation: single rater, static images, single-run Tier A scores
-->
```

- [ ] **Step 2: Execute notebook and save outputs**

```bash
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Open the notebook, fill in the Takeaways cell with real numbers, re-run if needed, and save with all cell outputs visible.

---

### Task 5: README eval section

**Files:**
- Create or modify: `README.md`

- [ ] **Step 1: Add Evaluation section**

Add (or create README with) a section along these lines — replace placeholders with actual numbers from the executed notebook:

```markdown
## Evaluation

We evaluate whether VLM `pedestrian_shade_score` agrees with human shade judgment better than a composite baseline derived from Global Streetscapes indices (`0.5·(1−SVI) + 0.5·GVI`). A single rater hand-labeled ~30 corridor images on a 1–5 afternoon pedestrian shade scale.

| Method | Spearman ρ | MAE |
|--------|------------|-----|
| VLM | _fill_ | _fill_ |
| Composite | _fill_ | _fill_ |

**Run stability (Tier A½):** 10 images scored 3× each; median run-to-run std = _fill_.

Full methodology, mismatch analysis, and image gallery: [`eval/evaluation.ipynb`](eval/evaluation.ipynb) (committed with saved outputs).

**Limitations:** single rater, N≈30, one corridor, static snapshots — shade varies by time of day and season. Tier A uses single-run scores.

To re-run: `uv run jupyter notebook eval/evaluation.ipynb`
```

- [ ] **Step 2: Add notebook run note to existing setup instructions**

Ensure README documents that eval requires labeled `eval/human_labels.csv`, scored `data/scores.csv`, and `eval/run_variance.csv` for Tier A½.

---

### Task 6: Update design spec status

**Files:**
- Modify: `docs/2026-06-19-evaluation-design.md`

- [ ] **Step 1: Mark spec as implemented**

Change header status from `Draft — pending review` to `Approved — implemented`.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Tier A½: run stability (10×3) | Task 1 (`per_image_run_stats`, `stability_summary`), Task 3, Task 4 (cells 3–4) |
| Tier A: Spearman + MAE vs human | Task 1 (`compute_method_metrics`), Task 4 (cells 5–6) |
| Tier B: composite baseline only | Task 1 (`composite_score`), Task 4 (cells 7–8) |
| Tier C: mismatch + place + images + run_std | Task 1 (`flag_mismatches`, `merge_run_stability`), Task 4 (cells 9–11) |
| `human_labels.csv` schema | Task 2 |
| `run_variance.csv` schema | Task 3 |
| Notebook with saved outputs | Task 4 |
| README integration | Task 5 |
| Error handling (missing uuids, images, run_variance) | Task 1, Task 4 |
| No new dependencies | ✓ |

---

## Verification

```bash
uv run pytest tests/test_eval_metrics.py -v
uv run python eval/collect_run_variance.py   # if run_variance.csv not yet committed
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Manual checks:
1. `eval/evaluation.ipynb` opens with tables and images visible (no re-run needed)
2. Tier A½ shows stability summary when `run_variance.csv` is present
3. Tier C gallery shows ≥5 mismatch cases once labels are complete
4. README numbers match notebook comparison table
