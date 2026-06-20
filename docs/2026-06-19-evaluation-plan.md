# Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-based evaluation (`eval/evaluation.ipynb`) that compares VLM shade scores against human labels and a composite GVI/SVI baseline, with Tier C mismatch galleries grouped by `place` metadata.

**Architecture:** Pure metric/join logic lives in `eval/metrics.py` (unit-tested). The notebook imports those functions, loads CSVs via `src.config` paths, and renders Tier A/B tables plus Tier C inline images. `eval/human_labels.csv` is the hand-label input scaffold.

**Tech Stack:** Python 3.10, pandas, matplotlib, Pillow, IPython/Jupyter (all already in `pyproject.toml`)

**Spec:** [2026-06-19-evaluation-design.md](./2026-06-19-evaluation-design.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `eval/metrics.py` | Join eval data, composite score, Spearman/MAE, mismatch flags |
| `tests/test_eval_metrics.py` | Unit tests for metrics module |
| `eval/human_labels.csv` | ~30 uuids + empty label columns for author |
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
    build_eval_frame,
    classify_miss_type,
    composite_score,
    compute_method_metrics,
    flag_mismatches,
    normalize_human,
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

### Task 3: Evaluation notebook

**Files:**
- Create: `eval/evaluation.ipynb`

- [ ] **Step 1: Create notebook with section cells**

Build `eval/evaluation.ipynb` with these cells in order:

**Cell 1 (markdown) — Intro**

```markdown
# Shade Index Evaluation

**Claim:** VLM `pedestrian_shade_score` agrees with human shade judgment better than a composite of `green_view_index` and `sky_view_index`.

**Ground truth:** Hand labels on 1–5 scale (normalized to 0–1), single rater, afternoon pedestrian context.

**Limitations:** N≈30, one Singapore corridor, static snapshots, no solar geometry modelling.
```

**Cell 2 (code) — Setup**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display

from eval.metrics import (
    MISMATCH_THRESHOLD,
    build_eval_frame,
    comparison_table,
    flag_mismatches,
    summarize_mismatches_by_place,
)
from src.config import IMAGES_DIR, METADATA_CSV, SCORES_CSV

LABELS_CSV = Path("eval/human_labels.csv")

labels = pd.read_csv(LABELS_CSV)
scores = pd.read_csv(SCORES_CSV)
metadata = pd.read_csv(METADATA_CSV)

eval_frame, warnings = build_eval_frame(labels, scores, metadata)
for w in warnings:
    print("WARNING:", w)

print(f"Eval set size: {len(eval_frame)}")
eval_frame.head()
```

**Cell 3 (markdown) — Tier A**

```markdown
## Tier A — VLM vs human
```

**Cell 4 (code) — Tier A metrics + scatter**

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

**Cell 5 (markdown) — Tier B**

```markdown
## Tier B — VLM vs composite baseline

Composite = `0.5 * (1 - sky_view_index) + 0.5 * green_view_index`
```

**Cell 6 (code) — Tier B comparison table**

```python
comparison_table(eval_frame)
```

**Cell 7 (markdown) — Tier C**

```markdown
## Tier C — Mismatch analysis

Images where either VLM or composite error exceeds **0.25** vs human labels. Grouped by `place` metadata.
```

**Cell 8 (code) — Tier C summary tables**

```python
mismatches = flag_mismatches(eval_frame, threshold=MISMATCH_THRESHOLD)
print(f"Mismatches: {len(mismatches)} (threshold={MISMATCH_THRESHOLD})")

if not mismatches.empty:
    display(summarize_mismatches_by_place(mismatches))
    display(mismatches.groupby("miss_type").size().to_frame("count"))
else:
    print("No mismatches at current threshold — consider lowering MISMATCH_THRESHOLD.")
```

**Cell 9 (code) — Tier C image gallery**

```python
from PIL import Image

def show_mismatch_card(row):
    image_path = IMAGES_DIR / f"{row['uuid']}.jpeg"
    display(HTML(
        f"<h4>{row['place']} · {row['miss_type']}</h4>"
        f"<p>"
        f"human={row['shade_1to5']} (norm {row['human_norm']:.2f}) · "
        f"VLM={row['pedestrian_shade_score']:.2f} (err {row['err_vlm']:.2f}) · "
        f"composite={row['composite_score']:.2f} (err {row['err_composite']:.2f})"
        f"</p>"
        f"<p>GVI={row['green_view_index']:.2f} · SVI={row['sky_view_index']:.2f} · "
        f"heading={row['heading']}°</p>"
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

**Cell 10 (markdown) — Takeaways**

```markdown
## Takeaways

<!-- Fill after reviewing results. Example structure:

- VLM Spearman ρ = X.XX vs composite ρ = Y.YY
- Biggest composite misses: `parking_lot` (high GVI, no canopy over path)
- Biggest VLM misses: ...
- Limitation: single rater, static images
-->
```

- [ ] **Step 2: Execute notebook and save outputs**

```bash
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Open the notebook, fill in the Takeaways cell with real numbers, re-run if needed, and save with all cell outputs visible.

---

### Task 4: README eval section

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

Full methodology, mismatch analysis, and image gallery: [`eval/evaluation.ipynb`](eval/evaluation.ipynb) (committed with saved outputs).

**Limitations:** single rater, N≈30, one corridor, static snapshots — shade varies by time of day and season.

To re-run: `uv run jupyter notebook eval/evaluation.ipynb`
```

- [ ] **Step 2: Add notebook run note to existing setup instructions**

Ensure README documents that eval requires labeled `eval/human_labels.csv` and scored `data/scores.csv`.

---

### Task 5: Update design spec status

**Files:**
- Modify: `docs/2026-06-19-evaluation-design.md`

- [ ] **Step 1: Mark spec as implemented**

Change header status from `Draft — pending review` to `Approved — implemented`.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Tier A: Spearman + MAE vs human | Task 1 (`compute_method_metrics`), Task 3 (cells 3–4) |
| Tier B: composite baseline only | Task 1 (`composite_score`), Task 3 (cells 5–6) |
| Tier C: mismatch + place + images | Task 1 (`flag_mismatches`, `summarize_mismatches_by_place`), Task 3 (cells 7–9) |
| `human_labels.csv` schema | Task 2 |
| Notebook with saved outputs | Task 3 |
| README integration | Task 4 |
| Error handling (missing uuids, images) | Task 1 (`build_eval_frame` warnings), Task 3 (gallery fallback) |
| No new dependencies | ✓ |

---

## Verification

```bash
uv run pytest tests/test_eval_metrics.py -v
uv run jupyter execute eval/evaluation.ipynb --inplace
```

Manual checks:
1. `eval/evaluation.ipynb` opens with tables and images visible (no re-run needed)
2. Tier C gallery shows ≥5 mismatch cases once labels are complete
3. README numbers match notebook comparison table
