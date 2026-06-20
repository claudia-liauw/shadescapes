# Evaluation Design: Shade Index VLM

**Date:** 2026-06-19  
**Status:** Approved  
**Related:** [instructions.md](./instructions.md), [vlm-approach.md](./vlm-approach.md)

---

## Problem

The GovTech brief requires an honest evaluation with a justified methodology. We need to demonstrate that VLM `pedestrian_shade_score` measures functional pedestrian shade in a way that aligns with human judgment.

**Primary claim:** VLM shade scores agree with human judgment on a hand-labeled sample of streetscape images.

**Unit of evaluation:** Individual streetscape images (~30 hand-labeled from the corridor dataset).

**Not claiming:** Island-wide accuracy, time-of-day precision, segment/route-level performance, or that the VLM is ground truth.

---

## Approaches Considered

### A. Notebook report (recommended)

Single `eval/evaluation.ipynb` runs metrics and renders Tier C image galleries inline. Committed with saved cell outputs so reviewers can read results offline.

| Pros | Cons |
|------|------|
| One artifact for numbers + visuals | Notebook diffs are noisy when re-run |
| Fits exploratory error analysis | Requires `ipykernel` (already in deps) |
| Natural fit for Tier C image display | |

### B. Script + static HTML gallery

`run_eval.py` writes `results.md` and `error_gallery.html`; notebook optional.

| Pros | Cons |
|------|------|
| Clean git diffs for metrics | Two artifacts to maintain |
| HTML gallery is polished | More boilerplate for marginal gain at N=30 |

### C. Script + markdown only

Metrics in `results.md`; no inline images.

| Pros | Cons |
|------|------|
| Minimal | Tier C loses impact without visuals |
| | Weak for demo and PROCESS.md material |

**Decision:** Approach A. The notebook is the eval deliverable; README links to it with headline numbers only.

---

## Architecture

```text
human_labels.csv ──┐
scores.csv ────────┤
run_variance.csv ──┼──► evaluation.ipynb ──► saved outputs (tables, charts, image gallery)
filtered_streetscapes.csv ──┘
data/images/exploration/{uuid}.jpeg ──► Tier C inline display
```

No separate eval service or API. The notebook imports from `src.config` for paths (`METADATA_CSV`, `SCORES_CSV`, `IMAGES_DIR`) to stay aligned with the app.

Optional `eval/run_eval.py` may hold pure functions (join, metrics, mismatch flags) imported by the notebook if cells grow long. Not required for v1.

---

## Evaluation Tiers

### Tier A — VLM vs human (primary accuracy)

**Ground truth:** Hand-label ~30 images on a 1–5 shade scale.

**Labeling rubric:**
- Anchor to afternoon pedestrian experience: 2–4pm tropical sun, sidewalk-level
- Score shade on the **walkable path**, not scene aesthetics
- Ignore cloud cover unless it materially blocks sun on the path
- Single rater (author) — document as a limitation

**Normalization:** `human_norm = (shade_1to5 - 1) / 4` → 0–1 for comparison with VLM output.

**Metrics:**
- **Spearman ρ** (primary) — rank agreement; robust to scale differences between 1–5 and 0–1
- **MAE** (secondary) — mean absolute error after normalization

**Implementation:** `pandas.Series.corr(method="spearman")` for ρ; `(pred - human_norm).abs().mean()` for MAE.

**Scores used:** Single-run values from `data/scores.csv` (the production scoring path). Run stability is assessed separately in Tier B.

### Tier B — Run stability (VLM reliability)

Measures whether the VLM returns **consistent** scores for the same image + prompt across repeated API calls. This is distinct from Tier A accuracy (correctness vs human).

**Protocol:**
- Select **10 images** from the eval set (`human_labels.csv`), spanning varied shade levels and `scene_category` values
- Score each image **3 times** with the same prompt/model as production (`src.score.score_image`)
- Store raw runs in `eval/run_variance.csv`

**Metrics (descriptive, no significance tests at N=10):**

| Metric | Definition |
|--------|------------|
| **Median std** | Median per-image standard deviation across 3 runs |
| **Max range** | Worst-case `max − min` across runs for any image |
| **% high-variance** | Share of images where range > `0.15` |
| **Test–retest Spearman** | Rank correlation between run 1 and run 2 scores across the 10 images |

**Framing:** Stability is a VLM deployment concern — a cool route should not flip on re-scoring.

**Scope limit:** 10 images × 3 runs = 30 API calls. Do not expand to full 30-image × 5-run grid.

### Tier C — Mismatch analysis with metadata and visuals

Qualitative error analysis for cases where the VLM disagrees with human judgment.

**Mismatch detection:**

```python
MISMATCH_THRESHOLD = 0.25

err_vlm = (pedestrian_shade_score - human_norm).abs()

# Flag when VLM exceeds threshold
is_mismatch = err_vlm > MISMATCH_THRESHOLD
```

Target ~8–12 flagged images from 30 labels. Adjust threshold if needed to hit an inspectable count.

**Metadata surfaced per mismatch:**

| Field | Source | Purpose |
|-------|--------|---------|
| `scene_category` | `human_labels.csv` | Primary grouping — do failures cluster by scene type? |
| `hour` | `filtered_streetscapes.csv` | Time-of-day context for the capture |
| `sidewalk_pct` | `filtered_streetscapes.csv` | How much of the frame is walkable sidewalk |
| `heading` | `filtered_streetscapes.csv` | Camera orientation context |
| `lat`, `lon` | `filtered_streetscapes.csv` | Spatial position on corridor |
| VLM `shade_sources`, `reasoning`, `confidence` | `scores.csv` | Qualitative explanation |

**Summary table:** Mismatch counts grouped by `scene_category`.

**Visual gallery:** For each flagged image, display inline in the notebook:
1. Thumbnail from `data/images/exploration/{uuid}.jpeg`
2. Scores card: human, VLM, error
3. Metadata line: `scene_category`, hour, sidewalk %, heading
4. VLM reasoning (and optional author notes from `human_labels.csv`)
5. **Run stability** (if image is in Tier B sample): `run_std`, `run_range` from `run_variance.csv`

**Interpreting mismatches with variance:**

| Pattern | Interpretation |
|---------|----------------|
| High error vs human + low run variance | Likely real model/prompt failure |
| High error vs human + high run variance | Ambiguous image or unreliable score — soften the claim |
| Borderline mismatch + high run variance | Do not cite as a Tier C case study |

Group gallery sections by `scene_category` with markdown headers.

**Optional chart:** Bar chart of mean error by `scene_category` (only if enough mismatches per bucket).

### Dropped

| Item | Reason |
|------|--------|
| Composite baseline | Dropped — eval focuses on VLM vs human labels only |
| Prompt ablation (heading/sun context) | Out of time-budget scope |
| LLM-as-judge | Circular — VLM is the method under test |
| Segment/route-level eval | Image is the inference unit; aggregation is deterministic mean |
| Separate HTML gallery | Notebook handles Tier C visuals |

---

## Data Files

### `eval/human_labels.csv`

Hand-curated ground truth. Pre-fill `uuid` from images that have VLM scores; leave label columns empty for author to complete.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Join key to metadata and scores |
| `shade_1to5` | int (1–5) | Human shade rating; empty until labeled |
| `scene_category` | string | Scene type for Tier C grouping; assign while labeling (see options below) |
| `notes` | string | Optional free-text (e.g. "dappled light on far lane, VLM missed") |

**`scene_category` options** — pick the single best fit for the walkable path in the image:

| Value | When to use |
|-------|-------------|
| `tree_canopy` | Primary shade from street trees or vegetation over the path |
| `building_shadow` | Shade from adjacent buildings, walls, or overhangs |
| `covered_walkway` | Linkway, awning, bus shelter, or other built overhead cover |
| `open_exposure` | Little or no structural shade on the walkable path |
| `mixed_sources` | Multiple shade mechanisms; none clearly dominant |
| `ambiguous_path` | Walkable path or shade-on-path is hard to judge from the image |

Assign `scene_category` for every labeled image, not only mismatches — Tier C summary tables need non-mismatch rows for context. Aim for ~4–6 categories represented across the ~30-image set.

~30 rows. Select images that span varied shade conditions and `scene_category` values across the corridor.

### `eval/stability_sample.csv`

10 uuids drawn from `human_labels.csv` for Tier B. Written once; subset of the eval set.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Image to re-score |

### `eval/run_variance.csv`

Raw multi-run VLM outputs for Tier B.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Image id |
| `run_id` | int (1–3) | Run index |
| `pedestrian_shade_score` | float | Score for that run |
| `confidence` | string | VLM confidence for that run |
| `scored_at` | string | ISO timestamp |

### Inputs (existing)

| File | Key columns |
|------|-------------|
| `data/scores.csv` | `uuid`, `pedestrian_shade_score`, `shade_sources`, `confidence`, `reasoning` |
| `data/filtered_streetscapes.csv` | `uuid`, `lat`, `lon`, `heading`, `hour`, `sidewalk_pct`, `place` |
| `data/images/exploration/{uuid}.jpeg` | Image files for Tier C display |

### Join logic

Inner join on `uuid` across human labels, scores, and metadata. Eval set = labeled images that also have VLM scores. Report actual N in the notebook.

---

## Notebook Structure (`eval/evaluation.ipynb`)

Committed **with saved cell outputs** (images, tables, charts visible offline).

| Section | Content |
|---------|---------|
| **1. Intro** | Methodology summary, eval claim, limitations |
| **2. Setup** | Imports, load CSVs, join, compute `human_norm` |
| **3. Tier A** | VLM ρ and MAE; optional scatter (human vs VLM with 45° line) |
| **4. Tier B** | Run stability summary from `run_variance.csv`; per-image std/range table |
| **5. Tier C** | Mismatch flags; summary by `scene_category`; inline image gallery (incl. run_std where available) |
| **6. Takeaways** | Headline result, stability note, 1–2 category-level patterns, top failure modes |

**Run instructions** (for README):

```bash
uv run jupyter notebook eval/evaluation.ipynb
```

Re-execute and save outputs before submission when labels or scores change.

---

## README Integration

Eval section (~150–200 words):
1. What was measured and why (human labels as construct proxy)
2. Sample size and corridor context
3. Headline numbers (VLM ρ and MAE vs human labels)
4. One Tier C insight with link to `eval/evaluation.ipynb`
5. Limitations: single rater, N≈30, static snapshots, no solar geometry, single-run scores in Tier A (stability checked on 10-image subset)
6. One-line stability result: median run-to-run std from Tier B

PROCESS.md: labeling experience, surprising mismatches from Tier C gallery, whether high-variance images aligned with ambiguous cases.

---

## Dependencies

Already in `pyproject.toml`: `pandas`, `numpy`, `matplotlib`, `pillow`, `ipykernel`.

No new dependencies required. Spearman via `pandas.Series.corr(method="spearman")`.

---

## Error Handling

| Case | Behaviour |
|------|-----------|
| Labeled uuid missing from scores | Exclude from eval; print warning listing uuids |
| Labeled uuid missing from metadata | Exclude; print warning |
| Image file missing for mismatch | Show placeholder text in gallery cell; do not crash notebook |
| `run_variance.csv` missing or incomplete | Tier B section shows warning; Tier A/C still run |
| Fewer than 10 scored images | Notebook runs but README notes eval is preliminary |

---

## Out of Scope

- Inter-rater reliability (no second human rater)
- Full-grid run variance (30 images × 5 runs)
- Formal significance tests on run variance (N too small)
- Automated labeling or LLM-as-judge
- CI execution of the notebook
- Eval of segment aggregation or cool-route routing

---

## Success Criteria

1. Notebook runs end-to-end on committed data with saved outputs
2. Tier A reports VLM ρ and MAE against human labels
3. Tier B reports median run std and % high-variance images (10 × 3 runs)
4. Tier C gallery shows ≥5 mismatch cases with images, `scene_category`, hour, and sidewalk %
5. README links to notebook and states methodology honestly
6. Results may show VLM losing on some strata — that is acceptable and expected

---

## Time Budget

| Task | Estimate |
|------|----------|
| Hand-label 30 images | 2–3 hrs |
| Collect run variance (10 img × 3 runs) | ~45 min |
| Build notebook (Tiers A, B) | 1–1.5 hrs |
| Tier C gallery + category summary | 1–1.5 hrs |
| Execute, save outputs, README link | 30 min |
| **Total** | **~6–7 hrs** |
