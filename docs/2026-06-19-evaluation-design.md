# Evaluation Design: Shade Index VLM

**Date:** 2026-06-19  
**Status:** Approved  
**Related:** [instructions.md](./instructions.md), [vlm-approach.md](./vlm-approach.md)

---

## Problem

The GovTech brief requires an honest evaluation with a justified methodology. We need to demonstrate that VLM `pedestrian_shade_score` measures functional pedestrian shade better than a naive proxy derived from existing Global Streetscapes indices.

**Primary claim:** VLM shade scores agree with human judgment better than a composite of `green_view_index` and `sky_view_index`.

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

**Scores used:** Single-run values from `data/scores.csv` (the production scoring path). Run stability is assessed separately in Tier A½.

### Tier A½ — Run stability (VLM reliability)

Measures whether the VLM returns **consistent** scores for the same image + prompt across repeated API calls. This is distinct from Tier A accuracy (correctness vs human).

**Protocol:**
- Select **10 images** from the eval set (`human_labels.csv`), spanning varied shade levels and `place` values
- Score each image **3 times** with the same prompt/model as production (`src.score.score_image`)
- Store raw runs in `eval/run_variance.csv`

**Metrics (descriptive, no significance tests at N=10):**

| Metric | Definition |
|--------|------------|
| **Median std** | Median per-image standard deviation across 3 runs |
| **Max range** | Worst-case `max − min` across runs for any image |
| **% high-variance** | Share of images where range > `0.15` |
| **Test–retest Spearman** | Rank correlation between run 1 and run 2 scores across the 10 images |

**Framing:** Composite baseline has zero run variance by construction. Stability is a VLM-only deployment concern — a cool route should not flip on re-scoring.

**Scope limit:** 10 images × 3 runs = 30 API calls. Do not expand to full 30-image × 5-run grid.

### Tier B — Composite baseline comparison

One baseline only — the simplest reasonable proxy from existing dataset fields.

| Method | Formula |
|--------|---------|
| **Composite** | `0.5 * (1 - sky_view_index) + 0.5 * green_view_index` |
| **Ours** | VLM `pedestrian_shade_score` |

Run the same Tier A metrics (Spearman ρ, MAE) for both methods against human labels. Present a side-by-side summary table.

**Framing:** The composite represents what a planner might derive from Global Streetscapes without a VLM. The comparison is honest, not a strawman.

### Tier C — Mismatch analysis with place metadata and visuals

Qualitative error analysis for cases where either method disagrees with human judgment.

**Mismatch detection:**

```python
MISMATCH_THRESHOLD = 0.25

err_vlm = (pedestrian_shade_score - human_norm).abs()
err_composite = (composite_score - human_norm).abs()

# Flag when EITHER method exceeds threshold
is_mismatch = (err_vlm > MISMATCH_THRESHOLD) | (err_composite > MISMATCH_THRESHOLD)

# Sub-classify
# vlm_miss:       err_vlm > threshold and err_composite <= threshold
# composite_miss: err_composite > threshold and err_vlm <= threshold
# both_miss:      both > threshold
```

Target ~8–12 flagged images from 30 labels. Adjust threshold if needed to hit an inspectable count.

**Metadata surfaced per mismatch** (from `filtered_streetscapes.csv`):

| Field | Purpose |
|-------|---------|
| `place` | Primary grouping — do failures cluster by land use? |
| `green_view_index`, `sky_view_index` | Explain composite behaviour |
| `heading` | Camera orientation context |
| `lat`, `lon` | Spatial position on corridor |
| VLM `shade_sources`, `reasoning`, `confidence` | Qualitative explanation (from `scores.csv`) |

**Summary table:** Mismatch counts grouped by `place` and `miss_type`.

**Visual gallery:** For each flagged image, display inline in the notebook:
1. Thumbnail from `data/images/exploration/{uuid}.jpeg`
2. Scores card: human, VLM, composite, per-method errors
3. Metadata line: place, GVI, SVI, heading
4. VLM reasoning (and optional author notes from `human_labels.csv`)
5. **Run stability** (if image is in Tier A½ sample): `run_std`, `run_range` from `run_variance.csv`

**Interpreting mismatches with variance:**

| Pattern | Interpretation |
|---------|----------------|
| High error vs human + low run variance | Likely real model/prompt failure |
| High error vs human + high run variance | Ambiguous image or unreliable score — soften the claim |
| Borderline mismatch + high run variance | Do not cite as a Tier C case study |

Group gallery sections by `miss_type` or `place` with markdown headers.

**Optional chart:** Bar chart of mean error by `place` (only if enough mismatches per bucket).

### Dropped

| Item | Reason |
|------|--------|
| Individual SVI / GVI baselines | Composite subsumes them |
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
| `notes` | string | Optional free-text (e.g. "linkway, composite missed") |

~30 rows. Select images that span varied shade conditions and `place` values across the corridor.

### `eval/stability_sample.csv`

10 uuids drawn from `human_labels.csv` for Tier A½. Written once; subset of the eval set.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Image to re-score |

### `eval/run_variance.csv`

Raw multi-run VLM outputs for Tier A½.

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
| `data/filtered_streetscapes.csv` | `uuid`, `lat`, `lon`, `heading`, `green_view_index`, `sky_view_index`, `place` |
| `data/images/exploration/{uuid}.jpeg` | Image files for Tier C display |

### Join logic

Inner join on `uuid` across human labels, scores, and metadata. Eval set = labeled images that also have VLM scores. Report actual N in the notebook.

---

## Notebook Structure (`eval/evaluation.ipynb`)

Committed **with saved cell outputs** (images, tables, charts visible offline).

| Section | Content |
|---------|---------|
| **1. Intro** | Methodology summary, eval claim, limitations |
| **2. Setup** | Imports, load CSVs, join, compute `composite_score` and `human_norm` |
| **3. Tier A½** | Run stability summary from `run_variance.csv`; per-image std/range table |
| **4. Tier A** | VLM ρ and MAE; optional scatter (human vs VLM with 45° line) |
| **5. Tier B** | Composite ρ and MAE; side-by-side comparison table |
| **6. Tier C** | Mismatch flags; summary by `place`; inline image gallery (incl. run_std where available) |
| **7. Takeaways** | Headline result, stability note, 1–2 place-level patterns, top failure modes |

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
3. Headline numbers (VLM vs composite ρ and MAE)
4. One Tier C insight with link to `eval/evaluation.ipynb`
5. Limitations: single rater, N≈30, static snapshots, no solar geometry, single-run scores in Tier A (stability checked on 10-image subset)
6. One-line stability result: median run-to-run std from Tier A½

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
| `run_variance.csv` missing or incomplete | Tier A½ section shows warning; Tier A/B/C still run |
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
2. Tier A½ reports median run std and % high-variance images (10 × 3 runs)
3. Tier A and B produce a comparison table (VLM vs composite)
4. Tier C gallery shows ≥5 mismatch cases with images and `place` metadata
5. README links to notebook and states methodology honestly
6. Results may show VLM losing on some strata — that is acceptable and expected

---

## Time Budget

| Task | Estimate |
|------|----------|
| Hand-label 30 images | 2–3 hrs |
| Collect run variance (10 img × 3 runs) | ~45 min |
| Build notebook (Tiers A½, A–B) | 1–1.5 hrs |
| Tier C gallery + place summary | 1–1.5 hrs |
| Execute, save outputs, README link | 30 min |
| **Total** | **~6–7 hrs** |
