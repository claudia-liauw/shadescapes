# Evaluation Design: Shade Index VLM

**Date:** 2026-06-19  
**Status:** Approved  
**Related:** [instructions.md](./instructions.md), [vlm-approach.md](./vlm-approach.md)

---

## Problem

The GovTech brief requires an honest evaluation with a justified methodology. We need to demonstrate that VLM `pedestrian_shade_score` measures functional pedestrian shade in a way that aligns with human judgment.

**Primary claim:** VLM shade scores agree with human judgment on a hand-labeled sample of streetscape images.

**Unit of evaluation:** Individual streetscape images (~30 hand-labeled, curated from the `sample/` image pool, optionally supplemented by ~6–8 synthetic gap-fill images).

**Not claiming:** Island-wide accuracy, time-of-day precision, segment/route-level performance, synthetic-image realism, or that the VLM is ground truth.

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

### Image directories

| Directory | Count | Role |
|-----------|-------|------|
| `data/images/sample/` | ~50 | **Eval image pool** — full Mapillary download for the corridor |
| `data/images/exploration/` | ~9 | **Demo only** — subset of `sample/`; `config.IMAGES_DIR` for the FastAPI app |
| `data/images/synthetic/` | 0–8 | **Optional gap-fill** — generated images for Tier C stress tests |

`exploration/` is not the eval set. The app scores and maps the 9 demo images; eval uses a separate curated list.

### Eval set selection

The eval does **not** auto-pick images from a folder. **`eval/human_labels.csv` defines the eval set** (~30 uuids chosen by the author from `sample/`, plus any synthetic uuids).

```text
filtered_streetscapes.csv (~53 metadata rows)
        │
data/images/sample/ (~50 images) ──► author picks ~30 uuids
        │                                    │
        │                                    ▼
        │                          eval/human_labels.csv  ◄── eval set
        │                                    │
eval/score_eval.py (or notebook cell)        │ join
  scores uuids in human_labels               │
  from sample/ + synthetic/                  ▼
        │                          eval/scores.csv
        └──────────────────────────────────► evaluation.ipynb

data/images/exploration/ (~9) ──► FastAPI demo ──► data/scores.csv (production)
```

Keep the full `sample/` corpus (~50). Do not reduce it to 30 — the ~30 is a **labeling budget**, not "all committed images."

### Data flow

```text
human_labels.csv ──────────────┐
eval/scores.csv ───────────────┤
run_variance.csv ──────────────┼──► evaluation.ipynb ──► saved outputs
filtered_streetscapes.csv ─────┤
synthetic_streetscapes.csv ────┘
data/images/sample/{uuid}.jpeg ──────► Tier C display (real)
data/images/synthetic/{uuid}.jpeg ───► Tier C display (synthetic)

eval/synthetic_prompts.csv ──► eval/generate_images.py ──► synthetic images + metadata
```

No separate eval service or API. The notebook imports `METADATA_CSV` from `src.config` and reads `eval/scores.csv` for VLM outputs — **not** `data/scores.csv` (production/demo). For image paths, use `data/images/sample/` (real) and `data/images/synthetic/` — **not** `config.IMAGES_DIR` (`exploration/`).

Eval scoring uses `src.score.score_image` with the same prompt/model as production, but reads images from `sample/` or `synthetic/` via `eval/score_eval.py` (or an equivalent notebook cell). Do not use `run_scoring()` — that only discovers images in `exploration/`.

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

**Scores used:** Single-run values from `eval/scores.csv` for uuids in `human_labels.csv`, written by `eval/score_eval.py` (same `score_image` function as production). Run stability is assessed separately in Tier B.

**Scope:** Tier A headline metrics use **real** Mapillary images only. Synthetic gap-fill images (see below) are excluded from ρ and MAE unless explicitly noted as a supplementary table.

### Tier B — Run stability (VLM reliability)

Measures whether the VLM returns **consistent** scores for the same image + prompt across repeated API calls. This is distinct from Tier A accuracy (correctness vs human).

**Protocol:**
- Select **10 images** from `human_labels.csv`, spanning varied shade levels and `scene_category` values
- Score each image **3 times** from `data/images/sample/{uuid}.jpeg` using `src.score.score_image`
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
| `source` | metadata CSV | `mapillary` or `synthetic` — flag generated images |
| `hour` | metadata CSV | Time-of-day context for the capture |
| `sidewalk_pct` | metadata CSV | How much of the frame is walkable sidewalk |
| `heading` | metadata CSV | Camera orientation context |
| `lat`, `lon` | metadata CSV | Spatial position on corridor (placeholder for synthetic) |
| VLM `shade_sources`, `reasoning`, `confidence` | `eval/scores.csv` | Qualitative explanation |

**Summary table:** Mismatch counts grouped by `scene_category`.

**Visual gallery:** For each flagged image, display inline in the notebook:
1. Thumbnail from `data/images/sample/{uuid}.jpeg` or `data/images/synthetic/{uuid}.jpeg`
2. Scores card: human, VLM, error, `source`
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

## Synthetic gap-fill images

The `sample/` pool (~50 images) is skewed toward campus/parking scenes, afternoon hours (14:00–15:00), and low `sidewalk_pct` (median ~2.5%). Several `scene_category` values and known VLM failure modes are underrepresented. Generate a small set of synthetic streetscapes to stress-test Tier C without inflating Tier A claims.

### Gaps in real data

| Signal | What we have | Likely missing |
|--------|--------------|----------------|
| **Time** | Mostly 14:00–15:00 | Low sun (7–9), harsh noon (12), long afternoon shadows (17–18) |
| **Framing** | Low `sidewalk_pct`; road-centric | Walkable path dominates lower frame (~30–50%) |
| **Setting** | Campus, parking, hospital | HDB linkway, shopfront awning, bus stop, urban street corridor |
| **`scene_category`** | Sparse institutional outdoor shots | Several categories below may be absent |

### Edge cases to generate (first pass: 6–8 images)

| `scene_category` | Edge case | Why generate |
|------------------|-----------|--------------|
| `covered_walkway` | HDB sheltered linkway, MRT overhead cover | Singapore-specific; rare in campus captures |
| `building_shadow` | Narrow gap between towers, deep wall shadow on sidewalk | Urban-canyon geometry |
| `tree_canopy` | Street-tree row with **dappled** shade on path | Park lawn trees ≠ functional street shade |
| `open_exposure` | Wide arterial, no trees, midday glare | Path-focused open scene |
| `mixed_sources` | Half building shadow, half open sky on same path | Partial-shade judgment |
| `ambiguous_path` | Shared path, construction barriers, unclear walk zone | Classic VLM failure mode |

**Additional traps** (1–2 images; map to nearest `scene_category`):

| Trap | Notes |
|------|-------|
| **Off-path shade** | Trees shade road/lawn; sidewalk exposed — VLM often over-scores |
| **Distant canopy** | Green overhead in background; path in sun |
| **Localized shelter** | Bus stop roof or umbrella shading a small pocket only |
| **High sidewalk %** | Camera low; path fills bottom third of frame |

### Component: `eval/generate_images.py`

CLI entrypoint. Reads `eval/synthetic_prompts.csv`, calls an image-generation model (Gemini Imagen via existing `google-genai` client), writes images and metadata. Skips existing uuids unless `--force`.

```bash
uv run python -m eval.generate_images
uv run python -m eval.generate_images --uuid syn-linkway-01
```

**Responsibilities:**

1. Read prompt rows from `eval/synthetic_prompts.csv`
2. Generate image via API; save to `data/images/synthetic/{uuid}.jpeg`
3. Append metadata row to `data/synthetic_streetscapes.csv` (`source=synthetic`)
4. Print summary of created/skipped uuids

**Prompt template** (keep synthetic images comparable):

```text
Photorealistic street-level photograph, eye height ~1.5m, Singapore tropical setting.
{prompt}
Concrete walkable path visible in the lower third of the frame.
Afternoon lighting consistent with {hour}:00.
No text overlays, no watermarks, no people.
```

### `eval/synthetic_prompts.csv`

Author-curated gap list. `scene_category` pre-fills `human_labels.csv`.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | e.g. `syn-linkway-01` |
| `scene_category` | string | One of the `scene_category` options above |
| `prompt` | string | Scene description passed to image model |
| `hour` | int | Metadata for scoring prompt and Tier C display |
| `heading` | float | Camera bearing (degrees) |
| `sidewalk_pct` | float | Author estimate for metadata (0–1) |
| `notes` | string | Intended failure mode or eval purpose |

Example rows:

```csv
uuid,scene_category,prompt,hour,heading,sidewalk_pct,notes
syn-linkway-01,covered_walkway,"Sheltered HDB walkway with overhead cover, concrete sidewalk in foreground",14,90,0.35,full overhead cover
syn-offpath-01,tree_canopy,"Large trees shade the road lane but sidewalk on right is in direct sun",14,180,0.25,VLM trap - shade not on path
syn-ambiguous-01,ambiguous_path,"Shared path with faded markings, unclear where pedestrians should walk",15,45,0.20,path ambiguity
```

### `data/synthetic_streetscapes.csv`

Same schema as `filtered_streetscapes.csv` plus `source=synthetic`. Use placeholder `lat`/`lon` on the corridor (or leave empty); not used for map demo.

| Column | Notes |
|--------|-------|
| `uuid` | Matches `synthetic_prompts.csv` |
| `source` | Always `synthetic` |
| `orig_id` | Empty |
| `lat`, `lon` | Optional placeholder coordinates |
| `hour`, `heading`, `sidewalk_pct` | Copied from prompt file |
| `place` | Descriptive tag, e.g. `synthetic_linkway` |

### Workflow

1. Author fills `eval/synthetic_prompts.csv` for missing categories
2. Run `eval/generate_images.py`; review images manually — re-roll or edit prompts if artifacts are obvious
3. Add synthetic uuids to `eval/human_labels.csv` (pre-fill `scene_category`; label `shade_1to5`)
4. Run `eval/score_eval.py` to score all uuids in `human_labels.csv` from `sample/` and `synthetic/`; write to `eval/scores.csv`
5. Notebook joins real + synthetic metadata via `pd.concat`; filter by `source` for Tier A vs Tier C

**Framing:** Synthetic images are supplementary stress tests for Tier C category coverage. Disclose in README limitations; do not include in map demo unless explicitly desired.

---

## Data Files

### `eval/human_labels.csv`

**Defines the eval set.** Author curates ~30 uuids from `data/images/sample/` (plus optional synthetic uuids). This file is the single source of truth for which images are evaluated — not the image directories themselves.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Join key to metadata and scores |
| `shade_1to5` | int (1–5) | Human shade rating; empty until labeled |
| `scene_category` | string | Scene type for Tier C grouping; assign while labeling (see options below) |
| `notes` | string | Optional free-text (e.g. "dappled light on far lane, VLM missed") |

**How to pick the ~30:** Browse `data/images/sample/`. Select uuids that span varied shade conditions and `scene_category` values. Prefer images with matching rows in `filtered_streetscapes.csv`. Skip uuids with no image file in `sample/`. The remaining ~20 images in `sample/` stay outside eval unless added here later.

Synthetic images: add uuids from `eval/synthetic_prompts.csv` after generation; `scene_category` can be pre-filled from that file.

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

### `eval/score_eval.py`

Scores eval images only — uuids listed in `human_labels.csv` that are not yet in `eval/scores.csv` (or all, with `--force`).

| Input | Path |
|-------|------|
| Labels | `eval/human_labels.csv` |
| Real images | `data/images/sample/{uuid}.jpeg` |
| Synthetic images | `data/images/synthetic/{uuid}.jpeg` |
| Metadata | `filtered_streetscapes.csv` + `synthetic_streetscapes.csv` |
| Output | `eval/scores.csv` |

```bash
uv run python -m eval.score_eval
```

Uses `src.score.score_image` (same prompt/model as production). Writes to `eval/scores.csv` only — does not read or modify `data/scores.csv`, `exploration/`, or the demo scoring path.

### `eval/scores.csv`

VLM outputs for the eval set. Same schema as `data/scores.csv` but kept separate from production/demo scores.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Join key to `human_labels.csv` |
| `pedestrian_shade_score` | float | VLM shade score (0–1) |
| `shade_sources` | string | JSON array |
| `confidence` | string | `low` / `medium` / `high` |
| `reasoning` | string | VLM explanation |
| `scored_at` | string | ISO timestamp |

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

| File | Key columns | Used by |
|------|-------------|---------|
| `data/scores.csv` | `uuid`, `pedestrian_shade_score`, … | Demo app only |
| `eval/scores.csv` | same schema | Eval notebook (Tier A, C) |
| `data/filtered_streetscapes.csv` | `uuid`, `lat`, `lon`, `heading`, `hour`, `sidewalk_pct`, `place` | Eval + demo |
| `data/synthetic_streetscapes.csv` | Same schema; `source=synthetic` | Eval only |
| `data/images/sample/{uuid}.jpeg` | Real eval images | Eval |
| `data/images/synthetic/{uuid}.jpeg` | Synthetic eval images | Eval |
| `data/images/exploration/{uuid}.jpeg` | Demo only | Demo app |

### Join logic

Concatenate `filtered_streetscapes.csv` and `synthetic_streetscapes.csv` into a single metadata frame (add `source` column: `mapillary` / `synthetic`).

**Eval set** = inner join on `uuid` across `human_labels.csv`, `eval/scores.csv`, and metadata. Only uuids present in `human_labels.csv` enter Tier A/B/C. Report actual N and `source` breakdown in the notebook.

Image lookup for gallery: `sample/{uuid}.jpeg`, falling back to `synthetic/{uuid}.jpeg`.

---

## Notebook Structure (`eval/evaluation.ipynb`)

Committed **with saved cell outputs** (images, tables, charts visible offline).

| Section | Content |
|---------|---------|
| **1. Intro** | Methodology summary, eval claim, limitations |
| **2. Setup** | Imports, load CSVs (`eval/scores.csv`, not `data/scores.csv`), concat metadata, join, compute `human_norm` |
| **3. Tier A** | VLM ρ and MAE on real images; optional scatter (human vs VLM with 45° line) |
| **4. Tier B** | Run stability summary from `run_variance.csv`; per-image std/range table |
| **5. Tier C** | Mismatch flags; summary by `scene_category`; inline gallery (real + synthetic, incl. `source`) |
| **6. Takeaways** | Headline result, stability note, category-level patterns, synthetic gap-fill note |

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
5. Limitations: single rater, N≈30 real images, static snapshots, no solar geometry, single-run scores in Tier A (stability checked on 10-image subset), optional synthetic gap-fill images excluded from headline metrics
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
| Labeled uuid missing from `eval/scores.csv` | Exclude from eval; print warning listing uuids |
| Labeled uuid missing from metadata | Exclude; print warning |
| Labeled uuid missing image in `sample/` or `synthetic/` | Exclude from eval; print warning |
| Image file missing for mismatch | Show placeholder text in gallery cell; do not crash notebook |
| `synthetic_streetscapes.csv` missing | Tier C runs on real images only; print warning |
| `generate_images.py` API failure | Skip uuid; print error; do not write partial metadata |
| `eval/scores.csv` missing or empty | Notebook runs through setup but Tier A/C show warning; prompt to run `eval/score_eval.py` |
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
2. Tier A reports VLM ρ and MAE against human labels (real images only)
3. Tier B reports median run std and % high-variance images (10 × 3 runs)
4. Tier C gallery shows ≥5 mismatch cases with images, `scene_category`, hour, sidewalk %, and `source`
5. Optional: 6–8 synthetic images generated and labeled for underrepresented `scene_category` values
6. README links to notebook and states methodology honestly
7. Results may show VLM losing on some strata — that is acceptable and expected

---

## Time Budget

| Task | Estimate |
|------|----------|
| Hand-label 30 images | 2–3 hrs |
| Generate + label 6–8 synthetic images | 1–1.5 hrs |
| Collect run variance (10 img × 3 runs) | ~45 min |
| Build `eval/score_eval.py` + `eval/generate_images.py` | 1 hr |
| Build notebook (Tiers A, B) | 1–1.5 hrs |
| Tier C gallery + category summary | 1–1.5 hrs |
| Execute, save outputs, README link | 30 min |
| **Total** | **~7–8 hrs** |
