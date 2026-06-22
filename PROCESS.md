# Build process

How ShadeScapes was put together over a two-day prototype sprint — what worked, what was dropped, and where judgment mattered more than automation.

---

## Dataset curation

The instructions to download the images required that I first had metadata downloaded from [NUS Global Streetscapes](https://huggingface.co/datasets/NUS-UAL/global-streetscapes) on Hugging Face. The full dataset was over 70 GB (tabular data only), so it was clear that I had to be selective about what I downloaded. I passed the schema to AI for an initial pass on which fields looked useful for a Singapore shade study, but its suggestions weren't that great. Therefore, I explored the actual columns in the data before settling on the metadata filters documented in the README. I downloaded the 2 GB parquet file and used DuckDB SQL to load only required rows into memory. I further filtered spatially to ~50 rows for demo, then saved to CSV for use with the [scripts from the provider](https://github.com/ualsg/global-streetscapes).

---

## Architecture

I manually tested out some components (Gemini API and Folium map) and sketched the architecture in [`docs/concept.md`](docs/concept.md). Initially I thought to download Singapore's geodata and do a spatial join with the points or a map from OpenStreetMap (`osmnx`) and display with `plotly`, but Folium was the better approach.

This architecture was passed to AI for spec and plan generation as part of spec-driven development.

---

## Implementation and code quality

Wrote code and tests iteratively, reviewing each chunk as it landed. Ran an AI code review pass midway through, then manually thought of test gaps and guardrails:

- Missing images / missing metadata CSV / images that are not in metadata CSV / metadata CSV without certain columns
- When AI generated a plan that used the same CSV for production and eval, I noticed and questioned it.
- Refactoring eval to call shared `src/score.py` helpers (`score_image`, `run_scoring_batch`) instead of duplicating Gemini logic

---

## Performance

First sequential scoring run was painfully slow on the free API tier. Added parallel batching via `ThreadPoolExecutor`, capped at 15 concurrent requests/min to stay within quota.

---

## Evaluation — pivots

### GVI / SVI baseline → human agreement + stability

Early plan: compare VLM `pedestrian_shade_score` against **Green View Index** and **Sky View Index** from the dataset as a cheap baseline. Inspection showed why that was weak — GVI/SVI measure greenery and open sky in the frame, not functional shade on the walkable path at a given hour. A tree-lined road can score high on GVI while the sidewalk bakes. Discarded in favour of measuring variance over repeated calls + error analysis to avoid cluttering the evaluation.

### `place` metadata → manual `scene_category` labels

Wanted to slice errors by the dataset’s `place` field (campus, hospital, etc.). Spot-checking images showed the labels were unreliable — grass tagged as picnic area, parking lots as campus, and so on. Not trustworthy enough for error analysis.

Manually labeled each eval image with `scene_category` (`tree_canopy`, `building_shadow`, `covered_walkway`, `open_exposure`, `mixed_sources`, `ambiguous_path`) alongside a 1–5 afternoon shade rating in `eval/data/human_labels.csv`.

### Synthetic gap-fill

Browsing labels revealed holes: almost no real `building_shadow` examples, few true covered linkways, low `sidewalk_pct` across the Mapillary pool. Wrote prompts for seven edge-case images and planned a `eval/generate_images.py` script.

No free image-generation API in the Gemini stack (and image gen wasn’t the core of the project), so I didn’t push further on automation. Generated the seven synthetic JPEGs manually in Google AI Studio (Nano Banana), saved under `data/images/synthetic/`, and labeled them like the real set. 

---

## Other dropped ideas

**Shaded walking routes** — The original concept in [`docs/concept.md`](docs/concept.md) included “Cool Routes” on OneMap: pick a path optimised for afternoon shade. That needs segment-level routing, time-of-day modelling, and graph search — out of scope for a two-day prototype. Left as a deployment footnote in the README.

---

## Results and write-up

Interpreted eval outputs by hand — which categories drove mismatches and where the rater was also uncertain. `README.md` and this document were generated from my notes in [`docs/concept_docs.md`](docs/concept_docs.md) and further edited.
