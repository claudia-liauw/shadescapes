# Build process

How ShadeScapes was put together over a two-day prototype sprint — what worked, what was dropped, and where judgment mattered more than automation.

---

## Dataset curation

Started from the [Global Streetscapes](https://huggingface.co/datasets/NUS-UAL/global-streetscapes) dataset on Hugging Face: read the dataset card and docs, then used AI for an initial pass on which fields looked useful for a Singapore shade study.

That first cut was too loose. I explored the actual columns in the data before settling on the metadata filters documented in the README. A spatial cluster around one western corridor brought the working set down to ~53 metadata rows and ~50 downloaded images.

---

## Architecture

Sketched the system in [`docs/concept.md`](docs/concept.md) after brainstorming with AI: FastAPI backend, batch VLM scoring to `scores.csv`, Folium map joined on `uuid`, minimal HTML frontend with a single “Run Shade Scoring” button.

The final shape was hand-edited rather than generated wholesale — especially the split between demo scoring (`data/scores.csv`, exploration images) and eval scoring (`eval/data/scores.csv`, curated sample set) so reviewers could re-run the demo without touching eval artifacts.

---

## Implementation and code quality

Wrote code and tests iteratively, reviewing each chunk as it landed. Ran an AI code review pass midway through, then kept a running list of test gaps and guardrails:

- Missing images / missing metadata CSV / metadata CSV without certain columns
- Refactoring eval to call shared `src/score.py` helpers (`score_image`, `run_scoring_batch`) instead of duplicating Gemini logic

---

## Performance

First sequential scoring run was painfully slow on the free API tier. Added parallel batching via `ThreadPoolExecutor`, capped at 15 concurrent requests/min to stay within quota.

---

## Evaluation — pivots

### GVI / SVI baseline → human agreement + stability

Early plan: compare VLM `pedestrian_shade_score` against **Green View Index** and **Sky View Index** from the dataset as a cheap baseline. Inspection showed why that was weak — GVI/SVI measure greenery and open sky in the frame, not functional shade on the walkable path at a given hour. A tree-lined road can score high on GVI while the sidewalk bakes.

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

Interpreted eval outputs by hand — which categories drove mismatches, where the rater was also uncertain, whether high VLM confidence lined up with human agreement. Handcrafted limitations.
