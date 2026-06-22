# Build process

How ShadeScapes was put together over a two-day prototype sprint — what worked, what was dropped, and where judgment mattered more than automation.

---

## Dataset curation

The instructions to download the images required that I first had metadata downloaded from [NUS Global Streetscapes](https://huggingface.co/datasets/NUS-UAL/global-streetscapes) on Hugging Face. The full dataset was over 70 GB (tabular data only), so it was clear that I had to be selective about what I downloaded. I fed the schema to AI for an initial pass on which fields looked useful for a Singapore shade study, but its suggestions weren't that great. Therefore, I explored the actual columns in the data as well as some sample images before settling on the metadata filters documented in the README. I downloaded a 2 GB parquet file and used DuckDB SQL to load only required rows into memory. I further filtered spatially to ~50 rows for demo, then saved to CSV for use with the [scripts from the provider](https://github.com/ualsg/global-streetscapes).

Initially I wanted to have image download as part of the workflow, but it was difficult to resolve conflicting dependencies. Thus, I did the downloading in a separate repository and chose to commit a small set of images for demo instead, given that a real workflow will not use these images.

---

## Architecture

I manually tested out some components (Gemini API and Folium map) and sketched the architecture in [`docs/concept.md`](docs/concept.md). Initially I thought to download Singapore's geodata or a map from OpenStreetMap (`osmnx`) and do a spatial join with the points and display with `plotly`, but Folium was the better approach as it had map details.

This architecture was passed to AI for spec and plan generation as part of spec-driven development.

---

## Coding

Composer 2.5 was used for more complex tasks like design, planning, and debugging. Codex 5.1 Mini (a cheaper model), was used for implementation.

Worked in a loose PR workflow (not strictly one issue one PR and sometimes committing directly to main) to reduce friction for a one-person two-day project.

Wrote code and tests iteratively, reviewing each chunk as it landed. Ran an AI code review pass midway through, then manually thought of more test gaps and guardrails and other design issues:

- Missing images / missing metadata CSV / images that are not in metadata CSV / metadata CSV without certain columns
- First sequential scoring run was painfully slow on the free API tier. Added parallel batching, capped at 15 concurrent requests/min to stay within quota.
- When AI generated a plan that used the same CSV for production and eval, I noticed and questioned it.
- Refactoring eval to call shared `src/score.py` helpers (`score_image`, `run_scoring_batch`) instead of duplicating Gemini logic

---

## Evaluation

### Manual labelling of shade score

Initially I thought to evaluate against `green_view_index` (GVI) or `sky_view_index` (SVI) in the dataset (proportion of greenery/sky), but I realised that didn't correlate directly with shade. Thus I decided to manually label the shade score to keep the scope manageable. From my experience, LLM-as-a-judge does not work out of the box and would likely require its own set of evaluation and calibration.

### GVI / SVI baseline → stability

Despite not using GVI and SVI as direct evaluation, I initially thought I could retain that as a baseline comparison to enrich evaluation beyond simple correlation. I realised the evaluation lacked a stability study that checked score agreement between runs and added that in. With correlation, stability and error analysis (with category labels and image generation - see below), I felt I had enough scope for evaluation and discarded the weak baseline idea. I didn't necessarily expect the shade score to correlate with GVI/SVI (see above), thus it did not serve any meaningful signal.

### `place` metadata → manual `scene_category` labels

Wanted to slice errors by the dataset’s `place` field (campus, hospital, etc.). Spot-checking images showed the labels were unreliable, such as grass tagged as picnic area.

Manually labeled each eval image with `scene_category` (`tree_canopy`, `building_shadow`, `covered_walkway`, `open_exposure`, `mixed_sources`, `ambiguous_path`) on top of the 1–5 afternoon shade rating. Again, this was done manually to keep the project in scope without having to invoke further evaluation.

### Synthetic gap-fill

After labelling `scene_category`, I realised there were no `building_shadow` examples and few covered linkways across the real image pool. Planned to generate synthetic images to cover edge cases via API, but there was no free image-generation API in the Gemini stack (and image gen wasn’t the core of the project), so I didn’t push further on automation. Generated the seven synthetic images manually in Google AI Studio (Nano Banana). 

---

## Other dropped ideas

**Shaded walking routes** — The original concept included “Cool Routes” on OneMap: pick a path optimised for afternoon shade. That needs segment-level routing, time-of-day modelling, and graph search — out of scope for a two-day prototype. Left as a deployment footnote in the README.

---

## Results and write-up

Interpreted eval outputs by hand — which categories drove mismatches and where the rater was also uncertain. `README.md` and this document were generated from my notes in [`docs/concept_docs.md`](docs/concept_docs.md) and further edited.
