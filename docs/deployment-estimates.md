# Deployment compute and cost estimates

Back-of-envelope figures cited in the README **Deployment considerations** section. None of these were profiled or logged automatically in the repo — they are planning assumptions documented here for transparency.

**Related:** [README — Deployment considerations](../README.md#deployment-considerations), [`src/score.py`](../src/score.py) (batching and rate limits).

---

## Batch latency (~2–3 s for 15 images)

**Source:** Observed during development (not from a benchmark script in-repo).

**How scoring works:**

- `BATCH_SIZE = 15` — up to 15 Gemini calls run in parallel per batch (`ThreadPoolExecutor` in `src/score.py`).
- Wall-clock per batch ≈ latency of the **slowest** call in that batch, not 15× a single call.

So if one VLM request takes ~2–3 s, a full 15-image batch finishes in roughly the same time.

---

## Demo wall-clock vs inference time

The demo feels much slower than 2–3 s per batch because of the free-tier rate limit:

| Constant | Value |
| -------- | ----- |
| `RATE_LIMIT_MAX_REQUESTS_PER_MINUTE` | 15 |
| `RATE_LIMIT_PERIOD_SECONDS` | 60 |

After each batch, the app sleeps `60 s − batch_duration` before starting the next. Demo throughput is therefore capped at **one batch per minute**, regardless of how fast inference is.

**Example — 50 sample images:**

| | |
| - | - |
| Batches | ⌈50 ÷ 15⌉ = **4** |
| Inference only | 4 × ~2–3 s ≈ **8–12 s** |
| Demo wall-clock | 4 × ~60 s ≈ **~4 min** |

Production with a higher API quota could run batches back-to-back at the ~2–3 s cadence instead of waiting a full minute between batches.

---

## Island-scale time (“tens of hours”)

**Assumption:** ~500,000 street-level images — an illustrative Singapore-scale order of magnitude, **not** measured from this repo’s 50-image demo set.

**Formula (no rate-limit waits):**

```
total_time ≈ (image_count ÷ batch_size) × seconds_per_batch
```

| Step | Calculation |
| ---- | ----------- |
| Images | 500,000 |
| Batch size | 15 |
| Batches | 500,000 ÷ 15 ≈ **33,300** |
| At 2 s/batch | 33,300 × 2 ≈ **18.5 h** |
| At 3 s/batch | 33,300 × 3 ≈ **27.8 h** |

Hence **“tens of hours”** (~20–30 h of API time if batches run continuously).

---

## API cost (“thousands of dollars”)

**Not calculated in-repo.** Order-of-magnitude estimate from token-based pricing.

**Model:** `gemini-3.1-flash-lite` (see [`src/config.py`](../src/config.py)).

**Published rates** ([Gemini API pricing](https://ai.google.dev/gemini-api/docs/gemini-3), as of project build):

| | Rate |
| - | ---- |
| Input (text, image, video) | $0.25 / 1M tokens |
| Output | $1.50 / 1M tokens |

**Per-image rough token budget:**

| Component | Tokens (estimate) |
| --------- | ----------------- |
| JPEG + prompt | ~500–2,000 (resolution-dependent) |
| JSON response | ~50–200 |

**Example at ~1,500 input + 100 output tokens:**

| | |
| - | - |
| Input cost | 1,500 × $0.25/1M ≈ **$0.000375** |
| Output cost | 100 × $1.50/1M ≈ **$0.00015** |
| **Per image** | ≈ **$0.0005** |
| **500k images** | ≈ **$250** |

**“Thousands of dollars”** is a conservative planning range. It holds if any of the following apply:

- Larger images or heavier prompts (more input tokens)
- Retries and failed-parse re-runs
- Multiple scores per image (e.g. stability / re-scoring)
- A pricier model tier
- A simple per-image rule of thumb of **$0.002–$0.01**:

| Per-image cost | 500k images |
| -------------- | ----------- |
| $0.002 | ~$1,000 |
| $0.01 | ~$5,000 |

For a defensible number, log token usage from a sample of real calls and multiply by current list price.

---

## Container RAM (~1–2 GB)

**Not profiled** (no `docker stats` or memory profiling in-repo). Deployment planning heuristic.

| Factor | Notes |
| ------ | ----- |
| Stack | `python:3.10-slim` + FastAPI, pandas, Folium, Pillow ([`Dockerfile`](../Dockerfile)) |
| Inference | Runs on **Google’s API** — model weights are not loaded in the container |
| Typical use | Web app, CSV I/O, map rendering — often **~300–800 MB** in practice |
| **1–2 GB** | Safe capacity cushion for deployment sizing |

API inference spend dominates operating cost; container CPU/RAM is secondary for this architecture.

---

## Summary

| Claim | Basis | Confidence |
| ----- | ----- | ---------- |
| ~2–3 s / 15 images | Observed batch latency; matches parallel scoring in `score.py` | Medium — spot-checked, not automated |
| Demo feels slow | 15 req/min rate limit + sleep between batches | High — enforced in code |
| Tens of hours @ 500k | 500k ÷ 15 × 2–3 s ≈ 18–28 h, no rate limit | Medium — depends on image count assumption |
| Thousands of $ | Token pricing × image count; sensitive to tokens/image | Low–medium — order of magnitude only |
| 1–2 GB RAM | Slim Python API container heuristic | Low — not measured |

---

## Tightening these estimates

1. **Latency** — Log `batch_start` → batch complete in `run_scoring_batch` over the full 50-image demo set; report p50/p95.
2. **Cost** — Read `usage_metadata` (or equivalent) from Gemini responses for a sample of images; extrapolate.
3. **RAM** — Run `docker stats` while serving the map and during a scoring run; record peak RSS.
