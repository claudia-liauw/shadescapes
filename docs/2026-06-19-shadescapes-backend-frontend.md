# ShadeScapes Backend & Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a FastAPI app where clicking "Run Shade Scoring" runs Gemini VLM on `data/images/exploration/`, writes `data/scores.csv`, and renders a Folium map with shade-coloured markers and rich popups.

**Architecture:** FastAPI monolith with `score.py` (VLM batch), `map_builder.py` (Folium), and Jinja2 template embedding the map. Inner-join images to `filtered_streetscapes.csv` on uuid. Vanilla JS handles the score button via `fetch`.

**Tech Stack:** Python 3.10.14, FastAPI, uvicorn, Jinja2, Folium, pandas, google-genai, Pillow, pytest, httpx

**Spec:** `docs/superpowers/specs/2026-06-19-shadescapes-backend-frontend-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `src/config.py` | Project paths, env vars, model name |
| `src/models.py` | Pydantic schemas + custom exceptions |
| `src/map_builder.py` | Join CSVs, build Folium map |
| `src/score.py` | VLM inference, CSV I/O, batch orchestration |
| `src/main.py` | FastAPI routes, template rendering |
| `templates/index.html` | Page layout, score button, map embed |
| `static/style.css` | Layout and button states |
| `tests/conftest.py` | Shared fixtures (tmp data dir, sample CSV) |
| `tests/test_models.py` | Schema validation |
| `tests/test_map_builder.py` | Map data join + marker colours |
| `tests/test_score.py` | Scoring logic with mocked Gemini |
| `tests/test_main.py` | HTTP routes via TestClient |

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/__init__.py`

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

Add to `[project].dependencies`:

```toml
"fastapi>=0.115.0",
"uvicorn[standard]>=0.34.0",
"jinja2>=3.1.0",
"python-multipart>=0.0.20",
```

Add dev dependency group:

```toml
[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "httpx>=0.28.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync --group dev`
Expected: lockfile updated, packages installed without error

- [ ] **Step 3: Create empty package**

Create `src/__init__.py`:

```python
"""ShadeScapes — pedestrian shade index from street imagery."""
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/__init__.py
git commit -m "chore: add FastAPI stack and test dependencies"
```

---

### Task 2: Configuration module

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from src.config import (
    IMAGES_DIR,
    METADATA_CSV,
    PROJECT_ROOT,
    SCORES_CSV,
    get_gemini_api_key,
)


def test_paths_are_under_project_root():
    assert PROJECT_ROOT.is_dir()
    assert METADATA_CSV == PROJECT_ROOT / "data" / "filtered_streetscapes.csv"
    assert IMAGES_DIR == PROJECT_ROOT / "data" / "images" / "exploration"
    assert SCORES_CSV == PROJECT_ROOT / "data" / "scores.csv"


def test_get_gemini_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_gemini_api_key() is None


def test_get_gemini_api_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert get_gemini_api_key() == "test-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement `src/config.py`**

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "filtered_streetscapes.csv"
IMAGES_DIR = DATA_DIR / "images" / "exploration"
SCORES_CSV = DATA_DIR / "scores.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add project config and path constants"
```

---

### Task 3: Pydantic models and exceptions

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from src.models import ShadeScore, ScoreSummary


def test_shade_score_valid():
    score = ShadeScore(
        pedestrian_shade_score=0.72,
        shade_sources=["street_trees", "building_overhang"],
        confidence="high",
        reasoning="Dense canopy over sidewalk.",
    )
    assert score.pedestrian_shade_score == 0.72
    assert score.confidence == "high"


def test_shade_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ShadeScore(
            pedestrian_shade_score=1.5,
            shade_sources=[],
            confidence="low",
            reasoning="Too bright.",
        )


def test_shade_score_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        ShadeScore(
            pedestrian_shade_score=0.5,
            shade_sources=[],
            confidence="certain",
            reasoning="Invalid confidence level.",
        )


def test_score_summary_defaults():
    summary = ScoreSummary(scored=3, skipped=1, errors=["abc: parse error"])
    assert summary.scored == 3
    assert summary.skipped == 1
    assert len(summary.errors) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Implement `src/models.py`**

```python
from typing import Literal

from pydantic import BaseModel, Field


class ShadeScore(BaseModel):
    pedestrian_shade_score: float = Field(ge=0.0, le=1.0)
    shade_sources: list[str]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(min_length=1, max_length=500)


class ScoreSummary(BaseModel):
    scored: int
    skipped: int
    errors: list[str]


class NoImagesError(Exception):
    """Raised when the exploration image directory has no JPEG files."""


class MissingApiKeyError(Exception):
    """Raised when GEMINI_API_KEY is not set."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add ShadeScore schema and scoring exceptions"
```

---

### Task 4: Map builder

**Files:**
- Create: `src/map_builder.py`
- Create: `tests/conftest.py`
- Create: `tests/test_map_builder.py`

- [ ] **Step 1: Write shared test fixtures**

Create `tests/conftest.py`:

```python
import pandas as pd
import pytest

from src.config import IMAGES_DIR, METADATA_CSV, SCORES_CSV


@pytest.fixture
def sample_metadata_rows():
    return [
        {
            "uuid": "aaa-111",
            "source": "Mapillary",
            "orig_id": 1,
            "lat": 1.3000,
            "lon": 103.8000,
            "heading": 90.0,
            "green_view_index": 0.45,
            "sky_view_index": 0.10,
            "place": "hospital",
        },
        {
            "uuid": "bbb-222",
            "source": "Mapillary",
            "orig_id": 2,
            "lat": 1.3010,
            "lon": 103.8010,
            "heading": 180.0,
            "green_view_index": 0.20,
            "sky_view_index": 0.50,
            "place": "parking_lot",
        },
    ]


@pytest.fixture
def data_dir(tmp_path, sample_metadata_rows, monkeypatch):
    exploration = tmp_path / "data" / "images" / "exploration"
    exploration.mkdir(parents=True)
    (exploration / "aaa-111.jpeg").write_bytes(b"fake-image-1")
    (exploration / "bbb-222.jpeg").write_bytes(b"fake-image-2")

    metadata_path = tmp_path / "data" / "filtered_streetscapes.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_metadata_rows).to_csv(metadata_path, index=False)

    scores_path = tmp_path / "data" / "scores.csv"
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.config.METADATA_CSV", metadata_path)
    monkeypatch.setattr("src.config.IMAGES_DIR", exploration)
    monkeypatch.setattr("src.config.SCORES_CSV", scores_path)
    return tmp_path
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_map_builder.py`:

```python
import pandas as pd

from src.map_builder import marker_color, load_map_points, build_map


def test_marker_color_unscored():
    assert marker_color(None) == "#808080"


def test_marker_color_high_shade():
    assert marker_color(0.8) == "#2ecc71"


def test_marker_color_medium_shade():
    assert marker_color(0.5) == "#f1c40f"


def test_marker_color_low_shade():
    assert marker_color(0.2) == "#e74c3c"


def test_load_map_points_joins_images(data_dir):
    points = load_map_points()
    assert len(points) == 2
    assert set(points["uuid"]) == {"aaa-111", "bbb-222"}
    assert points.loc[0, "pedestrian_shade_score"] is None or pd.isna(
        points.loc[0, "pedestrian_shade_score"]
    )


def test_load_map_points_with_scores(data_dir):
    scores_path = data_dir / "data" / "scores.csv"
    pd.DataFrame(
        [
            {
                "uuid": "aaa-111",
                "pedestrian_shade_score": 0.8,
                "shade_sources": '["street_trees"]',
                "confidence": "high",
                "reasoning": "Shaded sidewalk.",
                "scored_at": "2026-06-19T12:00:00",
            }
        ]
    ).to_csv(scores_path, index=False)

    points = load_map_points()
    row = points.loc[points["uuid"] == "aaa-111"].iloc[0]
    assert row["pedestrian_shade_score"] == 0.8
    assert row["confidence"] == "high"


def test_build_map_returns_folium_map(data_dir):
    folium_map = build_map()
    assert folium_map is not None
    html = folium_map.get_root().render()
    assert "aaa-111" in html or "1.3" in html
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_map_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.map_builder'`

- [ ] **Step 4: Implement `src/map_builder.py`**

```python
import json
from html import escape
from pathlib import Path

import folium
import pandas as pd

from src import config


def marker_color(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "#808080"
    if score >= 0.7:
        return "#2ecc71"
    if score >= 0.4:
        return "#f1c40f"
    return "#e74c3c"


def _discover_image_uuids(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.stem for path in directory.glob("*.jpeg")}


def _load_scores() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "uuid",
                "pedestrian_shade_score",
                "shade_sources",
                "confidence",
                "reasoning",
                "scored_at",
            ]
        )
    return pd.read_csv(SCORES_CSV)


def load_map_points() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_CSV)
    scores = _load_scores()
    image_uuids = _discover_image_uuids(IMAGES_DIR)

    points = metadata[metadata["uuid"].isin(image_uuids)].copy()
    if scores.empty:
        points["pedestrian_shade_score"] = None
        points["shade_sources"] = None
        points["confidence"] = None
        points["reasoning"] = None
        points["scored_at"] = None
        return points.reset_index(drop=True)

    merged = points.merge(scores, on="uuid", how="left")
    return merged.reset_index(drop=True)


def _format_sources(raw: str | float | None) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "—"
    text = str(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return ", ".join(parsed) if parsed else "—"
    except json.JSONDecodeError:
        pass
    return text


def _popup_html(row: pd.Series) -> str:
    uuid = escape(str(row["uuid"]))
    score = row.get("pedestrian_shade_score")
    score_text = "Not scored yet" if score is None or pd.isna(score) else f"{float(score):.2f}"
    sources = escape(_format_sources(row.get("shade_sources")))
    confidence = row.get("confidence")
    confidence_text = (
        "—" if confidence is None or (isinstance(confidence, float) and pd.isna(confidence)) else escape(str(confidence))
    )
    reasoning = row.get("reasoning")
    reasoning_text = (
        "—" if reasoning is None or (isinstance(reasoning, float) and pd.isna(reasoning)) else escape(str(reasoning))
    )
    place = escape(str(row.get("place", "—")))
    gvi = row.get("green_view_index")
    svi = row.get("sky_view_index")
    gvi_text = "—" if gvi is None or pd.isna(gvi) else f"{float(gvi):.2f}"
    svi_text = "—" if svi is None or pd.isna(svi) else f"{float(svi):.2f}"

    return f"""
    <div style="min-width:220px">
      <img src="/images/{uuid}.jpeg" alt="Street view" style="width:100%;max-width:240px;border-radius:4px;margin-bottom:8px;" />
      <strong>Shade score:</strong> {score_text}<br/>
      <strong>Sources:</strong> {sources}<br/>
      <strong>Confidence:</strong> {confidence_text}<br/>
      <strong>Place:</strong> {place}<br/>
      <strong>GVI:</strong> {gvi_text} &nbsp; <strong>SVI:</strong> {svi_text}<br/>
      <p style="margin:8px 0 0">{reasoning_text}</p>
    </div>
    """


def build_map() -> folium.Map:
    points = load_map_points()
    if points.empty:
        center = [1.3521, 103.8198]
        zoom = 12
    else:
        center = [points["lat"].mean(), points["lon"].mean()]
        zoom = 16

    sg_map = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    for _, row in points.iterrows():
        score = row.get("pedestrian_shade_score")
        color = marker_color(None if pd.isna(score) else score)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(_popup_html(row), max_width=320),
            tooltip=f"Shade: {score:.2f}" if score is not None and not pd.isna(score) else "Not scored",
        ).add_to(sg_map)

    legend_html = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.2);font-size:13px;">
      <strong>Shade index</strong><br/>
      <span style="color:#2ecc71">&#9679;</span> High (&ge; 0.7)<br/>
      <span style="color:#f1c40f">&#9679;</span> Medium (0.4–0.7)<br/>
      <span style="color:#e74c3c">&#9679;</span> Low (&lt; 0.4)<br/>
      <span style="color:#808080">&#9679;</span> Not scored
    </div>
    """
    sg_map.get_root().html.add_child(folium.Element(legend_html))
    return sg_map
```

> **Note:** Importing the `config` module (rather than individual path constants) keeps the implementation aligned with the fixtures’ `monkeypatch.setattr("src.config.*")` usage so the temporary directories configured in `tests/conftest.py` can override the paths at runtime instead of stale values being cached at import time.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_map_builder.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/map_builder.py tests/conftest.py tests/test_map_builder.py
git commit -m "feat: add Folium map builder with shade-coloured markers"
```

---

### Task 5: VLM scoring module

**Files:**
- Create: `src/score.py`
- Create: `tests/test_score.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_score.py`:

```python
import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.models import MissingApiKeyError, NoImagesError, ShadeScore
from src.score import (
    build_prompt,
    discover_images,
    load_existing_scores,
    parse_vlm_response,
    run_scoring,
    score_image,
)


def test_discover_images(data_dir):
    images = discover_images()
    assert sorted(images) == ["aaa-111.jpeg", "bbb-222.jpeg"]


def test_discover_images_empty(tmp_path, monkeypatch):
    exploration = tmp_path / "data" / "images" / "exploration"
    exploration.mkdir(parents=True)
    monkeypatch.setattr("src.score.IMAGES_DIR", exploration)
    assert discover_images() == []


def test_build_prompt_includes_heading():
    prompt = build_prompt(heading=270.0)
    assert "270" in prompt
    assert "pedestrian_shade_score" in prompt


def test_parse_vlm_response_valid_json():
    raw = json.dumps(
        {
            "pedestrian_shade_score": 0.6,
            "shade_sources": ["street_trees"],
            "confidence": "medium",
            "reasoning": "Partial tree cover.",
        }
    )
    score = parse_vlm_response(raw)
    assert isinstance(score, ShadeScore)
    assert score.pedestrian_shade_score == 0.6


def test_parse_vlm_response_strips_markdown_fence():
    raw = """```json
{"pedestrian_shade_score":0.3,"shade_sources":[],"confidence":"low","reasoning":"Open road."}
```"""
    score = parse_vlm_response(raw)
    assert score.pedestrian_shade_score == 0.3


def test_load_existing_scores_empty(data_dir):
    df = load_existing_scores()
    assert df.empty


@patch("src.score._call_gemini")
def test_score_image_success(mock_call, data_dir, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_call.return_value = json.dumps(
        {
            "pedestrian_shade_score": 0.75,
            "shade_sources": ["building_overhang"],
            "confidence": "high",
            "reasoning": "Building shadow on sidewalk.",
        }
    )
    image_path = data_dir / "data" / "images" / "exploration" / "aaa-111.jpeg"
    metadata = pd.read_csv(data_dir / "data" / "filtered_streetscapes.csv")
    row = metadata.loc[metadata["uuid"] == "aaa-111"].iloc[0]
    result = score_image(image_path, row)
    assert result.pedestrian_shade_score == 0.75


@patch("src.score.score_image")
def test_run_scoring_skips_existing(mock_score_image, data_dir, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    scores_path = data_dir / "data" / "scores.csv"
    pd.DataFrame(
        [
            {
                "uuid": "aaa-111",
                "pedestrian_shade_score": 0.5,
                "shade_sources": "[]",
                "confidence": "medium",
                "reasoning": "Already scored.",
                "scored_at": "2026-06-19T12:00:00",
            },
            {
                "uuid": "bbb-222",
                "pedestrian_shade_score": 0.6,
                "shade_sources": "[]",
                "confidence": "medium",
                "reasoning": "Already scored.",
                "scored_at": "2026-06-19T13:00:00",
            },
        ]
    ).to_csv(scores_path, index=False)

    summary = run_scoring(force=False)
    assert summary.scored == 0
    assert summary.skipped == 2
    mock_score_image.assert_not_called()


def test_run_scoring_no_images(tmp_path, monkeypatch):
    exploration = tmp_path / "data" / "images" / "exploration"
    exploration.mkdir(parents=True)
    monkeypatch.setattr("src.score.IMAGES_DIR", exploration)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(NoImagesError):
        run_scoring()


def test_run_scoring_missing_api_key(data_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        run_scoring()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.score'`

- [ ] **Step 3: Implement `src/score.py`**

```python
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google import genai
from PIL import Image

from src import config
from src.models import MissingApiKeyError, NoImagesError, ScoreSummary, ShadeScore

IMAGES_DIR = config.IMAGES_DIR
METADATA_CSV = config.METADATA_CSV
SCORES_CSV = config.SCORES_CSV


def discover_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    return [path.name for path in sorted(IMAGES_DIR.glob("*.jpeg"))]


def build_prompt(heading: float | None = None) -> str:
    heading_line = (
        f"The camera heading is {heading:.0f} degrees. Assume tropical afternoon sun from the west/southwest relative to this view.\n"
        if heading is not None and not pd.isna(heading)
        else "Assume tropical afternoon sun from the west/southwest.\n"
    )
    return (
        "You are assessing pedestrian shade on the sidewalk in this street-level photo.\n"
        "Context: Singapore, 2–4pm, hot afternoon. Score how shaded the walkable pedestrian path is, not overall scene aesthetics.\n"
        f"{heading_line}"
        "Respond with JSON only, no markdown, using exactly this schema:\n"
        "{\n"
        '  "pedestrian_shade_score": 0.72,\n'
        '  "shade_sources": ["street_trees", "building_overhang"],\n'
        '  "confidence": "high",\n'
        '  "reasoning": "One or two sentences."\n'
        "}\n"
        "pedestrian_shade_score must be a float from 0.0 (fully exposed) to 1.0 (fully shaded).\n"
        'confidence must be one of: "low", "medium", "high".'
    )


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_vlm_response(raw: str) -> ShadeScore:
    payload = json.loads(_strip_markdown_fence(raw))
    return ShadeScore.model_validate(payload)


def _call_gemini(image_path: Path, prompt: str) -> str:
    client = genai.Client()
    image = Image.open(image_path)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[image, prompt],
    )
    return response.text or ""


def score_image(image_path: Path, metadata_row: pd.Series, retry: bool = True) -> ShadeScore:
    prompt = build_prompt(metadata_row.get("heading"))
    try:
        raw = _call_gemini(image_path, prompt)
        return parse_vlm_response(raw)
    except Exception:
        if not retry:
            raise
        retry_prompt = prompt + "\nReturn valid JSON only. No prose outside the JSON object."
        raw = _call_gemini(image_path, retry_prompt)
        return parse_vlm_response(raw)


def load_existing_scores() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "uuid",
                "pedestrian_shade_score",
                "shade_sources",
                "confidence",
                "reasoning",
                "scored_at",
            ]
        )
    return pd.read_csv(SCORES_CSV)


def _write_scores(df: pd.DataFrame) -> None:
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)


def run_scoring(force: bool = False) -> ScoreSummary:
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise MissingApiKeyError("GEMINI_API_KEY is not configured")

    images = discover_images()
    if not images:
        raise NoImagesError("No images found in data/images/exploration")

    metadata = pd.read_csv(METADATA_CSV)
    metadata_by_uuid = metadata.set_index("uuid")
    existing = load_existing_scores()
    existing_uuids = set(existing["uuid"].astype(str)) if not existing.empty else set()

    scored_count = 0
    skipped_count = 0
    errors: list[str] = []
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    rows_by_uuid = {str(row["uuid"]): row for row in rows}

    image_paths = [IMAGES_DIR / name for name in images]

    for image_path in image_paths:
        uuid = image_path.stem
        if uuid not in metadata_by_uuid.index:
            skipped_count += 1
            errors.append(f"{uuid}: no metadata row in filtered_streetscapes.csv")
            continue

        if not force and uuid in existing_uuids:
            skipped_count += 1
            continue

        try:
            result = score_image(image_path, metadata_by_uuid.loc[uuid])
            rows_by_uuid[uuid] = {
                "uuid": uuid,
                "pedestrian_shade_score": result.pedestrian_shade_score,
                "shade_sources": json.dumps(result.shade_sources),
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            scored_count += 1
        except Exception as exc:
            errors.append(f"{uuid}: {exc}")

    _write_scores(pd.DataFrame(rows_by_uuid.values()))
    return ScoreSummary(scored=scored_count, skipped=skipped_count, errors=errors)
```

> **Note:** The score module now exposes `IMAGES_DIR`, `METADATA_CSV`, and `SCORES_CSV` as symbols backed by `src.config`, and `discover_images()` returns bare filenames so fixtures can monkeypatch these attributes. This is why the tests manually patch `src.score.*` and the warning about stale constants from Task 4 applies here as well.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_score.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/score.py tests/test_score.py
git commit -m "feat: add Gemini VLM batch scoring with CSV persistence"
```

---

### Task 6: FastAPI application

**Files:**
- Create: `src/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:

```python
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models import ScoreSummary


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_renders_map(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ShadeScapes" in response.text
    assert "Run Shade Scoring" in response.text


def test_image_route_404(client):
    response = client.get("/images/does-not-exist.jpeg")
    assert response.status_code == 404


@patch("src.main.run_scoring")
def test_score_endpoint_success(mock_run_scoring, client):
    mock_run_scoring.return_value = ScoreSummary(scored=2, skipped=1, errors=[])
    response = client.post("/api/score")
    assert response.status_code == 200
    assert response.json()["scored"] == 2


@patch("src.main.run_scoring", side_effect=Exception("NoImagesError"))
def test_score_endpoint_handles_no_images(mock_run_scoring, client):
    from src.models import NoImagesError

    mock_run_scoring.side_effect = NoImagesError("No images found")
    response = client.post("/api/score")
    assert response.status_code == 400


def test_score_endpoint_missing_api_key(client, monkeypatch):
    from src.models import MissingApiKeyError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("src.main.run_scoring", side_effect=MissingApiKeyError("missing")):
        response = client.post("/api/score")
    assert response.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.main'`

- [ ] **Step 3: Implement `src/main.py`**

```python
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import IMAGES_DIR, PROJECT_ROOT
from src.map_builder import build_map
from src.models import MissingApiKeyError, NoImagesError
from src.score import run_scoring

app = FastAPI(title="ShadeScapes")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    folium_map = build_map()
    map_html = folium_map.get_root().render()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"map_html": map_html},
    )


@app.post("/api/score")
def score_images(force: bool = Query(default=False)):
    try:
        summary = run_scoring(force=force)
        return summary.model_dump()
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NoImagesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/images/{filename}")
def get_image(filename: str):
    if not filename.endswith(".jpeg"):
        raise HTTPException(status_code=404, detail="Image not found")
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: 5 passed (adjust if import paths differ)

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add FastAPI routes for map, scoring, and images"
```

---

### Task 7: Frontend template and styles

**Files:**
- Create: `templates/index.html`
- Create: `static/style.css`

- [ ] **Step 1: Create `templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ShadeScapes</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header class="header">
    <div>
      <h1>ShadeScapes</h1>
      <p class="subtitle">Pedestrian shade index for a Singapore street corridor</p>
    </div>
    <div class="actions">
      <button id="score-btn" type="button">Run Shade Scoring</button>
      <p id="status" class="status" aria-live="polite"></p>
    </div>
  </header>

  <main>
    <div id="map" class="map">{{ map_html|safe }}</div>
  </main>

  <script>
    const button = document.getElementById("score-btn");
    const status = document.getElementById("status");

    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "Scoring…";
      status.textContent = "";
      status.className = "status";

      try {
        const response = await fetch("/api/score", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Scoring failed");
        }
        status.textContent = `Scored ${data.scored}, skipped ${data.skipped}. Reloading map…`;
        status.className = "status status-success";
        window.location.reload();
      } catch (error) {
        status.textContent = error.message;
        status.className = "status status-error";
        button.disabled = false;
        button.textContent = "Run Shade Scoring";
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: Create `static/style.css`**

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1f2933;
  background: #f7f9fb;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 16px 20px;
  background: #ffffff;
  border-bottom: 1px solid #d9e2ec;
}

h1 {
  margin: 0;
  font-size: 1.5rem;
}

.subtitle {
  margin: 4px 0 0;
  color: #52606d;
  font-size: 0.95rem;
}

.actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

#score-btn {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  background: #147d64;
  color: #ffffff;
  font-size: 0.95rem;
  cursor: pointer;
}

#score-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.status {
  margin: 0;
  font-size: 0.85rem;
}

.status-success {
  color: #147d64;
}

.status-error {
  color: #ba2525;
}

.map {
  min-height: 80vh;
}

@media (max-width: 720px) {
  .header {
    flex-direction: column;
  }

  .actions {
    align-items: flex-start;
    width: 100%;
  }
}
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add templates/index.html static/style.css
git commit -m "feat: add map page template and scoring button UI"
```

---

### Task 8: Manual integration verification

**Files:** none (verification only)

- [ ] **Step 1: Start the server**

Run: `uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`
Expected: `Uvicorn running on http://0.0.0.0:8000`

- [ ] **Step 2: Verify map without scores**

Open: `http://localhost:8000`
Expected: 9 grey markers on map centred on corridor (~1.30°N, 103.80°E)

- [ ] **Step 3: Verify scoring without API key**

Run in another terminal:
```bash
env -u GEMINI_API_KEY curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/api/score
```
Expected: `503`

- [ ] **Step 4: Run live scoring**

Ensure `.env` or shell has `GEMINI_API_KEY` set. Click **Run Shade Scoring** in browser.
Expected: `data/scores.csv` created; page reloads; markers turn green/yellow/red

- [ ] **Step 5: Verify popup content**

Click any coloured marker.
Expected: thumbnail loads, shade score, sources, confidence, reasoning, GVI/SVI visible

- [ ] **Step 6: Verify skip behaviour**

Click **Run Shade Scoring** again without force.
Expected: status shows skipped count ≈ 9, scored ≈ 0

- [ ] **Step 7: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address integration issues from manual verification"
```

Only run this step if fixes were required in Step 1–6.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Discover `*.jpeg` in exploration | Task 5 |
| Join on uuid | Task 4 |
| VLM JSON schema + retry | Task 5 |
| Guards: no images, no API key | Task 5, 6 |
| `scores.csv` columns | Task 5 |
| Skip already-scored unless force | Task 5, 6 (`?force=true`) |
| Folium map with colour scale + legend | Task 4 |
| Popup with thumbnail + metadata | Task 4, 6 |
| GET /, POST /api/score, GET /images, GET /health | Task 6 |
| Frontend button + reload | Task 7 |
| Manual test scenarios | Task 8 |

---

## Notes for implementer

- `data/` is gitignored; local images must exist at `data/images/exploration/`.
- If `gemini-2.0-flash` is unavailable, set `GEMINI_MODEL` env var to a working model id.
- Folium legend uses `position:fixed`; acceptable for demo.
- Do not commit `.env` or `data/scores.csv` unless explicitly requested.
