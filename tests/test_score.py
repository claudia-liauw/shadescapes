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


def test_load_existing_scores_empty_file(data_dir):
    (data_dir / "data" / "scores.csv").write_text("")
    df = load_existing_scores()
    assert df.empty
    assert list(df.columns) == [
        "uuid",
        "pedestrian_shade_score",
        "shade_sources",
        "confidence",
        "reasoning",
        "scored_at",
    ]


@patch("src.score._call_gemini")
def test_score_image_success(mock_call, data_dir, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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


@patch("src.score._call_gemini", side_effect=RuntimeError("API down"))
def test_score_image_does_not_retry_on_api_error(mock_call, data_dir, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    image_path = data_dir / "data" / "images" / "exploration" / "aaa-111.jpeg"
    metadata = pd.read_csv(data_dir / "data" / "filtered_streetscapes.csv")
    row = metadata.loc[metadata["uuid"] == "aaa-111"].iloc[0]
    with pytest.raises(RuntimeError, match="API down"):
        score_image(image_path, row)
    assert mock_call.call_count == 1


@patch("src.score._call_gemini")
def test_run_scoring_scores_new_image(mock_call, data_dir, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
            }
        ]
    ).to_csv(scores_path, index=False)
    mock_call.return_value = json.dumps(
        {
            "pedestrian_shade_score": 0.4,
            "shade_sources": ["street_trees"],
            "confidence": "low",
            "reasoning": "Sparse cover.",
        }
    )

    summary = run_scoring(force=False)
    assert summary.scored == 1
    assert summary.skipped == 1

    scores = pd.read_csv(scores_path)
    assert len(scores) == 2
    bbb_row = scores.loc[scores["uuid"] == "bbb-222"].iloc[0]
    assert bbb_row["pedestrian_shade_score"] == 0.4


@patch("src.score.score_image")
def test_run_scoring_skips_existing(mock_score_image, data_dir, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
            }
            ,
            {
                "uuid": "bbb-222",
                "pedestrian_shade_score": 0.6,
                "shade_sources": "[]",
                "confidence": "medium",
                "reasoning": "Already scored.",
                "scored_at": "2026-06-19T13:00:00",
            }
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
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with pytest.raises(NoImagesError):
        run_scoring()


def test_run_scoring_missing_api_key(data_dir, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        run_scoring()
