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
    assert summary.message == "Scored 3 images, skipped 1 image. 1 scoring error."


def test_score_summary_message_with_skip_reasons():
    summary = ScoreSummary(
        scored=1,
        skipped=3,
        skip_reasons={"already_scored": 2, "missing_metadata": 1},
        errors=[],
    )
    assert summary.message == "Scored 1 image, skipped 3 images (2 images already scored, 1 image missing metadata)."


def test_score_summary_message_with_scoring_errors():
    summary = ScoreSummary(
        scored=1,
        skipped=0,
        errors=["abc-123: API down"],
    )
    assert summary.message == "Scored 1 image, skipped 0 images. 1 scoring error."
