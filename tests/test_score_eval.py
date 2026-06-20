import pandas as pd
from unittest.mock import patch

from eval.score_eval import score_eval_uuids
from src.score import SCORE_COLUMNS, load_existing_scores
from src.models import ShadeScore


def test_load_existing_scores_for_eval_path(tmp_path):
    scores_csv = tmp_path / "scores.csv"
    df = load_existing_scores(scores_csv)
    assert list(df.columns) == [
        "uuid",
        "pedestrian_shade_score",
        "shade_sources",
        "confidence",
        "reasoning",
        "scored_at",
    ]
    assert df.empty


@patch("src.score.score_image")
def test_score_eval_uuids_writes_eval_scores(mock_score_image, tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    data_dir = tmp_path / "data"
    sample_dir = data_dir / "images" / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "u1.jpeg").write_bytes(b"fake")

    labels_csv = eval_dir / "human_labels.csv"
    scores_csv = eval_dir / "scores.csv"
    meta_csv = data_dir / "filtered_streetscapes.csv"
    eval_dir.mkdir()
    pd.DataFrame({"uuid": ["u1"], "shade_1to5": [3], "scene_category": ["tree_canopy"], "notes": [""]}).to_csv(labels_csv, index=False)
    pd.DataFrame(
        {
            "uuid": ["u1"],
            "source": ["Mapillary"],
            "orig_id": [""],
            "lat": [1.3],
            "lon": [103.8],
            "hour": [14],
            "heading": [90.0],
            "place": ["campus"],
            "sidewalk_pct": [0.02],
        }
    ).to_csv(meta_csv, index=False)

    monkeypatch.setattr("eval.score_eval.HUMAN_LABELS_CSV", labels_csv)
    monkeypatch.setattr("eval.score_eval.EVAL_SCORES_CSV", scores_csv)
    monkeypatch.setattr("eval.score_eval.FILTERED_METADATA_CSV", meta_csv)
    monkeypatch.setattr("eval.score_eval.SYNTHETIC_METADATA_CSV", data_dir / "synthetic_streetscapes.csv")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", data_dir / "images" / "synthetic")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    mock_score_image.return_value = ShadeScore(
        pedestrian_shade_score=0.6,
        shade_sources=["street_trees"],
        confidence="high",
        reasoning="test",
    )

    summary = score_eval_uuids(force=True)
    assert summary.scored == 1
    out = pd.read_csv(scores_csv)
    assert len(out) == 1
    assert out.loc[0, "uuid"] == "u1"
    assert out.loc[0, "pedestrian_shade_score"] == 0.6
