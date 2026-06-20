import pandas as pd
import pytest

from eval.metrics import (
    HIGH_VARIANCE_RANGE,
    MISMATCH_THRESHOLD,
    build_eval_frame,
    compute_method_metrics,
    filter_real_images,
    flag_mismatches,
    load_eval_metadata,
    merge_run_stability,
    normalize_human,
    per_image_run_stats,
    resolve_image_path,
    stability_summary,
    summarize_mismatches_by_scene_category,
)
from eval.paths import SAMPLE_IMAGES_DIR, SYNTHETIC_IMAGES_DIR


def test_normalize_human():
    s = pd.Series([1, 3, 5])
    result = normalize_human(s)
    pd.testing.assert_series_equal(result, pd.Series([0.0, 0.5, 1.0]), check_names=False)


def test_load_eval_metadata_real_only(tmp_path, monkeypatch):
    real_csv = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "uuid": ["a"],
            "source": ["Mapillary"],
            "lat": [1.3],
            "lon": [103.8],
            "hour": [14],
            "heading": [90.0],
            "place": ["campus"],
            "sidewalk_pct": [0.02],
        }
    ).to_csv(real_csv, index=False)
    monkeypatch.setattr("eval.metrics.FILTERED_METADATA_CSV", real_csv)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_METADATA_CSV", tmp_path / "missing.csv")

    meta = load_eval_metadata()
    assert len(meta) == 1
    assert meta.loc[0, "source"] == "mapillary"


def test_load_eval_metadata_concat_real_and_synthetic(tmp_path, monkeypatch):
    real_csv = tmp_path / "filtered.csv"
    syn_csv = tmp_path / "synthetic.csv"
    pd.DataFrame({"uuid": ["a"], "source": ["Mapillary"], "hour": [14], "heading": [90.0], "sidewalk_pct": [0.02], "lat": [1.3], "lon": [103.8], "place": ["campus"]}).to_csv(real_csv, index=False)
    pd.DataFrame({"uuid": ["syn-1"], "source": ["synthetic"], "hour": [14], "heading": [180.0], "sidewalk_pct": [0.35], "lat": [1.31], "lon": [103.81], "place": ["synthetic_linkway"]}).to_csv(syn_csv, index=False)
    monkeypatch.setattr("eval.metrics.FILTERED_METADATA_CSV", real_csv)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_METADATA_CSV", syn_csv)

    meta = load_eval_metadata()
    assert set(meta["uuid"]) == {"a", "syn-1"}
    assert set(meta["source"]) == {"mapillary", "synthetic"}


def test_resolve_image_path_prefers_sample(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    synthetic_dir = tmp_path / "synthetic"
    sample_dir.mkdir()
    synthetic_dir.mkdir()
    (sample_dir / "u1.jpeg").write_bytes(b"x")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", synthetic_dir)

    assert resolve_image_path("u1") == sample_dir / "u1.jpeg"


def test_resolve_image_path_falls_back_to_synthetic(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    synthetic_dir = tmp_path / "synthetic"
    sample_dir.mkdir()
    synthetic_dir.mkdir()
    (synthetic_dir / "syn-1.jpeg").write_bytes(b"x")
    monkeypatch.setattr("eval.metrics.SAMPLE_IMAGES_DIR", sample_dir)
    monkeypatch.setattr("eval.metrics.SYNTHETIC_IMAGES_DIR", synthetic_dir)

    assert resolve_image_path("syn-1") == synthetic_dir / "syn-1.jpeg"


def test_build_eval_frame_inner_join_and_warnings():
    labels = pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "shade_1to5": [3, 4, 5],
            "scene_category": ["tree_canopy", "open_exposure", "building_shadow"],
            "notes": ["", "", ""],
        }
    )
    scores = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "pedestrian_shade_score": [0.5, 0.8],
            "shade_sources": ['["street_trees"]', '["street_trees"]'],
            "confidence": ["high", "high"],
            "reasoning": ["r1", "r2"],
        }
    )
    metadata = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "source": ["mapillary", "mapillary"],
            "hour": [14, 15],
            "sidewalk_pct": [0.02, 0.05],
            "heading": [90.0, 180.0],
            "lat": [1.3, 1.31],
            "lon": [103.8, 103.81],
            "place": ["campus", "hospital"],
        }
    )
    frame, warnings = build_eval_frame(labels, scores, metadata)
    assert len(frame) == 2
    assert "human_norm" in frame.columns
    assert "err_vlm" in frame.columns
    assert frame.loc[frame["uuid"] == "a", "scene_category"].iloc[0] == "tree_canopy"
    assert any("c" in w for w in warnings)


def test_filter_real_images():
    frame = pd.DataFrame(
        {
            "uuid": ["a", "syn-1"],
            "source": ["mapillary", "synthetic"],
            "human_norm": [0.5, 0.5],
            "pedestrian_shade_score": [0.5, 0.9],
        }
    )
    real = filter_real_images(frame)
    assert list(real["uuid"]) == ["a"]


def test_compute_method_metrics():
    frame = pd.DataFrame(
        {
            "human_norm": [0.0, 0.5, 1.0],
            "pedestrian_shade_score": [0.1, 0.5, 0.9],
        }
    )
    metrics = compute_method_metrics(frame, "pedestrian_shade_score")
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(0.06666666666666667)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_flag_mismatches_vlm_only():
    frame = pd.DataFrame(
        {
            "uuid": ["x", "y", "z"],
            "human_norm": [0.5, 0.5, 0.5],
            "pedestrian_shade_score": [0.5, 0.9, 0.2],
            "scene_category": ["tree_canopy", "open_exposure", "building_shadow"],
        }
    )
    frame["err_vlm"] = (frame["pedestrian_shade_score"] - frame["human_norm"]).abs()
    mismatches = flag_mismatches(frame, threshold=0.25)
    assert set(mismatches["uuid"]) == {"y", "z"}


def test_summarize_mismatches_by_scene_category():
    mismatches = pd.DataFrame(
        {
            "scene_category": ["tree_canopy", "tree_canopy", "open_exposure"],
        }
    )
    summary = summarize_mismatches_by_scene_category(mismatches)
    assert summary.loc[summary["scene_category"] == "tree_canopy", "count"].iloc[0] == 2
    assert summary.loc[summary["scene_category"] == "open_exposure", "count"].iloc[0] == 1


def test_per_image_run_stats():
    runs = pd.DataFrame(
        {
            "uuid": ["a", "a", "a", "b", "b", "b"],
            "run_id": [1, 2, 3, 1, 2, 3],
            "pedestrian_shade_score": [0.5, 0.6, 0.55, 0.2, 0.8, 0.3],
        }
    )
    stats = per_image_run_stats(runs)
    assert len(stats) == 2
    row_a = stats.loc[stats["uuid"] == "a"].iloc[0]
    assert row_a["score_std"] == pytest.approx(0.05, abs=0.01)
    assert row_a["score_range"] == pytest.approx(0.1, abs=0.01)


def test_stability_summary():
    stats = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "score_std": [0.05, 0.12],
            "score_range": [0.10, 0.20],
        }
    )
    runs = pd.DataFrame(
        {
            "uuid": ["a", "a", "b", "b"],
            "run_id": [1, 2, 1, 2],
            "pedestrian_shade_score": [0.5, 0.6, 0.2, 0.3],
        }
    )
    summary = stability_summary(stats, runs)
    assert summary["n_images"] == 2
    assert summary["median_std"] == pytest.approx(0.085, abs=0.01)
    assert summary["pct_high_variance"] == pytest.approx(50.0)
    assert summary["test_retest_spearman"] == pytest.approx(1.0)


def test_merge_run_stability():
    frame = pd.DataFrame({"uuid": ["a", "b"], "scene_category": ["tree_canopy", "open_exposure"]})
    stats = pd.DataFrame(
        {
            "uuid": ["a"],
            "score_std": [0.05],
            "score_range": [0.10],
        }
    )
    merged = merge_run_stability(frame, stats)
    assert merged.loc[merged["uuid"] == "a", "run_std"].iloc[0] == pytest.approx(0.05)
    assert pd.isna(merged.loc[merged["uuid"] == "b", "run_std"].iloc[0])
