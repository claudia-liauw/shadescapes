import pandas as pd

from src.map_builder import marker_color, load_map_points, build_map, render_map_html


def test_marker_color_unscored():
    assert marker_color(None) == "#808080"


def test_marker_color_high_shade():
    assert marker_color(0.8) == "#2ecc71"


def test_marker_color_medium_shade():
    assert marker_color(0.5) == "#f1c40f"


def test_marker_color_low_shade():
    assert marker_color(0.2) == "#e74c3c"


def test_load_map_points_with_empty_scores_file(data_dir):
    (data_dir / "data" / "scores.csv").write_text("")
    points = load_map_points()
    assert len(points) == 2
    assert points["pedestrian_shade_score"].isna().all()


def test_load_map_points_missing_metadata(data_dir):
    (data_dir / "data" / "filtered_streetscapes.csv").unlink()
    points = load_map_points()
    assert points.empty


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
    html = render_map_html(folium_map)
    assert folium_map is not None
    assert "folium-map" in html
    assert "iframe" not in html
    assert "L.map" in html
