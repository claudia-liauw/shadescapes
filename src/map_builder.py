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
    if not config.SCORES_CSV.exists():
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
    return pd.read_csv(config.SCORES_CSV)


def load_map_points() -> pd.DataFrame:
    metadata = pd.read_csv(config.METADATA_CSV)
    scores = _load_scores()
    image_uuids = _discover_image_uuids(config.IMAGES_DIR)

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
