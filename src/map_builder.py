import json
import re
from html import escape

import folium
import pandas as pd
from pandas.errors import EmptyDataError

from src import config
from src.score import load_metadata


def marker_color(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "#808080"
    if score >= 0.7:
        return "#2ecc71"
    if score >= 0.4:
        return "#f1c40f"
    return "#e74c3c"


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
    try:
        return pd.read_csv(config.SCORES_CSV)
    except EmptyDataError:
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


def load_map_points() -> pd.DataFrame:
    metadata = load_metadata()
    if metadata.empty:
        return pd.DataFrame()
    scores = _load_scores()
    points = metadata.copy()
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


def _format_optional(value, formatter=None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        formatted = formatter(value) if formatter else value
        return escape(str(formatted))
    except (ValueError, TypeError):
        return escape(str(value))


def _popup_html(row: pd.Series) -> str:
    uuid = escape(str(row["uuid"]))
    score = row.get("pedestrian_shade_score")
    score_text = "Not scored yet" if score is None or pd.isna(score) else f"{float(score):.2f}"
    sources = escape(_format_sources(row.get("shade_sources")))
    confidence = row.get("confidence")
    confidence_text = _format_optional(confidence)
    reasoning_text = _format_optional(row.get("reasoning"))
    hour_text = _format_optional(row.get("hour"), lambda v: int(float(v)))
    sidewalk_text = _format_optional(row.get("sidewalk_pct"), lambda v: f"{float(v) * 100:.0f}%")

    return f"""
    <div style="min-width:220px">
      <img src="/images/{uuid}.jpeg" alt="Street view" style="width:100%;max-width:240px;border-radius:4px;margin-bottom:8px;" /><br/>
      <strong>Shade score:</strong> {score_text}<br/>
      <strong>Sources:</strong> {sources}<br/>
      <strong>Confidence:</strong> {confidence_text}<br/>
      <strong>Hour:</strong> {hour_text}<br/>
      <strong>Sidewalk:</strong> {sidewalk_text}<br/>
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

    return sg_map


def render_map_html(sg_map: folium.Map) -> str:
    """Return Folium map markup safe to embed in our page template."""
    html = sg_map.get_root().render()
    head_match = re.search(r"<head>(.*?)</head>", html, re.S)
    body_match = re.search(r"<body>(.*?)</body>", html, re.S)
    scripts_match = re.search(r"</body>(.*?)</html>", html, re.S)
    if not head_match or not body_match:
        return html

    head = re.sub(
        r"<style>\s*#map\s*\{[^}]*\}\s*</style>\s*",
        "",
        head_match.group(1),
        flags=re.S,
    )
    scripts = scripts_match.group(1) if scripts_match else ""
    return f"{head}{body_match.group(1)}{scripts}"
