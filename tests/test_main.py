from unittest.mock import patch
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import get_google_api_key
from src.main import app
from src.models import ScoreSummary


def parse_score_events(response):
    return [json.loads(line) for line in response.text.strip().split("\n") if line]


@pytest.fixture
def client(data_dir):
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
    assert "No metadata file detected" not in response.text


def test_index_with_missing_metadata_file(client, data_dir):
    (data_dir / "data" / "filtered_streetscapes.csv").unlink()
    response = client.get("/")
    assert response.status_code == 200
    assert "ShadeScapes" in response.text
    assert "No metadata file detected: data/filtered_streetscapes.csv" in response.text


def test_index_with_missing_metadata_optional_fields(client, data_dir):
    metadata_path = data_dir / "data" / "filtered_streetscapes.csv"
    minimal_metadata = pd.DataFrame(
        [
            {"uuid": "aaa-111", "lat": 1.3000, "lon": 103.8000},
        ]
    )
    minimal_metadata.to_csv(metadata_path, index=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "ShadeScapes" in response.text
    assert "folium-map" in response.text
    assert "No metadata file detected" not in response.text


def test_index_plots_all_metadata_regardless_of_images(client, data_dir):
    metadata_path = data_dir / "data" / "filtered_streetscapes.csv"
    metadata = pd.read_csv(metadata_path)
    extra_row = pd.DataFrame(
        [
            {
                "uuid": "ccc-333",
                "lat": 1.3020,
                "lon": 103.8020,
            }
        ]
    )
    pd.concat([metadata, extra_row], ignore_index=True).to_csv(metadata_path, index=False)

    images_dir = data_dir / "data" / "images"
    (images_dir / "bbb-222.jpeg").unlink()

    response = client.get("/")
    assert response.status_code == 200
    assert "folium-map" in response.text
    assert "1.302" in response.text
    assert "103.802" in response.text


def test_image_route_404(client):
    response = client.get("/images/does-not-exist.jpeg")
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found: data/images/does-not-exist.jpeg"


def test_image_route_success(client):
    response = client.get("/images/aaa-111.jpeg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"fake-image-1"


def test_image_route_rejects_path_traversal(client):
    response = client.get("/images/../../../etc/passwd.jpeg")
    assert response.status_code == 404


def test_score_endpoint_missing_metadata(client, data_dir, monkeypatch):
    (data_dir / "data" / "filtered_streetscapes.csv").unlink()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    response = client.post("/api/score")
    assert response.status_code == 200
    events = parse_score_events(response)
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 400
    assert events[-1]["detail"] == "data/filtered_streetscapes.csv not found"


def test_score_endpoint_no_images(client, data_dir, monkeypatch):
    images_dir = data_dir / "data" / "images"
    for image_path in images_dir.glob("*.jpeg"):
        image_path.unlink()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    response = client.post("/api/score")
    assert response.status_code == 200
    events = parse_score_events(response)
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 400
    assert events[-1]["detail"] == "No images found in data/images"


@patch("src.main.run_scoring")
def test_score_endpoint_success(mock_run_scoring, client):
    mock_run_scoring.return_value = ScoreSummary(
        scored=2,
        skipped=1,
        skip_reasons={"missing_metadata": 1},
        errors=[],
    )
    response = client.post("/api/score")
    assert response.status_code == 200
    events = parse_score_events(response)
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["scored"] == 2
    assert complete["skipped"] == 1
    assert complete["skip_reasons"] == {"missing_metadata": 1}
    assert complete["message"] == (
        "Scored 2 images, skipped 1 image (1 image missing metadata)."
    )


def test_score_endpoint_missing_api_key(client, monkeypatch):
    from src.models import MissingApiKeyError

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("src.main.run_scoring", side_effect=MissingApiKeyError("missing")):
        response = client.post("/api/score")
    events = parse_score_events(response)
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 503


@patch("src.score._call_gemini")
def test_score_endpoint_completes_with_api_errors(
    mock_call_gemini, client, gemini_api_fails, monkeypatch
):
    mock_call_gemini.side_effect = gemini_api_fails
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    response = client.post("/api/score?force=true")
    assert response.status_code == 200

    events = parse_score_events(response)
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["scored"] == 1
    assert complete["skipped"] == 0
    assert complete["errors"] == ["bbb-222: API down"]
    assert complete["message"].startswith(
        "Scored 1 image, skipped 0 images. 1 scoring error."
    )


@patch("src.score._call_gemini")
def test_score_endpoint_combines_skips_and_api_errors(
    mock_call_gemini,
    client,
    image_without_metadata,
    gemini_api_fails,
    monkeypatch,
):
    mock_call_gemini.side_effect = gemini_api_fails
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    response = client.post("/api/score?force=true")
    assert response.status_code == 200

    events = parse_score_events(response)
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["scored"] == 1
    assert complete["skipped"] == 1
    assert complete["skip_reasons"] == {"missing_metadata": 1}
    assert complete["skips"] == [
        "ccc-333: no metadata row in data/filtered_streetscapes.csv"
    ]
    assert complete["errors"] == ["bbb-222: API down"]
    assert complete["message"].startswith(
        "Scored 1 image, skipped 1 image (1 image missing metadata). 1 scoring error."
    )


@pytest.mark.integration
def test_score_endpoint_live(integration_client, integration_data_dir):
    if not get_google_api_key():
        pytest.skip("GOOGLE_API_KEY must be set for live tests")

    response = integration_client.post("/api/score")
    assert response.status_code == 200

    events = parse_score_events(response)
    body = events[-1]
    assert body["type"] == "complete"
    assert body["scored"] == 1
    assert body["skipped"] == 0
    assert body["errors"] == []

    scores_path = integration_data_dir / "data" / "scores.csv"
    scores = pd.read_csv(scores_path)
    assert len(scores) == 1
    assert scores.iloc[0]["uuid"] == "aaa-111"
    assert 0.0 <= scores.iloc[0]["pedestrian_shade_score"] <= 1.0
