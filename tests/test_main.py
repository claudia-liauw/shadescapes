from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import get_google_api_key
from src.main import app
from src.models import ScoreSummary


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


def test_index_with_missing_metadata(client, data_dir):
    (data_dir / "data" / "filtered_streetscapes.csv").unlink()
    response = client.get("/")
    assert response.status_code == 200
    assert "ShadeScapes" in response.text


def test_image_route_404(client):
    response = client.get("/images/does-not-exist.jpeg")
    assert response.status_code == 404


def test_image_route_success(client):
    response = client.get("/images/aaa-111.jpeg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"fake-image-1"


def test_image_route_rejects_path_traversal(client):
    response = client.get("/images/../../../etc/passwd.jpeg")
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

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("src.main.run_scoring", side_effect=MissingApiKeyError("missing")):
        response = client.post("/api/score")
    assert response.status_code == 503


@pytest.mark.integration
def test_score_endpoint_live(integration_client, integration_data_dir):
    if not get_google_api_key():
        pytest.skip("GOOGLE_API_KEY must be set for live tests")

    response = integration_client.post("/api/score")
    assert response.status_code == 200

    body = response.json()
    assert body["scored"] == 1
    assert body["skipped"] == 0
    assert body["errors"] == []

    scores_path = integration_data_dir / "data" / "scores.csv"
    scores = pd.read_csv(scores_path)
    assert len(scores) == 1
    assert scores.iloc[0]["uuid"] == "aaa-111"
    assert 0.0 <= scores.iloc[0]["pedestrian_shade_score"] <= 1.0
