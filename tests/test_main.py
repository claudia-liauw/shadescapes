from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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
