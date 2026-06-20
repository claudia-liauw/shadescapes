import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import IMAGES_DIR, METADATA_CSV, SCORES_CSV
from src.main import app


@pytest.fixture
def sample_metadata_rows():
    return [
        {
            "uuid": "aaa-111",
            "source": "Mapillary",
            "orig_id": 1,
            "lat": 1.3000,
            "lon": 103.8000,
            "heading": 90.0,
            "green_view_index": 0.45,
            "sky_view_index": 0.10,
            "place": "hospital",
        },
        {
            "uuid": "bbb-222",
            "source": "Mapillary",
            "orig_id": 2,
            "lat": 1.3010,
            "lon": 103.8010,
            "heading": 180.0,
            "green_view_index": 0.20,
            "sky_view_index": 0.50,
            "place": "parking_lot",
        },
    ]


@pytest.fixture
def data_dir(tmp_path, sample_metadata_rows, monkeypatch):
    exploration = tmp_path / "data" / "images" / "exploration"
    exploration.mkdir(parents=True)
    (exploration / "aaa-111.jpeg").write_bytes(b"fake-image-1")
    (exploration / "bbb-222.jpeg").write_bytes(b"fake-image-2")

    metadata_path = tmp_path / "data" / "filtered_streetscapes.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_metadata_rows).to_csv(metadata_path, index=False)

    scores_path = tmp_path / "data" / "scores.csv"
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.config.METADATA_CSV", metadata_path)
    monkeypatch.setattr("src.config.IMAGES_DIR", exploration)
    monkeypatch.setattr("src.config.SCORES_CSV", scores_path)
    monkeypatch.setattr("src.score.IMAGES_DIR", exploration)
    monkeypatch.setattr("src.score.METADATA_CSV", metadata_path)
    monkeypatch.setattr("src.score.SCORES_CSV", scores_path)
    return tmp_path


@pytest.fixture
def integration_data_dir(tmp_path, sample_metadata_rows, monkeypatch):
    """Temp data dir with a real JPEG for live Gemini integration tests."""
    exploration = tmp_path / "data" / "images" / "exploration"
    exploration.mkdir(parents=True)

    image_path = exploration / "aaa-111.jpeg"
    Image.new("RGB", (128, 128), color=(90, 140, 70)).save(image_path, format="JPEG")

    metadata_path = tmp_path / "data" / "filtered_streetscapes.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sample_metadata_rows[0]]).to_csv(metadata_path, index=False)

    scores_path = tmp_path / "data" / "scores.csv"
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.config.METADATA_CSV", metadata_path)
    monkeypatch.setattr("src.config.IMAGES_DIR", exploration)
    monkeypatch.setattr("src.config.SCORES_CSV", scores_path)
    monkeypatch.setattr("src.score.IMAGES_DIR", exploration)
    monkeypatch.setattr("src.score.METADATA_CSV", metadata_path)
    monkeypatch.setattr("src.score.SCORES_CSV", scores_path)
    return tmp_path


@pytest.fixture
def integration_client(integration_data_dir):
    return TestClient(app)
