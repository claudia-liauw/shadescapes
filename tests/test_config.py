from pathlib import Path

from src.config import (
    IMAGES_DIR,
    METADATA_CSV,
    PROJECT_ROOT,
    SCORES_CSV,
    get_google_api_key,
)


def test_paths_are_under_project_root():
    assert PROJECT_ROOT.is_dir()
    assert METADATA_CSV == PROJECT_ROOT / "data" / "filtered_streetscapes.csv"
    assert IMAGES_DIR == PROJECT_ROOT / "data" / "images" / "sample"
    assert SCORES_CSV == PROJECT_ROOT / "data" / "scores.csv"


def test_get_google_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert get_google_api_key() is None


def test_get_google_api_key_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert get_google_api_key() == "test-key"
