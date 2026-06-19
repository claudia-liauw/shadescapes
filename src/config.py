import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "filtered_streetscapes.csv"
IMAGES_DIR = DATA_DIR / "images" / "exploration"
SCORES_CSV = DATA_DIR / "scores.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")
