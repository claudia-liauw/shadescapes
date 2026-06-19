import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "filtered_streetscapes.csv"
EXPLORATION_DIR = DATA_DIR / "images" / "exploration"
SCORES_CSV = DATA_DIR / "scores.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")
