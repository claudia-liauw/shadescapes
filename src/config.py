import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "filtered_streetscapes.csv"
IMAGES_DIR = DATA_DIR / "images" / "exploration"
SCORES_CSV = DATA_DIR / "scores.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def get_google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY")
