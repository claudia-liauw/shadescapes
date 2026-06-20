from typing import Literal

from pydantic import BaseModel, Field


class ShadeScore(BaseModel):
    pedestrian_shade_score: float = Field(ge=0.0, le=1.0)
    shade_sources: list[str]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(min_length=1, max_length=500)


class ScoreSummary(BaseModel):
    scored: int
    skipped: int
    errors: list[str]


class NoImagesError(Exception):
    """Raised when the exploration image directory has no JPEG files."""


class NoMetadataError(Exception):
    """Raised when filtered_streetscapes.csv is missing."""


class MissingApiKeyError(Exception):
    """Raised when GOOGLE_API_KEY is not set."""
