from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ShadeScore(BaseModel):
    pedestrian_shade_score: float = Field(ge=0.0, le=1.0)
    shade_sources: list[str]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(min_length=1, max_length=500)


class ScoreSummary(BaseModel):
    scored: int
    skipped: int
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    skips: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    progress: list[str] = Field(default_factory=list)
    elapsed_seconds: float | None = None

    @computed_field
    @property
    def message(self) -> str:
        scored_noun = "image" if self.scored == 1 else "images"
        skipped_noun = "image" if self.skipped == 1 else "images"
        message = f"Scored {self.scored} {scored_noun}, skipped {self.skipped} {skipped_noun}"

        reason_labels = {
            "already_scored": "already scored",
            "missing_metadata": "missing metadata",
        }
        reason_parts = []
        for key, count in self.skip_reasons.items():
            if not count:
                continue
            noun = "image" if count == 1 else "images"
            reason_parts.append(f"{count} {noun} {reason_labels.get(key, key.replace('_', ' '))}")
        if reason_parts:
            message += f" ({', '.join(reason_parts)})"

        message += "."
        if self.errors:
            error_noun = "error" if len(self.errors) == 1 else "errors"
            message += f" {len(self.errors)} scoring {error_noun}."
        if self.elapsed_seconds is not None:
            message += f" Completed in {self.elapsed_seconds:.1f}s."
        return message


class NoImagesError(Exception):
    """Raised when the exploration image directory has no JPEG files."""


class NoMetadataError(Exception):
    """Raised when filtered_streetscapes.csv is missing."""


class MissingApiKeyError(Exception):
    """Raised when GOOGLE_API_KEY is not set."""
