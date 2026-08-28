"""Pydantic response schemas for the voice attribute inference service.

These models define the public API contract for ``POST /analyze``. They are
intentionally the only fleshed-out part of the scaffold; every other module
is a stub to be implemented one at a time.
"""

from enum import Enum

from pydantic import BaseModel, Field
from uuid import UUID


class GenderPrediction(str, Enum):
    """Possible values for an inferred speaker gender."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeBracketPrediction(str, Enum):
    """Coarse age buckets used for logistics call-center analytics."""

    B_18_30 = "18-30"
    B_31_45 = "31-45"
    B_46_60 = "46-60"
    B_60_PLUS = "60+"
    UNKNOWN = "unknown"


class AudioQuality(str, Enum):
    """Overall usability rating for the submitted audio.

    ``insufficient`` means the clip could not be scored reliably (too short,
    too noisy, or mostly silence) and predictions will be ``unknown``.
    """

    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class GenderResult(BaseModel):
    """Gender inference with an associated model confidence."""

    prediction: GenderPrediction
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted gender, in [0, 1].",
    )


class AgeBracketResult(BaseModel):
    """Age-bracket inference with an associated model confidence."""

    prediction: AgeBracketPrediction
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted age bracket, in [0, 1].",
    )


class AnalyzeResponse(BaseModel):
    """Response body for ``POST /analyze``.

    Example::

        {
            "contact_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            "gender": {"prediction": "male", "confidence": 0.87},
            "age_bracket": {"prediction": "31-45", "confidence": 0.63},
            "processing_ms": 142,
            "audio_quality": "good"
        }
    """

    contact_id: UUID = Field(
        ...,
        description="Identifier for the contact/call this audio belongs to.",
    )
    gender: GenderResult
    age_bracket: AgeBracketResult
    processing_ms: int = Field(
        ...,
        ge=0,
        description="Wall-clock time spent producing this response, milliseconds.",
    )
    audio_quality: AudioQuality
