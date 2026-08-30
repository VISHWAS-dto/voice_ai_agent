"""Integration tests for ``POST /analyze``.

Drives the real FastAPI app through ``TestClient`` — including the lifespan
that loads the actual ``AttributeInferencer`` — so the whole pipeline
(ingest -> VAD -> quality -> inference / skip -> response) is exercised
end to end.

The model load (~1.3 GB, from the local ``.model_cache/``) happens once for
the module via the ``client`` fixture. These tests need the ``ffmpeg``
binary on PATH and the model weights already cached.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.models import (
    AgeBracketPrediction,
    AudioQuality,
    GenderPrediction,
)

SAMPLE_WAV = Path(__file__).resolve().parents[1] / "sample.wav"

_VALID_GENDERS = {g.value for g in GenderPrediction}
_VALID_AGE_BRACKETS = {a.value for a in AgeBracketPrediction}
_VALID_QUALITIES = {q.value for q in AudioQuality}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient whose context runs startup/shutdown (loads the model once)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _assert_response_schema(body: dict) -> None:
    """Assert ``body`` matches the AnalyzeResponse contract shape and enums."""
    assert set(body) == {
        "contact_id",
        "gender",
        "age_bracket",
        "processing_ms",
        "audio_quality",
    }

    # contact_id is a uuid string.
    import uuid

    uuid.UUID(body["contact_id"])

    assert set(body["gender"]) == {"prediction", "confidence"}
    assert body["gender"]["prediction"] in _VALID_GENDERS
    assert 0.0 <= body["gender"]["confidence"] <= 1.0

    assert set(body["age_bracket"]) == {"prediction", "confidence"}
    assert body["age_bracket"]["prediction"] in _VALID_AGE_BRACKETS
    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0

    assert isinstance(body["processing_ms"], int)
    assert body["processing_ms"] >= 0

    assert body["audio_quality"] in _VALID_QUALITIES


@pytest.mark.skipif(not SAMPLE_WAV.is_file(), reason="sample.wav not present")
def test_analyze_sample_wav_matches_schema(client: TestClient) -> None:
    """Uploading a real WAV returns 200 with a schema-valid body."""
    resp = client.post(
        "/analyze",
        files={"audio": ("sample.wav", SAMPLE_WAV.read_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200, resp.text
    _assert_response_schema(resp.json())


def test_analyze_corrupt_file_degrades_to_insufficient(client: TestClient) -> None:
    """A corrupt/undecodable upload returns 'insufficient' + unknowns, not a 500."""
    garbage = io.BytesIO(b"this is definitely not audio \x00\x01\x02\x03" * 8)

    resp = client.post(
        "/analyze",
        files={"audio": ("broken.wav", garbage, "audio/wav")},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_response_schema(body)
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == GenderPrediction.UNKNOWN.value
    assert body["gender"]["confidence"] == 0.0
    assert body["age_bracket"]["prediction"] == AgeBracketPrediction.UNKNOWN.value
    assert body["age_bracket"]["confidence"] == 0.0


def test_analyze_empty_file_degrades_to_insufficient(client: TestClient) -> None:
    """An empty upload is handled gracefully (AudioDecodeError path), not a 500."""
    resp = client.post(
        "/analyze",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_response_schema(body)
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == GenderPrediction.UNKNOWN.value
    assert body["age_bracket"]["prediction"] == AgeBracketPrediction.UNKNOWN.value
