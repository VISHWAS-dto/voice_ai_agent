"""Integration tests for ``POST /analyze`` and ``GET /health``.

Drives the real FastAPI app through ``TestClient`` — including the lifespan
that loads the actual ``AttributeInferencer`` — so the whole pipeline
(ingest -> VAD -> quality -> inference / skip -> response) is exercised
end to end against the real model, not a mock.

The model load (~1.3 GB, from the local ``.model_cache/``) happens once for
the module via the ``client`` fixture. These tests need the ``ffmpeg``
binary on PATH and the model weights already cached; if ``sample.wav`` is
absent the WAV test is skipped rather than failed.

Note on the upload field name: the endpoint takes the file part as
``audio`` (see ``app/api/routes.py``), so every ``files={...}`` below uses
that key.
"""

from __future__ import annotations

import io
import uuid
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

_VALID_GENDERS = {g.value for g in GenderPrediction}          # male / female / unknown
_VALID_AGE_BRACKETS = {a.value for a in AgeBracketPrediction}  # 18-30 / 31-45 / 46-60 / 60+ / unknown
_VALID_QUALITIES = {q.value for q in AudioQuality}             # good / degraded / insufficient

_REQUIRED_TOP_LEVEL_KEYS = {
    "contact_id",
    "gender",
    "age_bracket",
    "processing_ms",
    "audio_quality",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient whose context runs startup/shutdown (loads the model once)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _assert_is_uuid4(value: str) -> None:
    """contact_id must be a parseable, version-4 UUID string."""
    parsed = uuid.UUID(value)
    assert parsed.version == 4, f"contact_id is UUID v{parsed.version}, expected v4"


def _assert_response_schema(body: dict) -> None:
    """Assert ``body`` matches the AnalyzeResponse contract shape and value domains."""
    # 1. exactly the required keys, nothing more, nothing missing.
    assert set(body) == _REQUIRED_TOP_LEVEL_KEYS

    # contact_id: a valid uuid4.
    assert isinstance(body["contact_id"], str)
    _assert_is_uuid4(body["contact_id"])

    # gender: prediction in the enum, confidence a float in [0, 1].
    assert set(body["gender"]) == {"prediction", "confidence"}
    assert body["gender"]["prediction"] in _VALID_GENDERS
    g_conf = body["gender"]["confidence"]
    assert isinstance(g_conf, float)
    assert 0.0 <= g_conf <= 1.0

    # age_bracket: prediction one of the 4 brackets or "unknown", confidence float in [0, 1].
    assert set(body["age_bracket"]) == {"prediction", "confidence"}
    assert body["age_bracket"]["prediction"] in _VALID_AGE_BRACKETS
    a_conf = body["age_bracket"]["confidence"]
    assert isinstance(a_conf, float)
    assert 0.0 <= a_conf <= 1.0

    # processing_ms: a non-negative integer.
    assert isinstance(body["processing_ms"], int)
    assert body["processing_ms"] >= 0

    # audio_quality: one of the three ratings.
    assert body["audio_quality"] in _VALID_QUALITIES


# --- 1. a real WAV -----------------------------------------------------


@pytest.mark.skipif(not SAMPLE_WAV.is_file(), reason="sample.wav not present")
def test_analyze_sample_wav_returns_200_and_valid_schema(client: TestClient) -> None:
    """Uploading sample.wav returns 200 with a fully schema-valid body.

    We assert the *shape* and value domains, not specific predictions:
    which gender/age the model reports for this particular clip is not part
    of the API contract and would make the test brittle.
    """
    resp = client.post(
        "/analyze",
        files={"audio": ("sample.wav", SAMPLE_WAV.read_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_response_schema(body)

    # Spell out the checklist assertions individually too, so a failure
    # names the exact expectation that broke.
    assert body["gender"]["prediction"] in {"male", "female", "unknown"}
    assert body["age_bracket"]["prediction"] in {
        "18-30",
        "31-45",
        "46-60",
        "60+",
        "unknown",
    }
    assert isinstance(body["gender"]["confidence"], float)
    assert isinstance(body["age_bracket"]["confidence"], float)
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0
    _assert_is_uuid4(body["contact_id"])


@pytest.mark.skipif(not SAMPLE_WAV.is_file(), reason="sample.wav not present")
def test_analyze_mints_a_fresh_contact_id_per_request(client: TestClient) -> None:
    """Each call gets its own uuid4, even if the caller sends one in the form."""
    payload = {"audio": ("sample.wav", SAMPLE_WAV.read_bytes(), "audio/wav")}

    first = client.post("/analyze", files=payload).json()["contact_id"]
    second = client.post(
        "/analyze",
        files=payload,
        data={"contact_id": "caller-supplied-value"},
    ).json()["contact_id"]

    _assert_is_uuid4(first)
    _assert_is_uuid4(second)
    assert first != second


# --- 2. corrupt / empty uploads: 200, not 500 -----------------------


def test_analyze_corrupt_file_is_insufficient_not_500(client: TestClient) -> None:
    """A corrupt/undecodable upload returns 200 'insufficient' + unknowns."""
    garbage = io.BytesIO(b"this is definitely not audio \x00\x01\x02\x03" * 8)

    resp = client.post(
        "/analyze",
        files={"audio": ("broken.wav", garbage, "audio/wav")},
    )

    assert resp.status_code == 200, resp.text  # explicitly NOT 500
    body = resp.json()
    _assert_response_schema(body)
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == GenderPrediction.UNKNOWN.value
    assert body["gender"]["confidence"] == 0.0
    assert body["age_bracket"]["prediction"] == AgeBracketPrediction.UNKNOWN.value
    assert body["age_bracket"]["confidence"] == 0.0


def test_analyze_empty_file_is_insufficient_not_500(client: TestClient) -> None:
    """An empty upload (AudioDecodeError path) is handled gracefully, not a 500."""
    resp = client.post(
        "/analyze",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )

    assert resp.status_code == 200, resp.text  # explicitly NOT 500
    body = resp.json()
    _assert_response_schema(body)
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == GenderPrediction.UNKNOWN.value
    assert body["age_bracket"]["prediction"] == AgeBracketPrediction.UNKNOWN.value


# --- 3. GET /health --------------------------------------------------


def test_health_returns_200_and_model_loaded_true(client: TestClient) -> None:
    """Inside the lifespan, /health is 200 with model_loaded: true."""
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
