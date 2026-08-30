"""Phase 5 reliability + observability tests.

Covers the guarantees that a caller's voice AI system depends on:

* ``GET /health`` reports process + model-load state.
* A pipeline that overruns the wall-clock budget still returns a valid
  200 (all-unknown / insufficient), it does not hang.
* An unexpected exception anywhere in the pipeline returns 200 with the
  all-unknown body and is logged at ERROR with ``outcome=error_fallback``.
* Every ``/analyze`` request emits exactly one structured log line with
  the required fields and no PII beyond ``contact_id``.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.logging_config import JsonFormatter
from app.main import create_app
from app.schemas.models import AudioQuality, GenderPrediction

SAMPLE_WAV = Path(__file__).resolve().parents[1] / "sample.wav"


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as c:
        yield c


# --- GET /health ---------------------------------------------------------


def test_health_ok_and_model_loaded(client: TestClient) -> None:
    """Inside the lifespan, /health is 200 and model_loaded is True."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_before_model_load_reports_not_loaded() -> None:
    """Without the lifespan the model is absent; /health says so, still 200."""
    app = create_app()
    # No `with TestClient(...)` -> lifespan startup does not run.
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": False}


def test_healthz_alias_matches_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == client.get("/health").json()


# --- timeout guard -----------------------------------------------------


def test_processing_timeout_returns_valid_unknown_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """If the pipeline overruns the budget, the request still returns a
    valid all-unknown / insufficient 200 instead of hanging."""

    def _slow_pipeline(raw_bytes: bytes, inferencer):  # noqa: ARG001
        import time

        time.sleep(5.0)  # well past the (patched) budget
        raise AssertionError("should have been cancelled by wait_for")

    monkeypatch.setattr(routes, "_run_pipeline", _slow_pipeline)
    monkeypatch.setattr(routes, "_PROCESSING_BUDGET_S", 0.5)

    with caplog.at_level(logging.WARNING, logger="app.api.analyze"):
        resp = client.post(
            "/analyze",
            files={"audio": ("x.wav", b"anything", "audio/wav")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == GenderPrediction.UNKNOWN.value
    assert body["age_bracket"]["prediction"] == "unknown"
    assert body["processing_ms"] >= 0

    rec = next(r for r in caplog.records if getattr(r, "timed_out", False))
    assert rec.levelno == logging.WARNING
    assert rec.outcome == "degraded"


# --- exception fallback path -----------------------------------------


def test_unexpected_exception_degrades_to_unknown_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Any unexpected error -> 200 all-unknown, logged at ERROR."""

    def _boom(raw_bytes: bytes, inferencer):  # noqa: ARG001
        raise RuntimeError("simulated model blowup")

    monkeypatch.setattr(routes, "_run_pipeline", _boom)

    with caplog.at_level(logging.ERROR, logger="app.api.analyze"):
        resp = client.post(
            "/analyze",
            files={"audio": ("x.wav", b"anything", "audio/wav")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["audio_quality"] == AudioQuality.INSUFFICIENT.value
    assert body["gender"]["prediction"] == "unknown"
    assert body["age_bracket"]["prediction"] == "unknown"

    rec = next(r for r in caplog.records if getattr(r, "outcome", None) == "error_fallback")
    assert rec.levelno == logging.ERROR
    assert rec.error_type == "RuntimeError"


# --- structured logging / PII --------------------------------------


@pytest.mark.skipif(not SAMPLE_WAV.is_file(), reason="sample.wav not present")
def test_analyze_emits_one_structured_json_line_without_pii(
    client: TestClient, caplog
) -> None:
    """A normal /analyze logs exactly one JSON line carrying the required
    telemetry and nothing that identifies the upload beyond contact_id."""
    fmt = JsonFormatter()

    with caplog.at_level(logging.INFO, logger="app.api.analyze"):
        resp = client.post(
            "/analyze",
            files={"audio": ("caller-jane-doe-ssn-123.wav", SAMPLE_WAV.read_bytes(), "audio/wav")},
        )
    assert resp.status_code == 200

    lines = [
        r for r in caplog.records if r.name == "app.api.analyze"
    ]
    assert len(lines) == 1

    payload = json.loads(fmt.format(lines[0]))
    for key in (
        "contact_id",
        "audio_quality",
        "gender_prediction",
        "age_prediction",
        "processing_ms",
        "outcome",
    ):
        assert key in payload, key

    assert payload["outcome"] in {"ok", "degraded"}
    assert payload["contact_id"] == resp.json()["contact_id"]

    # No PII: the filename and its parts must not appear anywhere in the line.
    blob = json.dumps(payload).lower()
    for needle in ("caller-jane", "jane-doe", "ssn", ".wav", "filename"):
        assert needle not in blob
