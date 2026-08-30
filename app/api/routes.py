"""HTTP routes for the voice attribute inference service.

Primary surface::

    POST /analyze   multipart audio upload -> AnalyzeResponse
    GET  /health    liveness/readiness probe for container healthchecks

Pipeline for one ``/analyze`` request:

1. Read the uploaded file's bytes fully into memory. Nothing is written to
   disk and no ``tempfile`` is used — call recordings routinely contain
   PII, so the audio lives only in this process's memory for the life of
   the request (see :func:`app.audio.ingest.normalize_audio`).
2. Mint a ``contact_id`` (uuid4) for the request.
3. Decode / resample via ``app.audio.ingest.normalize_audio``. A decode
   failure is turned into an ``insufficient`` response, not a 500.
4. Run VAD (``run_vad``) and grade the clip (``assess_quality``).
5. If the grade is ``insufficient``, skip the model entirely and return
   ``unknown`` / ``unknown`` at ``0.0`` confidence — this saves the
   forward-pass compute and avoids the "confident guess on silence"
   failure mode seen in testing.
6. Otherwise run the shared, startup-loaded ``AttributeInferencer``.
7. ``processing_ms`` covers every step (ingest + VAD + inference), not
   just the model's own ``inference_ms``.

Reliability guarantees
----------------------
* **Never a 500.** The whole handler is wrapped in a broad ``try/except``
  final safety net. This is a *soft* personalization signal for a voice AI
  system, so on *any* unexpected error it logs (structured JSON, never raw
  audio bytes) at ERROR and returns a 200 with all-``unknown`` fields and
  ``audio_quality: "insufficient"``.
* **Never hangs.** The pipeline runs under a hard wall-clock budget
  (:data:`_PROCESSING_BUDGET_S`). If ingest + VAD + inference overruns it,
  the request still returns a valid all-``unknown`` / ``insufficient``
  body instead of blocking the caller indefinitely.

Observability
-------------
Exactly one structured log line per ``/analyze`` request, via
``app.logging_config``'s JSON formatter:

* INFO  — normal request (``outcome: "ok"``).
* WARNING — degraded or insufficient audio, incl. decode failure and
  timeout (``outcome: "degraded"``).
* ERROR — the exception fallback path (``outcome: "error_fallback"``).

Only the generated ``contact_id`` identifies the request. The upload's
filename, byte content, and any transcript are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.audio.ingest import AudioDecodeError, normalize_audio, run_vad
from app.audio.quality import assess_quality
from app.schemas.models import (
    AgeBracketResult,
    AnalyzeResponse,
    AudioQuality,
    GenderResult,
)

logger = logging.getLogger("app.api.analyze")

router = APIRouter()

# Sample rate everything downstream of normalize_audio() runs at.
_TARGET_SAMPLE_RATE = 16_000

# Hard wall-clock budget for the whole /analyze pipeline (read + ingest +
# VAD + inference). On a warm process the real cost is ~100-300 ms; 3 s is
# generous headroom for a slow clip while still guaranteeing the caller
# gets an answer promptly. On overrun we return all-unknown / insufficient
# rather than letting the request hang.
_PROCESSING_BUDGET_S = 3.0


def _unknown_response(contact_id: uuid.UUID, quality: str, processing_ms: int) -> AnalyzeResponse:
    """Build an all-``unknown`` response (used for skips, decode errors, timeouts, crashes)."""
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderResult(prediction="unknown", confidence=0.0),
        age_bracket=AgeBracketResult(prediction="unknown", confidence=0.0),
        processing_ms=processing_ms,
        audio_quality=quality,
    )


def _run_pipeline(raw_bytes: bytes, inferencer) -> tuple[str, dict]:
    """Blocking core of /analyze: decode -> VAD -> quality -> inference.

    Split out from the handler so it can run in a worker thread under an
    ``asyncio.wait_for`` timeout. Pure function of its args plus the shared
    model weights; holds no state.

    Returns:
        ``(audio_quality, prediction_dict)``. When the quality grade is
        ``insufficient`` the model is skipped and ``prediction_dict`` is
        the all-unknown shape.

    Raises:
        AudioDecodeError: If ``raw_bytes`` cannot be decoded.
    """
    unknown_prediction = {
        "gender_prediction": "unknown",
        "gender_confidence": 0.0,
        "age_bracket": "unknown",
        "age_confidence": 0.0,
        "inference_ms": 0.0,
    }

    waveform = normalize_audio(raw_bytes)

    vad_stats = run_vad(waveform, _TARGET_SAMPLE_RATE)
    audio_quality = assess_quality(vad_stats)

    if audio_quality == AudioQuality.INSUFFICIENT.value:
        return audio_quality, dict(unknown_prediction)

    prediction = inferencer.predict(waveform, _TARGET_SAMPLE_RATE)
    return audio_quality, prediction


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    audio: UploadFile = File(...),
    contact_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Infer speaker gender and age bracket from an uploaded audio clip.

    Args:
        request: The incoming request; used only to reach the
            startup-loaded ``AttributeInferencer`` on ``app.state``.
        audio: Uploaded call recording (wav/mp3/m4a/opus/…). Read fully
            into memory and decoded by ``app.audio.ingest`` — never
            written to disk.
        contact_id: Ignored for identity purposes; a fresh uuid4 is always
            minted per request so the value is unambiguous in logs. (Kept
            in the signature for backward compatibility with callers that
            still send the field.)

    Returns:
        An :class:`AnalyzeResponse`. Always HTTP 200: decode failures,
        unusable audio, processing timeouts, and unexpected internal
        errors all resolve to an all-``unknown`` body rather than an error
        status, because this is a best-effort personalization hint and
        must never break the calling voice AI system.
    """
    # Every step is inside the timed region: processing_ms is end-to-end
    # (read + ingest + VAD + quality + inference), not just model time.
    started = time.perf_counter()

    # uuid4 per request — see docstring. Generated before any work so we
    # can attach it to every log line, including failure logs.
    request_id = uuid.uuid4()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        # 1. Raw bytes into memory. NO disk write, NO tempfile: the audio
        #    can contain customer PII and consent-limited recording data,
        #    so it must never be persisted. UploadFile may have spooled a
        #    large body to a SpooledTemporaryFile internally, but we only
        #    ever hold the returned bytes and let it close immediately.
        raw_bytes = await audio.read()
        upload_bytes = len(raw_bytes)

        inferencer = request.app.state.inferencer

        # 2. Run the blocking pipeline in a worker thread under a hard
        #    wall-clock budget. If it overruns, we stop waiting and return
        #    a safe all-unknown body instead of hanging the caller. (The
        #    orphaned thread finishes on its own and is discarded; the
        #    pipeline holds no shared mutable state.)
        try:
            audio_quality, prediction = await asyncio.wait_for(
                asyncio.to_thread(_run_pipeline, raw_bytes, inferencer),
                timeout=_PROCESSING_BUDGET_S,
            )
        except asyncio.TimeoutError:
            processing_ms = _elapsed_ms()
            logger.warning(
                "analyze: processing budget exceeded, returning all-unknown",
                extra={
                    "contact_id": str(request_id),
                    "audio_quality": AudioQuality.INSUFFICIENT.value,
                    "processing_ms": processing_ms,
                    "gender_prediction": "unknown",
                    "age_prediction": "unknown",
                    "outcome": "degraded",
                    "timed_out": True,
                    "timeout_ms": int(_PROCESSING_BUDGET_S * 1000),
                },
            )
            return _unknown_response(
                request_id, AudioQuality.INSUFFICIENT.value, processing_ms
            )
        except AudioDecodeError as exc:
            # 3. A decode failure (empty upload, corrupt container, unknown
            #    codec) is a client-data problem, not a server fault:
            #    report "insufficient" with unknown predictions, not a 500.
            processing_ms = _elapsed_ms()
            logger.warning(
                "analyze: audio decode failed",
                extra={
                    "contact_id": str(request_id),
                    "audio_quality": AudioQuality.INSUFFICIENT.value,
                    "processing_ms": processing_ms,
                    "gender_prediction": "unknown",
                    "age_prediction": "unknown",
                    "outcome": "degraded",
                    "decode_error": str(exc),
                    "upload_bytes": upload_bytes,
                },
            )
            return _unknown_response(
                request_id, AudioQuality.INSUFFICIENT.value, processing_ms
            )

        processing_ms = _elapsed_ms()

        response = AnalyzeResponse(
            contact_id=request_id,
            gender=GenderResult(
                prediction=prediction["gender_prediction"],
                confidence=prediction["gender_confidence"],
            ),
            age_bracket=AgeBracketResult(
                prediction=prediction["age_bracket"],
                confidence=prediction["age_confidence"],
            ),
            processing_ms=processing_ms,
            audio_quality=audio_quality,
        )

        # One structured line per request. INFO for a clean "good" read;
        # WARNING when the audio was only "degraded" or "insufficient"
        # (still a 200, but the caller should weight the result less).
        degraded = audio_quality != AudioQuality.GOOD.value
        log = logger.warning if degraded else logger.info
        log(
            "analyze: %s",
            "degraded audio" if degraded else "ok",
            extra={
                "contact_id": str(request_id),
                "audio_quality": audio_quality,
                "processing_ms": processing_ms,
                "gender_prediction": prediction["gender_prediction"],
                "gender_confidence": round(prediction["gender_confidence"], 3),
                "age_prediction": prediction["age_bracket"],
                "age_confidence": round(prediction["age_confidence"], 3),
                "inference_ms": round(prediction.get("inference_ms", 0.0), 1),
                "outcome": "degraded" if degraded else "ok",
            },
        )
        return response

    except Exception as exc:  # noqa: BLE001 - final safety net: degrade, never 500.
        processing_ms = _elapsed_ms()
        # exc_info gives us the traceback; we log NO request body / audio
        # bytes, only the generated id, timing, and the exception type.
        logger.error(
            "analyze: unexpected error, returning all-unknown",
            exc_info=True,
            extra={
                "contact_id": str(request_id),
                "audio_quality": AudioQuality.INSUFFICIENT.value,
                "processing_ms": processing_ms,
                "gender_prediction": "unknown",
                "age_prediction": "unknown",
                "outcome": "error_fallback",
                "error_type": type(exc).__name__,
            },
        )
        return _unknown_response(
            request_id, AudioQuality.INSUFFICIENT.value, processing_ms
        )


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness / readiness probe for container healthchecks.

    Returns ``{"status": "ok", "model_loaded": <bool>}``. ``model_loaded``
    is ``True`` only once the startup lifespan has finished loading the
    ``AttributeInferencer`` onto ``app.state`` — so an orchestrator can
    hold traffic until the ~1.3 GB model is actually resident.
    """
    model_loaded = getattr(request.app.state, "inferencer", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


# Backwards-compatible alias. Some probes were wired to /healthz before the
# /health rename; keep both pointing at the same check.
@router.get("/healthz")
async def healthz(request: Request) -> dict:
    """Alias for :func:`health` (kept for existing probe configs)."""
    return await health(request)
