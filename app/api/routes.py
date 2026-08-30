"""HTTP routes for the voice attribute inference service.

Primary surface::

    POST /analyze   multipart audio upload -> AnalyzeResponse

Pipeline for one request:

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

The whole handler is wrapped in a broad ``try/except`` final safety net:
this is a soft personalization signal for a voice AI system, so on *any*
unexpected error it logs (structured, never raw audio bytes) and returns a
200 with all-``unknown`` fields rather than a 500 that could disrupt the
caller.
"""

from __future__ import annotations

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


def _unknown_response(contact_id: uuid.UUID, quality: str, processing_ms: int) -> AnalyzeResponse:
    """Build an all-``unknown`` response (used for skips, decode errors, crashes)."""
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderResult(prediction="unknown", confidence=0.0),
        age_bracket=AgeBracketResult(prediction="unknown", confidence=0.0),
        processing_ms=processing_ms,
        audio_quality=quality,
    )


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
        unusable audio, and unexpected internal errors all resolve to an
        all-``unknown`` body rather than an error status, because this is
        a best-effort personalization hint and must never break the
        calling voice AI system.
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

        # 3. Decode / resample to 16 kHz mono float32, in memory. A decode
        #    failure (empty upload, corrupt container, unknown codec) is a
        #    client-data problem, not a server fault: report it as
        #    "insufficient" with unknown predictions rather than a 500.
        try:
            waveform = normalize_audio(raw_bytes)
        except AudioDecodeError as exc:
            processing_ms = _elapsed_ms()
            logger.info(
                "analyze: audio decode failed",
                extra={
                    "contact_id": str(request_id),
                    "audio_quality": AudioQuality.INSUFFICIENT.value,
                    "processing_ms": processing_ms,
                    "gender_prediction": "unknown",
                    "age_prediction": "unknown",
                    "decode_error": str(exc),
                    "upload_bytes": len(raw_bytes),
                },
            )
            return _unknown_response(
                request_id, AudioQuality.INSUFFICIENT.value, processing_ms
            )

        # 5. Voice-activity stats, then 6. the coarse quality grade.
        vad_stats = run_vad(waveform, _TARGET_SAMPLE_RATE)
        audio_quality = assess_quality(vad_stats)

        # 7. Insufficient speech -> skip the model entirely. Saves the
        #    forward-pass compute and avoids the "confident guess on
        #    silence" failure mode.
        if audio_quality == AudioQuality.INSUFFICIENT.value:
            processing_ms = _elapsed_ms()
            logger.info(
                "analyze: insufficient audio, skipped inference",
                extra={
                    "contact_id": str(request_id),
                    "audio_quality": audio_quality,
                    "processing_ms": processing_ms,
                    "gender_prediction": "unknown",
                    "age_prediction": "unknown",
                    "speech_ratio": round(vad_stats.get("speech_ratio", 0.0), 3),
                    "total_duration_s": round(vad_stats.get("total_duration_s", 0.0), 2),
                },
            )
            return _unknown_response(request_id, audio_quality, processing_ms)

        # 8. Run the shared inferencer loaded once at startup (see
        #    app/main.py lifespan). predict() has its own internal
        #    try/except and degrades to unknown/0.0 on any model error.
        inferencer = request.app.state.inferencer
        prediction = inferencer.predict(waveform, _TARGET_SAMPLE_RATE)

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

        # 12. Request logging: contact_id, audio_quality, processing_ms,
        #     and the predictions. NEVER the raw audio — only its length.
        logger.info(
            "analyze: ok",
            extra={
                "contact_id": str(request_id),
                "audio_quality": audio_quality,
                "processing_ms": processing_ms,
                "gender_prediction": prediction["gender_prediction"],
                "gender_confidence": round(prediction["gender_confidence"], 3),
                "age_prediction": prediction["age_bracket"],
                "age_confidence": round(prediction["age_confidence"], 3),
                "inference_ms": round(prediction.get("inference_ms", 0.0), 1),
            },
        )
        return response

    except Exception:  # noqa: BLE001 - final safety net: degrade, never 500.
        processing_ms = _elapsed_ms()
        # exc_info gives us the traceback; we log NO request body / audio
        # bytes, only the generated id and timing.
        logger.exception(
            "analyze: unexpected error, returning all-unknown",
            extra={
                "contact_id": str(request_id),
                "audio_quality": AudioQuality.INSUFFICIENT.value,
                "processing_ms": processing_ms,
                "gender_prediction": "unknown",
                "age_prediction": "unknown",
            },
        )
        return _unknown_response(
            request_id, AudioQuality.INSUFFICIENT.value, processing_ms
        )


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    """Liveness / readiness probe.

    Reports process health and whether the inference model has finished
    loading onto ``app.state``.
    """
    model_loaded = getattr(request.app.state, "inferencer", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}
