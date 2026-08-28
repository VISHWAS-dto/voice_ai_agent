"""HTTP routes for the voice attribute inference service.

Stub only. This module will expose the REST surface, primarily::

    POST /analyze   multipart audio upload -> AnalyzeResponse

Implementation notes for later:
- Accept an uploaded audio file (multipart/form-data) plus an optional
  ``contact_id`` form field; generate one if absent.
- Delegate decoding/resampling to ``app.audio.ingest``.
- Run ``app.audio.quality`` gating; short-circuit to an ``insufficient``
  response when the clip is unusable.
- Call ``app.inference.model`` for gender + age-bracket predictions.
- Measure wall-clock time and populate ``processing_ms``.
"""

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.models import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    audio: UploadFile = File(...),
    contact_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Infer speaker gender and age bracket from an uploaded audio clip.

    Args:
        audio: Uploaded call recording (e.g. wav/mp3/opus). Decoded and
            resampled downstream by ``app.audio.ingest``.
        contact_id: Optional caller-supplied contact/call identifier. A new
            UUID is assigned when omitted.

    Returns:
        An :class:`AnalyzeResponse` with predictions, per-attribute
        confidences, the audio-quality rating, and processing time.
    """
    raise NotImplementedError


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe.

    Will return a small JSON payload indicating process health and, later,
    whether the inference model has finished loading.
    """
    raise NotImplementedError
