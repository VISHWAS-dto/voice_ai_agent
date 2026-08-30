"""WebSocket endpoint for streaming voice attribute inference.

A **bonus** near-real-time alternative to the batch ``POST /analyze`` route.
A client (simulating a live call leg) opens ``/ws/analyze`` and pushes raw
audio bytes progressively; the server buffers them into fixed-length
*windows* and, once a window has accumulated enough audio, runs it through
**exactly the same pipeline** as ``/analyze``::

    normalize_audio()  ->  run_vad()  ->  assess_quality()
        -> (skip if "insufficient")  ->  AttributeInferencer.predict()

After every window the server emits one JSON message with the same shape as
:class:`app.schemas.models.AnalyzeResponse`, plus:

* ``window_index``  — 0-based counter so the client can watch predictions
  evolve over the call.
* ``best_so_far``    — the running "best" prediction (see the strategy note
  on :class:`_StreamSession.observe`).

This module is a **thin orchestration layer**. It does not re-implement
decode / VAD / quality / inference — it calls
``app.audio.ingest``, ``app.audio.quality`` and ``app.inference.model``
as-is. The only new logic is (a) buffering bytes into windows and (b) the
running-best aggregation across windows.

Privacy — identical rules to ``/analyze``
----------------------------------------
* **Nothing is ever written to disk.** Incoming chunks are appended to an
  in-memory ``bytearray`` and decoded by streaming into ffmpeg over a pipe
  (``normalize_audio`` already guarantees this). No ``tempfile``, no
  scratch ``.wav``.
* **No raw audio and no raw bytes are logged.** Per-window log lines carry
  only the generated ``contact_id``, the window index, byte/sample *counts*,
  timing, the quality grade and the prediction *labels* — never the bytes
  themselves, a filename, or a transcript.

Resource guards
---------------
A client must not be able to hold the socket open forever or make the
server buffer unbounded audio:

* :data:`_MAX_SESSION_S`     — hard wall-clock cap on the whole session.
* :data:`_MAX_TOTAL_BYTES`   — hard cap on cumulative received audio bytes.
* :data:`_MAX_WINDOWS`       — hard cap on number of processed windows.

Hitting any guard sends a final ``{"event": "closing", "reason": ...}``
message and closes the socket with a policy-violation code.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.audio.ingest import AudioDecodeError, normalize_audio, run_vad
from app.audio.quality import assess_quality
from app.schemas.models import AudioQuality

logger = logging.getLogger("app.api.ws_analyze")

router = APIRouter()

# --- streaming parameters ------------------------------------------------
#
# The client sends *raw* PCM: 16 kHz, mono, little-endian float32 (the same
# canonical format normalize_audio() emits). That lets us size windows in
# bytes precisely and lets ffmpeg skip container demuxing. The wire format
# is fixed rather than negotiated to keep this bonus endpoint simple; the
# test client (scripts/test_websocket.py) converts sample.wav to it once up
# front.
_STREAM_SAMPLE_RATE = 16_000
_BYTES_PER_SAMPLE = 4  # float32
_STREAM_INPUT_FORMAT = "f32le"  # ffmpeg -f value for headerless float32 PCM

# Rolling analysis window. 2 s is the shortest span the quality gate and the
# model give a non-random read from (see app/audio/quality.py's 0.5 s floor
# and the model card notes) while still refreshing the prediction often
# enough to feel "live".
_WINDOW_S = 2.0
_WINDOW_BYTES = int(_WINDOW_S * _STREAM_SAMPLE_RATE) * _BYTES_PER_SAMPLE

# Non-overlapping windows: once we've processed [0, 2s) we drop it and start
# filling [2s, 4s). Simpler to reason about than a sliding window and each
# JSON message then corresponds to a distinct slice of the call.
#
# --- resource guards ---------------------------------------------------
# A misbehaving or malicious client must not pin the socket open or make us
# buffer unbounded audio.
_MAX_SESSION_S = 120.0  # hard wall-clock cap on one streaming session
_MAX_WINDOWS = 60  # at 2 s/window this is _MAX_SESSION_S of audio
_MAX_TOTAL_BYTES = int(
    (_MAX_WINDOWS * _WINDOW_S + _WINDOW_S) * _STREAM_SAMPLE_RATE * _BYTES_PER_SAMPLE
)  # cumulative received-bytes cap, with one window of slack

# Per-window pipeline budget, mirroring routes.py's _PROCESSING_BUDGET_S.
# If decode + VAD + inference for one window overruns this we emit an
# all-unknown / insufficient message for that window rather than stalling
# the stream.
_WINDOW_BUDGET_S = 3.0

# How long to wait for the next chunk before treating the client as gone.
_RECV_TIMEOUT_S = 30.0


def _unknown_payload() -> dict:
    """The all-``unknown`` prediction dict (decode failure / skip / timeout / crash)."""
    return {
        "gender_prediction": "unknown",
        "gender_confidence": 0.0,
        "age_bracket": "unknown",
        "age_confidence": 0.0,
        "inference_ms": 0.0,
    }


def _run_window_pipeline(pcm_bytes: bytes, inferencer) -> tuple[str, dict]:
    """Decode -> VAD -> quality -> inference for one window's worth of PCM.

    Byte-for-byte the same sequence as ``routes.py::_run_pipeline`` — this
    is the reuse the task asks for, not a re-implementation. Split out so it
    can run in a worker thread under a timeout.

    Args:
        pcm_bytes: Raw 16 kHz mono float32 little-endian PCM for one window.
        inferencer: The shared, startup-loaded ``AttributeInferencer``.

    Returns:
        ``(audio_quality, prediction_dict)``. When the grade is
        ``insufficient`` the model is skipped and the dict is all-unknown.

    Raises:
        AudioDecodeError: If ffmpeg cannot decode ``pcm_bytes``.
    """
    # normalize_audio() with an explicit format hint: the payload is
    # headerless float32 PCM, so ffmpeg cannot sniff it. It still goes
    # through ffmpeg (not np.frombuffer) so this stays a single code path
    # with /analyze and picks up the same clipping / validation.
    waveform = normalize_audio(pcm_bytes, input_format=_STREAM_INPUT_FORMAT)

    vad_stats = run_vad(waveform, _STREAM_SAMPLE_RATE)
    audio_quality = assess_quality(vad_stats)

    if audio_quality == AudioQuality.INSUFFICIENT.value:
        return audio_quality, _unknown_payload()

    prediction = inferencer.predict(waveform, _STREAM_SAMPLE_RATE)
    return audio_quality, prediction


class _StreamSession:
    """Per-connection state: the rolling byte buffer and the running best.

    One instance lives for the duration of one ``/ws/analyze`` connection
    and is discarded (with its buffer) when the socket closes.
    """

    def __init__(self, contact_id: uuid.UUID) -> None:
        self.contact_id = contact_id
        self._buf = bytearray()  # in-memory only; never flushed to disk
        self.total_bytes = 0
        self.window_index = 0
        self.started = time.perf_counter()

        # Running-best aggregation. See observe() for the strategy.
        self.best: dict | None = None

    # --- buffering ---------------------------------------------------
    def add_chunk(self, chunk: bytes) -> None:
        """Append a received binary chunk to the rolling buffer."""
        self._buf.extend(chunk)
        self.total_bytes += len(chunk)

    def has_full_window(self) -> bool:
        return len(self._buf) >= _WINDOW_BYTES

    def take_window(self) -> bytes:
        """Detach and return exactly one window of PCM from the front of the buffer."""
        window = bytes(self._buf[:_WINDOW_BYTES])
        del self._buf[:_WINDOW_BYTES]
        return window

    def take_remainder(self) -> bytes:
        """Detach and return whatever partial audio is left (for the final flush)."""
        rest = bytes(self._buf)
        self._buf.clear()
        return rest

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    # --- running-best strategy ------------------------------------
    def observe(self, quality: str, prediction: dict) -> dict:
        """Fold one window's result into the running "best" and return it.

        Strategy — **highest gender confidence seen so far wins** (a plain
        argmax over windows), with age carried alongside from that same
        window.

        Why this and not confidence-averaging: averaging only makes sense
        while the *same* class keeps winning, so it needs per-class
        bookkeeping and a tie/`switch` policy when the winning class
        changes mid-call — more moving parts, more to get subtly wrong.
        "keep the single most confident window" is one comparison, is
        obviously correct, and matches how a human would skim evolving
        predictions ("ignore the noisy early ones, trust the point where
        the model was most sure"). Windows graded ``insufficient`` (model
        skipped, confidence 0.0) never displace a real prediction.

        Returns:
            The current best as an ``AnalyzeResponse``-shaped-ish dict
            (prediction fields + the quality grade + which window it came
            from).
        """
        candidate = {
            "gender_prediction": prediction["gender_prediction"],
            "gender_confidence": prediction["gender_confidence"],
            "age_bracket": prediction["age_bracket"],
            "age_confidence": prediction["age_confidence"],
            "audio_quality": quality,
            "window_index": self.window_index,
        }
        if self.best is None or (
            candidate["gender_confidence"] > self.best["gender_confidence"]
        ):
            self.best = candidate
        return self.best


def _analyze_response_dict(
    contact_id: uuid.UUID,
    quality: str,
    prediction: dict,
    processing_ms: int,
) -> dict:
    """Build the ``/analyze``-shaped JSON body for one window.

    Same field layout as :class:`app.schemas.models.AnalyzeResponse` so a
    client can reuse its ``/analyze`` parsing. ``window_index`` and
    ``best_so_far`` are added by the caller.
    """
    return {
        "contact_id": str(contact_id),
        "gender": {
            "prediction": prediction["gender_prediction"],
            "confidence": round(float(prediction["gender_confidence"]), 4),
        },
        "age_bracket": {
            "prediction": prediction["age_bracket"],
            "confidence": round(float(prediction["age_confidence"]), 4),
        },
        "processing_ms": processing_ms,
        "audio_quality": quality,
    }


@router.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    """Stream audio in, receive a prediction after every rolling window.

    Wire protocol
    -------------
    * Client connects and sends **binary** messages: raw 16 kHz mono
      little-endian float32 PCM, any chunk size (need not align to a
      window).
    * After each ``_WINDOW_S`` of buffered audio the server sends a JSON
      text message: an ``/analyze``-shaped body plus ``window_index`` and
      ``best_so_far``.
    * Client signals end-of-audio by closing the socket (or sending a text
      message ``"close"``). The server flushes any remaining partial
      buffer as one last window, sends a ``{"event": "closing"}`` message,
      and closes.
    * On any resource guard (:data:`_MAX_SESSION_S`, :data:`_MAX_TOTAL_BYTES`,
      :data:`_MAX_WINDOWS`) the server sends ``{"event": "closing",
      "reason": ...}`` and closes with code 1008.

    The handler never raises out to the ASGI layer: a client disconnect is
    caught and turns into clean buffer cleanup, and any unexpected error is
    logged (no raw audio) and the socket closed.
    """
    await websocket.accept()

    contact_id = uuid.uuid4()
    session = _StreamSession(contact_id)
    inferencer = websocket.app.state.inferencer

    logger.info(
        "ws_analyze: session opened",
        extra={"contact_id": str(contact_id), "outcome": "ok"},
    )

    close_code = 1000
    close_reason = "eof"

    try:
        while True:
            # --- session-duration guard --------------------------------
            if session.elapsed_s() > _MAX_SESSION_S:
                close_code, close_reason = 1008, "max_session_duration"
                break

            # --- receive next chunk (with an idle timeout) -----------
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=_RECV_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                close_code, close_reason = 1008, "idle_timeout"
                break

            if message.get("type") == "websocket.disconnect":
                # Client went away. Not an error — fall through to cleanup.
                raise WebSocketDisconnect(message.get("code", 1005))

            text = message.get("text")
            if text is not None:
                if text.strip().lower() == "close":
                    close_code, close_reason = 1000, "client_requested_close"
                    break
                # Ignore any other stray text frame.
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            session.add_chunk(chunk)

            # --- cumulative-bytes guard ------------------------------
            if session.total_bytes > _MAX_TOTAL_BYTES:
                close_code, close_reason = 1008, "max_total_bytes"
                break

            # --- process every complete window we now have -----------
            while session.has_full_window():
                if session.window_index >= _MAX_WINDOWS:
                    close_code, close_reason = 1008, "max_windows"
                    break
                window_pcm = session.take_window()
                await _process_and_emit(websocket, session, inferencer, window_pcm)

            if close_reason == "max_windows":
                break

    except WebSocketDisconnect as exc:
        # Graceful: client closed the connection. Nothing to send; just
        # release the buffer and log one line.
        session.take_remainder()  # drop buffered audio explicitly
        logger.info(
            "ws_analyze: client disconnected",
            extra={
                "contact_id": str(contact_id),
                "processing_ms": int(session.elapsed_s() * 1000),
                "outcome": "ok",
                "error_type": f"WebSocketDisconnect({getattr(exc, 'code', 1005)})",
            },
        )
        return

    except Exception:  # noqa: BLE001 - never propagate to the ASGI layer
        logger.error(
            "ws_analyze: unexpected error, closing stream",
            exc_info=True,
            extra={
                "contact_id": str(contact_id),
                "processing_ms": int(session.elapsed_s() * 1000),
                "outcome": "error_fallback",
            },
        )
        session.take_remainder()
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass
        return

    # --- normal / guard exit: flush any partial window, then close ------
    try:
        remainder = session.take_remainder()
        # Only bother running a final partial window if it carries enough
        # audio for the quality gate to possibly accept it (its 0.5 s
        # floor). Anything shorter would just be an all-unknown message.
        min_flush_bytes = int(0.5 * _STREAM_SAMPLE_RATE) * _BYTES_PER_SAMPLE
        if (
            close_reason in ("eof", "client_requested_close")
            and len(remainder) >= min_flush_bytes
            and session.window_index < _MAX_WINDOWS
        ):
            await _process_and_emit(websocket, session, inferencer, remainder)

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {
                    "event": "closing",
                    "reason": close_reason,
                    "contact_id": str(contact_id),
                    "windows_processed": session.window_index,
                    "best_so_far": session.best,
                }
            )
            await websocket.close(code=close_code)
    except (WebSocketDisconnect, RuntimeError):
        # Client vanished during the final flush — fine, we're closing anyway.
        pass

    logger.info(
        "ws_analyze: session closed",
        extra={
            "contact_id": str(contact_id),
            "processing_ms": int(session.elapsed_s() * 1000),
            "outcome": "degraded" if close_code == 1008 else "ok",
        },
    )


async def _process_and_emit(
    websocket: WebSocket,
    session: _StreamSession,
    inferencer,
    window_pcm: bytes,
) -> None:
    """Run one window through the pipeline and send the client its JSON message.

    Mirrors the ``/analyze`` handler's timing + degrade-never-crash
    behaviour, scoped to a single window:

    * The decode -> VAD -> inference core runs in a worker thread under
      :data:`_WINDOW_BUDGET_S`; on overrun the window's message is
      all-unknown / insufficient.
    * A decode failure on the window is reported as ``insufficient``, not
      an error.
    * ``processing_ms`` is the wall-clock for *this window* end to end.
    """
    started = time.perf_counter()
    idx = session.window_index

    try:
        try:
            quality, prediction = await asyncio.wait_for(
                asyncio.to_thread(_run_window_pipeline, window_pcm, inferencer),
                timeout=_WINDOW_BUDGET_S,
            )
            outcome = "ok" if quality == AudioQuality.GOOD.value else "degraded"
        except asyncio.TimeoutError:
            quality, prediction = AudioQuality.INSUFFICIENT.value, _unknown_payload()
            outcome = "degraded"
        except AudioDecodeError:
            quality, prediction = AudioQuality.INSUFFICIENT.value, _unknown_payload()
            outcome = "degraded"
    except Exception:  # noqa: BLE001 - one bad window must not kill the stream
        quality, prediction = AudioQuality.INSUFFICIENT.value, _unknown_payload()
        outcome = "error_fallback"
        logger.error(
            "ws_analyze: window pipeline crashed, emitting all-unknown",
            exc_info=True,
            extra={"contact_id": str(session.contact_id), "outcome": "error_fallback"},
        )

    processing_ms = int((time.perf_counter() - started) * 1000)

    best = session.observe(quality, prediction)

    payload = _analyze_response_dict(session.contact_id, quality, prediction, processing_ms)
    payload["window_index"] = idx
    payload["best_so_far"] = best

    # One structured line per window. Byte/sample COUNTS only — never the
    # bytes, a filename, or a transcript (same PII rule as /analyze).
    logger.info(
        "ws_analyze: window %d processed",
        idx,
        extra={
            "contact_id": str(session.contact_id),
            "audio_quality": quality,
            "processing_ms": processing_ms,
            "inference_ms": round(float(prediction.get("inference_ms", 0.0)), 1),
            "gender_prediction": prediction["gender_prediction"],
            "gender_confidence": round(float(prediction["gender_confidence"]), 3),
            "age_prediction": prediction["age_bracket"],
            "age_confidence": round(float(prediction["age_confidence"]), 3),
            "outcome": outcome,
        },
    )

    session.window_index += 1

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json(payload)
