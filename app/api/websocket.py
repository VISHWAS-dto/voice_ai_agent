"""WebSocket endpoint for streaming voice attribute inference.

Stub only. This module will support near-real-time analysis over a live
call leg, as an alternative to the batch ``POST /analyze`` route.

Planned protocol (subject to change):
- Client connects to ``/ws/analyze`` and sends binary audio frames
  (fixed sample rate / frame size, negotiated in an opening JSON message).
- Server buffers frames, runs VAD + quality checks incrementally, and
  emits partial ``AnalyzeResponse``-shaped JSON updates as confidence
  stabilizes.
- Server sends a final message and closes when the utterance ends or a
  duration cap is hit.
"""

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    """Stream audio frames in, receive incremental attribute predictions.

    Args:
        websocket: The accepted client connection. Expects an opening JSON
            handshake describing the audio format, followed by binary
            PCM/opus frames.

    Emits:
        JSON messages shaped like :class:`app.schemas.models.AnalyzeResponse`,
        with a terminal message before close.
    """
    raise NotImplementedError
