"""Audio quality gating for the inference pipeline.

Stub only. This module decides whether a decoded clip is usable and maps
its characteristics onto the ``AudioQuality`` rating in the API contract.

Planned signals:
- Voiced-speech duration via ``webrtcvad`` (frame-level VAD).
- SNR / noise-floor estimate.
- Clipping ratio and overall level.
- Total duration vs. a minimum threshold.
"""

from dataclasses import dataclass

from app.audio.ingest import DecodedAudio
from app.schemas.models import AudioQuality

MIN_VOICED_SECONDS = 1.5


@dataclass
class QualityReport:
    """Detailed quality metrics behind the coarse ``AudioQuality`` rating.

    Attributes:
        rating: The bucket surfaced in the API response.
        voiced_seconds: Total detected voiced-speech duration.
        snr_db: Estimated signal-to-noise ratio in dB (``None`` if not computed).
        clipping_ratio: Fraction of samples at or near full scale.
    """

    rating: AudioQuality
    voiced_seconds: float
    snr_db: "float | None"
    clipping_ratio: float


def assess(audio: DecodedAudio) -> QualityReport:
    """Score a decoded clip and assign an :class:`AudioQuality` rating.

    Args:
        audio: Canonical decoded audio from ``app.audio.ingest``.

    Returns:
        A :class:`QualityReport`. A ``rating`` of
        :attr:`AudioQuality.INSUFFICIENT` signals the caller to skip
        inference and return ``unknown`` predictions.
    """
    raise NotImplementedError


def voiced_seconds(audio: DecodedAudio, *, aggressiveness: int = 2) -> float:
    """Return total voiced-speech duration in seconds using WebRTC VAD.

    Args:
        audio: Canonical decoded audio (must be 8/16/32/48 kHz mono for VAD).
        aggressiveness: WebRTC VAD aggressiveness, 0 (least) to 3 (most).

    Returns:
        Seconds of audio classified as speech.
    """
    raise NotImplementedError
