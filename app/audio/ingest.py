"""Audio ingestion: decode, resample, and normalize incoming clips.

Stub only. This module turns arbitrary uploaded/streamed audio into a
canonical form the quality checks and model expect (mono, fixed sample
rate, float32 PCM in [-1, 1]).

Planned dependencies: ``ffmpeg-python`` for container/codec decoding,
``torchaudio`` / ``numpy`` for resampling and tensor conversion.
"""

from dataclasses import dataclass

import numpy as np

TARGET_SAMPLE_RATE = 16_000


@dataclass
class DecodedAudio:
    """Canonical decoded audio ready for quality gating and inference.

    Attributes:
        samples: Mono float32 waveform in [-1, 1].
        sample_rate: Sample rate in Hz (expected to equal TARGET_SAMPLE_RATE).
        duration_s: Convenience duration in seconds.
    """

    samples: "np.ndarray"
    sample_rate: int
    duration_s: float


def decode_bytes(raw: bytes, *, target_sr: int = TARGET_SAMPLE_RATE) -> DecodedAudio:
    """Decode an in-memory audio file to canonical mono float32 PCM.

    Args:
        raw: Raw bytes of an audio file in any ffmpeg-supported container
            (wav, mp3, ogg/opus, m4a, ...).
        target_sr: Desired output sample rate in Hz.

    Returns:
        A :class:`DecodedAudio` resampled to ``target_sr`` and downmixed to
        mono.

    Raises:
        ValueError: If the bytes cannot be decoded as audio.
    """
    raise NotImplementedError


def resample(samples: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    """Resample a mono float32 waveform.

    Args:
        samples: Mono float32 waveform.
        orig_sr: Current sample rate in Hz.
        target_sr: Desired sample rate in Hz.

    Returns:
        The resampled waveform; returned unchanged when the rates match.
    """
    raise NotImplementedError


def to_mono(samples: "np.ndarray") -> "np.ndarray":
    """Downmix a possibly multi-channel waveform to mono by averaging channels."""
    raise NotImplementedError
