"""Audio ingestion: decode, resample, and normalize incoming clips.

This module turns arbitrary uploaded/streamed audio into the canonical form
the quality checks and the model expect: **16 kHz, mono, float32 PCM in
[-1, 1]**. It also runs a lightweight voice-activity pass so callers can
decide whether a clip carries enough speech to be worth a model forward
pass.

Design rationale
----------------
* **Everything stays in memory.** Call-center audio is sensitive (PII, and
  often recorded consent constraints). We never touch the filesystem: bytes
  go into ``ffmpeg`` over stdin and normalized PCM comes back over stdout.
  No ``tempfile``, no scratch ``.wav`` on disk, nothing for a forensic tool
  or a misconfigured backup job to pick up later.
* **ffmpeg does the container/codec work.** Incoming audio is wildly
  heterogeneous: browser ``webm/opus``, phone ``m4a``, and especially
  telephony ``G.711`` (8 kHz mu-law / a-law) from SIP trunks. ffmpeg
  already handles every one of these and its resampler (``soxr``) is
  better than anything worth hand-rolling. We shell out with pipes rather
  than binding libav so the dependency is a single well-known binary.
* **float32 in [-1, 1]** is what ``transformers`` feature extractors and
  ``torch`` want, so we ask ffmpeg for ``f32le`` directly and skip an
  int16 -> float conversion step.
* **VAD uses webrtcvad.** See :func:`run_vad` for the tradeoff note.

Requires the ``ffmpeg`` binary on ``PATH`` and the ``webrtcvad`` and
``numpy`` packages.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

TARGET_SAMPLE_RATE = 16_000

# ffmpeg emits little-endian 32-bit float PCM for this codec/format pair.
_FFMPEG_OUTPUT_DTYPE = np.float32

# webrtcvad only accepts 10, 20, or 30 ms frames. 30 ms gives the VAD the
# most context per decision and keeps the Python frame loop cheap.
_VAD_FRAME_MS = 30


class AudioDecodeError(Exception):
    """Raised when incoming audio bytes cannot be decoded to PCM.

    Covers an empty payload, a corrupt/truncated container, an unknown
    codec, or a non-zero exit from ``ffmpeg``. The message carries the tail
    of ffmpeg's stderr so the caller can log something actionable instead
    of a bare "conversion failed".
    """


@dataclass
class DecodedAudio:
    """Canonical decoded audio ready for quality gating and inference.

    Attributes:
        samples: Mono float32 waveform in [-1, 1], sampled at ``sample_rate``.
        sample_rate: Sample rate in Hz (equals :data:`TARGET_SAMPLE_RATE`).
        duration_s: Convenience duration in seconds.
    """

    samples: np.ndarray
    sample_rate: int
    duration_s: float


def normalize_audio(raw_bytes: bytes, input_format: str | None = None) -> np.ndarray:
    """Decode arbitrary audio bytes to a 16 kHz mono float32 waveform.

    The input may be any container/codec ffmpeg understands: ``wav``,
    ``mp3``, ``m4a``, ``ogg/opus``, ``webm``, or raw telephony formats such
    as 8 kHz mu-law / a-law (G.711). Decoding, downmixing to mono, and
    resampling to 16 kHz all happen inside ffmpeg; the bytes are streamed in
    over stdin and the normalized PCM is streamed back over stdout, so
    **nothing is ever written to disk**.

    Why in-memory only: call recordings routinely contain PII and may carry
    contractual limits on where the audio can live. Piping keeps the audio
    in this process's memory for exactly as long as the request needs it.

    Args:
        raw_bytes: Raw bytes of an audio file or raw PCM stream.
        input_format: Optional ffmpeg demuxer/format hint (ffmpeg ``-f``
            value), e.g. ``"mp3"``, ``"wav"``, ``"mulaw"``, ``"alaw"``.
            Usually unnecessary because ffmpeg sniffs the container, but
            **required for headerless telephony captures** (raw mu-law/a-law
            has no header to sniff). When ``"mulaw"``/``"alaw"`` is given we
            also tell ffmpeg the stream is 8 kHz mono, the G.711 norm.

    Returns:
        A 1-D ``np.ndarray`` of dtype ``float32`` in [-1, 1], sampled at
        16 kHz, ready to hand to a feature extractor or model.

    Raises:
        AudioDecodeError: If ``raw_bytes`` is empty, if the ``ffmpeg``
            binary is missing, if ffmpeg exits non-zero (corrupt/unknown
            input), or if ffmpeg produces no samples.
    """
    if not raw_bytes:
        raise AudioDecodeError("input audio is empty (0 bytes)")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise AudioDecodeError(
            "the 'ffmpeg' binary was not found on PATH; it is required for "
            "audio decoding"
        )

    cmd: list[str] = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]

    # Input format hint. Raw G.711 has no header, so also pin its canonical
    # sample rate / channel count; ffmpeg cannot infer them.
    fmt = input_format.lower() if input_format else None
    if fmt in {"mulaw", "alaw"}:
        cmd += ["-f", fmt, "-ar", "8000", "-ac", "1"]
    elif fmt:
        cmd += ["-f", fmt]

    cmd += ["-i", "pipe:0"]

    # Output: 16 kHz, mono, little-endian float32 PCM, raw (no WAV header) to
    # stdout. `-vn` drops any incidental video/album-art stream.
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=raw_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:  # binary vanished between which() and exec, etc.
        raise AudioDecodeError(f"failed to launch ffmpeg: {exc}") from exc

    if proc.returncode != 0:
        stderr_tail = _tail(proc.stderr.decode("utf-8", "replace"))
        raise AudioDecodeError(
            f"ffmpeg failed to decode input (exit {proc.returncode}): {stderr_tail}"
        )

    waveform = np.frombuffer(proc.stdout, dtype=_FFMPEG_OUTPUT_DTYPE)
    if waveform.size == 0:
        stderr_tail = _tail(proc.stderr.decode("utf-8", "replace"))
        raise AudioDecodeError(
            "ffmpeg produced no audio samples; the input may be silent-only "
            f"metadata or a non-audio file. ffmpeg said: {stderr_tail or '(nothing)'}"
        )

    # `np.frombuffer` returns a read-only view onto the subprocess buffer;
    # copy so downstream code can normalize / write in place.
    waveform = np.array(waveform, dtype=np.float32, copy=True)

    # ffmpeg's float output is already in [-1, 1] for normal input, but a
    # decoded clip can contain out-of-range samples (inter-sample peaks from
    # lossy codecs). Clip rather than rescale so we don't change loudness.
    np.clip(waveform, -1.0, 1.0, out=waveform)

    return waveform


def run_vad(waveform: np.ndarray, sample_rate: int = 16000) -> dict:
    """Estimate how much of ``waveform`` is speech using WebRTC VAD.

    Runs webrtcvad frame-by-frame (30 ms frames) and aggregates the
    speech/non-speech decisions into clip-level stats.

    Why webrtcvad and not silero-vad
    --------------------------------
    * **webrtcvad** (chosen): a tiny C extension, zero model download, no
      torch dependency in the hot path, and microsecond-per-frame cost. It
      is a GMM energy/spectral detector, so it is weaker in low-SNR noise
      and can trip on loud non-speech (music, hold tones). For our use --
      a coarse "is there enough speech here to bother running the model?"
      gate on already telephone-bandlimited audio -- that accuracy is
      fine, and the operational simplicity (no weights to ship in the
      Docker image, deterministic latency) wins.
    * **silero-vad** would be more robust to noise and non-speech, but it
      pulls in torch + an ONNX/JIT model, adds a real per-call inference
      cost, and is overkill for a gate. If downstream accuracy shows the
      VAD is admitting too much noise, swapping the implementation here is
      localized -- the returned dict is the contract, not the detector.

    Args:
        waveform: Mono float32 waveform in [-1, 1] (as returned by
            :func:`normalize_audio`).
        sample_rate: Sample rate of ``waveform`` in Hz. webrtcvad only
            supports 8000, 16000, 32000, or 48000; other values raise
            ``ValueError``.

    Returns:
        A dict with:

        * ``total_duration_s`` (float): length of the clip in seconds.
        * ``speech_duration_s`` (float): seconds of frames marked speech.
        * ``speech_ratio`` (float): ``speech_duration_s / total_duration_s``
          clamped to [0.0, 1.0]; ``0.0`` for empty or all-silence input.
        * ``num_speech_segments`` (int): count of maximal runs of
          consecutive speech frames (a rough proxy for number of
          utterances).

    Short clips (< 0.5 s) and pure silence are handled without raising:
    they simply yield ``speech_ratio: 0.0`` (a sub-frame clip has no full
    30 ms frame to classify).
    """
    import webrtcvad

    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError(
            f"webrtcvad supports 8/16/32/48 kHz only, got {sample_rate} Hz; "
            "resample with normalize_audio() first"
        )

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    total_samples = waveform.shape[0]
    total_duration_s = total_samples / sample_rate

    empty_result = {
        "total_duration_s": total_duration_s,
        "speech_duration_s": 0.0,
        "speech_ratio": 0.0,
        "num_speech_segments": 0,
    }

    frame_samples = int(sample_rate * _VAD_FRAME_MS / 1000)
    if total_samples < frame_samples:
        # Clip shorter than a single VAD frame (e.g. < 30 ms, and in
        # particular anything << 0.5 s of near-silence). Nothing to classify.
        return empty_result

    # webrtcvad wants 16-bit little-endian PCM. Convert once.
    pcm16 = _float_to_pcm16(waveform)

    vad = webrtcvad.Vad(2)  # aggressiveness 0..3; 2 balances FA vs miss.
    bytes_per_frame = frame_samples * 2  # int16 == 2 bytes/sample

    speech_frames = 0
    total_frames = 0
    num_segments = 0
    prev_was_speech = False

    for start in range(0, len(pcm16) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm16[start : start + bytes_per_frame]
        total_frames += 1
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception:  # pragma: no cover - defensive; malformed frame
            is_speech = False
        if is_speech:
            speech_frames += 1
            if not prev_was_speech:
                num_segments += 1
        prev_was_speech = is_speech

    if total_frames == 0:
        return empty_result

    frame_duration_s = frame_samples / sample_rate
    speech_duration_s = speech_frames * frame_duration_s
    speech_ratio = speech_frames / total_frames
    speech_ratio = float(min(1.0, max(0.0, speech_ratio)))

    return {
        "total_duration_s": total_duration_s,
        "speech_duration_s": speech_duration_s,
        "speech_ratio": speech_ratio,
        "num_speech_segments": num_segments,
    }


def decode_to_canonical(
    raw_bytes: bytes, input_format: str | None = None
) -> DecodedAudio:
    """Convenience wrapper: :func:`normalize_audio` plus duration bookkeeping.

    Returns the same waveform boxed in a :class:`DecodedAudio` so callers
    that want the sample rate and duration alongside the samples (quality
    gating, the inference layer) don't have to recompute them.

    Args:
        raw_bytes: Raw audio bytes.
        input_format: Optional ffmpeg format hint; see :func:`normalize_audio`.

    Returns:
        A :class:`DecodedAudio` at :data:`TARGET_SAMPLE_RATE`.

    Raises:
        AudioDecodeError: Propagated from :func:`normalize_audio`.
    """
    samples = normalize_audio(raw_bytes, input_format)
    return DecodedAudio(
        samples=samples,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_s=samples.shape[0] / TARGET_SAMPLE_RATE,
    )


def _float_to_pcm16(waveform: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] waveform to little-endian int16 PCM bytes."""
    clipped = np.clip(waveform, -1.0, 1.0)
    # Scale by 32767 (not 32768) so +1.0 maps exactly to int16 max and we
    # never wrap to -32768.
    return (clipped * 32767.0).astype("<i2").tobytes()


def _tail(text: str, max_chars: int = 500) -> str:
    """Return the last ``max_chars`` of ``text``, single-lined, for error messages."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return "..." + collapsed[-max_chars:]
