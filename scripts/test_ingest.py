#!/usr/bin/env python3
"""Manual smoke test for app/audio/ingest.py.

Runs a real audio file through ``normalize_audio()`` -> ``run_vad()`` and
prints the VAD stats, then repeats the exercise against a synthetic silent
buffer and an empty buffer to confirm the graceful-degradation paths.

Standalone-ish: it imports from the ``app`` package, so run it from the repo
root (or with the repo root on PYTHONPATH). Requires the ``ffmpeg`` binary
plus ``numpy`` and ``webrtcvad``.

Usage:
    python scripts/test_ingest.py path/to/sample.wav
    python scripts/test_ingest.py --file path/to/sample.wav
"""

from __future__ import annotations

import argparse
import struct
import sys
import wave
from io import BytesIO
from pathlib import Path

# Allow running as `python scripts/test_ingest.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.ingest import (  # noqa: E402
    AudioDecodeError,
    normalize_audio,
    run_vad,
)


def _print_stats(label: str, stats: dict) -> None:
    print(f"  {label}")
    print(f"    total_duration_s   : {stats['total_duration_s']:.3f}")
    print(f"    speech_duration_s  : {stats['speech_duration_s']:.3f}")
    print(f"    speech_ratio       : {stats['speech_ratio']:.3f}")
    print(f"    num_speech_segments: {stats['num_speech_segments']}")


def _silent_wav_bytes(duration_s: float = 2.0, sample_rate: int = 8000) -> bytes:
    """Build an in-memory mono 16-bit WAV of pure silence."""
    n = int(duration_s * sample_rate)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    return buf.getvalue()


def test_real_file(wav_path: Path) -> int:
    print(f"[1] real file: {wav_path}")
    if not wav_path.is_file():
        print(f"    SKIP: not a file: {wav_path}")
        return 1
    raw = wav_path.read_bytes()
    try:
        waveform = normalize_audio(raw)
    except AudioDecodeError as exc:
        print(f"    FAIL: normalize_audio raised: {exc}")
        return 1
    print(
        f"    normalized: {waveform.shape[0]} samples, dtype={waveform.dtype}, "
        f"range=[{waveform.min():.3f}, {waveform.max():.3f}]"
    )
    stats = run_vad(waveform)
    _print_stats("VAD stats:", stats)
    ok = waveform.dtype.name == "float32" and stats["speech_ratio"] >= 0.0
    print(f"    {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def test_silent_buffer() -> int:
    print("[2] synthetic silent buffer (2 s of zeros, 8 kHz mono WAV)")
    raw = _silent_wav_bytes()
    try:
        waveform = normalize_audio(raw)
    except AudioDecodeError as exc:
        print(f"    FAIL: normalize_audio raised on valid silent WAV: {exc}")
        return 1
    stats = run_vad(waveform)
    _print_stats("VAD stats:", stats)
    ok = stats["speech_ratio"] == 0.0 and stats["num_speech_segments"] == 0
    print(f"    expected speech_ratio == 0.0 -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def test_short_buffer() -> int:
    print("[3] very short buffer (0.1 s of zeros, sub-0.5 s)")
    raw = _silent_wav_bytes(duration_s=0.1)
    waveform = normalize_audio(raw)
    stats = run_vad(waveform)
    _print_stats("VAD stats:", stats)
    ok = stats["speech_ratio"] == 0.0
    print(f"    expected graceful speech_ratio == 0.0 -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def test_empty_buffer() -> int:
    print("[4] empty buffer (0 bytes)")
    try:
        normalize_audio(b"")
    except AudioDecodeError as exc:
        print(f"    OK: raised AudioDecodeError as expected: {exc}")
        return 0
    print("    FAIL: expected AudioDecodeError, got no exception")
    return 1


def test_garbage_buffer() -> int:
    print("[5] corrupt buffer (random non-audio bytes)")
    try:
        normalize_audio(b"not audio, just text pretending to be a file" * 10)
    except AudioDecodeError as exc:
        print(f"    OK: raised AudioDecodeError as expected: {exc}")
        return 0
    print("    FAIL: expected AudioDecodeError, got no exception")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="?", help="Path to a sample WAV.")
    parser.add_argument("--file", "-f", type=Path, dest="file", help="Alt to positional.")
    args = parser.parse_args()

    wav_path = args.file or args.wav

    failures = 0
    if wav_path is not None:
        failures += test_real_file(wav_path)
    else:
        print("[1] real file: SKIP (no path given; pass one positionally or with -f)")
    print()
    failures += test_silent_buffer()
    print()
    failures += test_short_buffer()
    print()
    failures += test_empty_buffer()
    print()
    failures += test_garbage_buffer()

    print()
    print("=" * 50)
    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    print("=" * 50)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
