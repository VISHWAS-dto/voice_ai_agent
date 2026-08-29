#!/usr/bin/env python3
"""Manual smoke test for app/inference/model.py.

Loads :class:`AttributeInferencer` once, then:
  1. Runs ``predict()`` on a real WAV file and prints the full result dict
     plus the total wall-clock time (load excluded, but reported too).
  2. Runs ``predict()`` on a synthetic silent buffer to confirm it does not
     crash and returns low-confidence / ``unknown`` output.

Imports from the ``app`` package, so run it from the repo root (or with the
repo root on PYTHONPATH). Requires ``torch``, ``transformers``, ``numpy``
and, for decoding the real file, the ``ffmpeg`` binary.

Usage:
    python scripts/test_inference.py path/to/sample.wav
    python scripts/test_inference.py --file path/to/sample.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as `python scripts/test_inference.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.ingest import AudioDecodeError, normalize_audio  # noqa: E402
from app.inference.model import AttributeInferencer  # noqa: E402

TARGET_SR = 16_000
EXPECTED_KEYS = {
    "gender_prediction",
    "gender_confidence",
    "age_bracket",
    "age_confidence",
    "inference_ms",
}


def _print_result(result: dict) -> None:
    print("    result dict:")
    for key in (
        "gender_prediction",
        "gender_confidence",
        "age_bracket",
        "age_confidence",
        "inference_ms",
    ):
        print(f"      {key:18s}: {result.get(key)!r}")


def _check_shape(result: dict) -> bool:
    ok = set(result) == EXPECTED_KEYS
    if not ok:
        print(f"    FAIL: unexpected keys: {sorted(result)}")
        return False
    checks = {
        "gender_prediction in {male,female,unknown}": result["gender_prediction"]
        in {"male", "female", "unknown"},
        "age_bracket in {18-30,31-45,46-60,60+,unknown}": result["age_bracket"]
        in {"18-30", "31-45", "46-60", "60+", "unknown"},
        "0 <= gender_confidence <= 1": 0.0 <= result["gender_confidence"] <= 1.0,
        "0 <= age_confidence <= 1": 0.0 <= result["age_confidence"] <= 1.0,
        "inference_ms >= 0": result["inference_ms"] >= 0.0,
    }
    for label, passed in checks.items():
        if not passed:
            print(f"    FAIL: {label}")
            ok = False
    return ok


def _silent_waveform(duration_s: float = 3.0) -> np.ndarray:
    """A 16 kHz mono float32 buffer of pure digital silence."""
    return np.zeros(int(duration_s * TARGET_SR), dtype=np.float32)


def test_real_file(inferencer: AttributeInferencer, wav_path: Path) -> int:
    print(f"[1] real file: {wav_path}")
    if not wav_path.is_file():
        print(f"    SKIP: not a file: {wav_path}")
        return 0
    try:
        waveform = normalize_audio(wav_path.read_bytes())
    except AudioDecodeError as exc:
        print(f"    FAIL: could not decode {wav_path}: {exc}")
        return 1

    dur = waveform.shape[0] / TARGET_SR
    print(f"    decoded: {waveform.shape[0]} samples ({dur:.2f} s), dtype={waveform.dtype}")

    t0 = time.perf_counter()
    result = inferencer.predict(waveform, TARGET_SR)
    total_ms = (time.perf_counter() - t0) * 1000.0

    _print_result(result)
    print(f"    total predict() wall time: {total_ms:.1f} ms")
    print(f"    (of which model forward: {result['inference_ms']:.1f} ms)")

    ok = _check_shape(result)
    # On a real speech clip we expect the model to actually commit to
    # something rather than degrade to all-unknown.
    if result["gender_prediction"] == "unknown" and result["age_bracket"] == "unknown":
        print("    WARN: both predictions are 'unknown' on a real clip "
              "(model may have failed or clip is unusable)")
    print(f"    {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def test_silent_buffer(inferencer: AttributeInferencer) -> int:
    print("[2] synthetic silent buffer (3 s of zeros, 16 kHz mono float32)")
    waveform = _silent_waveform()

    try:
        t0 = time.perf_counter()
        result = inferencer.predict(waveform, TARGET_SR)
        total_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001 - this is exactly what must NOT happen
        print(f"    FAIL: predict() raised on silent buffer: {type(exc).__name__}: {exc}")
        return 1

    _print_result(result)
    print(f"    total predict() wall time: {total_ms:.1f} ms")

    ok = _check_shape(result)
    # Silence carries no speaker: we want low-confidence and/or unknown,
    # never a confident age/gender call. The gender softmax on silence is
    # near-uniform (top class well under 0.7); age_confidence is a capped
    # heuristic that by construction never exceeds ~0.65.
    sensible = (
        result["gender_confidence"] <= 0.7
        and result["age_confidence"] <= 0.66
    )
    if not sensible:
        print(
            "    FAIL: silence produced a high-confidence prediction "
            f"(gender={result['gender_confidence']:.3f}, "
            f"age={result['age_confidence']:.3f})"
        )
        ok = False
    else:
        print("    OK: no crash, confidences are not spuriously high")
    print(f"    {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="?", help="Path to a sample WAV.")
    parser.add_argument("--file", "-f", type=Path, dest="file", help="Alt to positional.")
    args = parser.parse_args()

    wav_path = args.file or args.wav

    print("loading AttributeInferencer (one-time model load)...")
    t0 = time.perf_counter()
    inferencer = AttributeInferencer()
    print(f"  loaded in {time.perf_counter() - t0:.1f} s\n")

    failures = 0
    if wav_path is not None:
        failures += test_real_file(inferencer, wav_path)
    else:
        print("[1] real file: SKIP (no path given; pass one positionally or with -f)")
    print()
    failures += test_silent_buffer(inferencer)

    print()
    print("=" * 50)
    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    print("=" * 50)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
