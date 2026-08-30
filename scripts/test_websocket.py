#!/usr/bin/env python3
"""Manual test client for the streaming endpoint ``WS /ws/analyze``.

Simulates a live call feed:

  1. Connects to ``ws://localhost:8000/ws/analyze``.
  2. Decodes ``sample.wav`` once to the wire format the endpoint expects
     (16 kHz mono little-endian float32 PCM) via ``normalize_audio`` — the
     same ingest code the service uses. Nothing is written to disk.
  3. Sends the PCM in small chunks (~0.5 s each) with a short sleep between
     sends to mimic real-time arrival.
  4. Prints every prediction message the server pushes back, as it arrives.
  5. On EOF, sends a ``"close"`` text frame, prints the final ``closing``
     message, and exits.

Run the service first (``uvicorn app.main:app``), then::

    python scripts/test_websocket.py
    python scripts/test_websocket.py --file path/to/other.wav
    python scripts/test_websocket.py --url ws://localhost:8000/ws/analyze

Requires the ``websockets`` package (a dependency of ``uvicorn[standard]``)
and the ``ffmpeg`` binary for decoding the input file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Allow running as `python scripts/test_websocket.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402

from app.audio.ingest import normalize_audio  # noqa: E402

_SAMPLE_RATE = 16_000
_BYTES_PER_SAMPLE = 4  # float32
_DEFAULT_URL = "ws://localhost:8000/ws/analyze"
_DEFAULT_WAV = Path(__file__).resolve().parent.parent / "sample.wav"

# ~0.5 s of audio per send, with a matching real-time-ish delay between
# sends so the server fills its ~2 s windows over "wall time" the way a
# live call leg would.
_CHUNK_S = 0.5
_CHUNK_BYTES = int(_CHUNK_S * _SAMPLE_RATE) * _BYTES_PER_SAMPLE
_SEND_DELAY_S = 0.4


def _load_pcm(wav_path: Path) -> bytes:
    """Decode ``wav_path`` to 16 kHz mono float32 little-endian PCM bytes.

    Uses the service's own ``normalize_audio`` (ffmpeg over a pipe, no
    temp file), then converts the float32 numpy array back to raw bytes —
    which is exactly the wire format ``/ws/analyze`` expects.
    """
    waveform = normalize_audio(wav_path.read_bytes())
    return waveform.astype("<f4").tobytes()


def _fmt_prediction(msg: dict) -> str:
    g = msg.get("gender", {})
    a = msg.get("age_bracket", {})
    best = msg.get("best_so_far") or {}
    return (
        f"  window {msg.get('window_index'):>2} | "
        f"quality={msg.get('audio_quality'):<12} | "
        f"gender={g.get('prediction'):<7}({g.get('confidence'):.3f}) | "
        f"age={a.get('prediction'):<7}({a.get('confidence'):.3f}) | "
        f"{msg.get('processing_ms')} ms | "
        f"best: gender={best.get('gender_prediction')}"
        f"({best.get('gender_confidence', 0.0):.3f}) "
        f"age={best.get('age_bracket')} [win {best.get('window_index')}]"
    )


async def stream_file(url: str, wav_path: Path) -> int:
    if not wav_path.is_file():
        print(f"ERROR: not a file: {wav_path}")
        return 1

    print(f"decoding {wav_path.name} to 16 kHz mono float32 PCM ...")
    pcm = _load_pcm(wav_path)
    total_s = len(pcm) / (_SAMPLE_RATE * _BYTES_PER_SAMPLE)
    print(
        f"  {len(pcm)} bytes ({total_s:.2f} s); sending ~{_CHUNK_S}s chunks "
        f"with {_SEND_DELAY_S}s between sends\n"
    )

    print(f"connecting to {url} ...")
    try:
        async with websockets.connect(url, max_size=None) as ws:
            print("connected. streaming...\n")

            async def sender() -> None:
                for start in range(0, len(pcm), _CHUNK_BYTES):
                    await ws.send(pcm[start : start + _CHUNK_BYTES])
                    await asyncio.sleep(_SEND_DELAY_S)
                # Signal end-of-audio; server flushes the partial window.
                await ws.send("close")

            async def receiver() -> None:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        print(f"  <non-JSON frame: {raw!r}>")
                        continue
                    if msg.get("event") == "closing":
                        print(
                            f"\nserver closing: reason={msg.get('reason')} "
                            f"windows_processed={msg.get('windows_processed')}"
                        )
                        best = msg.get("best_so_far")
                        if best:
                            print(
                                f"  final best: gender={best.get('gender_prediction')}"
                                f"({best.get('gender_confidence', 0.0):.3f}) "
                                f"age={best.get('age_bracket')}"
                                f"({best.get('age_confidence', 0.0):.3f}) "
                                f"from window {best.get('window_index')}"
                            )
                        return
                    print(_fmt_prediction(msg))

            t0 = time.perf_counter()
            send_task = asyncio.create_task(sender())
            await receiver()
            await send_task
            print(f"\ndone in {time.perf_counter() - t0:.1f} s")
    except OSError as exc:
        print(f"ERROR: could not connect to {url}: {exc}")
        print("Is the service running?  uvicorn app.main:app")
        return 1
    except websockets.exceptions.WebSocketException as exc:
        print(f"ERROR: websocket failure: {exc}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=_DEFAULT_URL, help=f"WebSocket URL (default: {_DEFAULT_URL})"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=_DEFAULT_WAV,
        help=f"WAV to stream (default: {_DEFAULT_WAV.name})",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(stream_file(args.url, args.file)))


if __name__ == "__main__":
    main()
