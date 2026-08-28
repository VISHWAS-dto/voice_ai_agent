#!/usr/bin/env python3
"""Sanity-check script for the audeering age/gender wav2vec2 model.

Loads "audeering/wav2vec2-large-robust-24-ft-age-gender", runs one forward
pass on a WAV file, and dumps the raw model outputs so we can see exactly
what comes back before writing any mapping logic.

Standalone: no dependency on the `app` package. Requires
`transformers`, `torch`, `torchaudio`, `numpy`.

Usage:
    python scripts/test_model_load.py path/to/sample.wav
    python scripts/test_model_load.py --file path/to/sample.wav
    python scripts/test_model_load.py path/to/sample.wav --cache-dir ./.model_cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
TARGET_SR = 16_000
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"


# ---------------------------------------------------------------------------
# Model definition
#
# The audeering checkpoint ships a custom architecture that is NOT part of
# `transformers`. The two classes below are copied verbatim from the model
# card so `from_pretrained` has something to load the weights into.
# ---------------------------------------------------------------------------
try:
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor
except ImportError as exc:  # pragma: no cover - environment problem
    print(
        "ERROR: could not import transformers. Install deps first:\n"
        "    pip install transformers torch torchaudio numpy\n"
        f"(import error: {exc})",
        file=sys.stderr,
    )
    raise SystemExit(1)


class ModelHead(nn.Module):
    """Regression/classification head on top of the pooled hidden states."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """wav2vec2 backbone with an age regression head and a 3-class gender head.

    forward() returns: (hidden_states, logits_age, logits_gender)
      - hidden_states: pooled embeddings, shape [batch, hidden_size]
      - logits_age:    regression value in ~[0, 1] mapping to 0-100 years,
                       shape [batch, 1]
      - logits_gender: softmax probabilities over (child, female, male),
                       shape [batch, 3]
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def load_model(cache_dir: Path):
    """Download (if needed) and load the processor + model.

    Raises SystemExit with a clear message on any failure.
    """
    print(f"[load] model     : {MODEL_NAME}")
    print(f"[load] cache dir  : {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        processor = Wav2Vec2Processor.from_pretrained(
            MODEL_NAME, cache_dir=str(cache_dir)
        )
        model = AgeGenderModel.from_pretrained(MODEL_NAME, cache_dir=str(cache_dir))
        model.eval()
    except OSError as exc:
        raise SystemExit(
            "ERROR: model download / load failed.\n"
            "  Likely causes: no network, HF Hub outage, bad cache dir, or\n"
            "  insufficient disk space (~1.3 GB needed).\n"
            f"  Underlying error: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - surface anything else clearly
        raise SystemExit(
            f"ERROR: unexpected failure while loading the model: {type(exc).__name__}: {exc}"
        )
    print("[load] ok")
    return processor, model


def load_audio(wav_path: Path) -> np.ndarray:
    """Load a WAV file, downmix to mono, resample to 16 kHz.

    Returns a 1-D float32 numpy array. Raises SystemExit with a clear
    message on any failure.
    """
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: torchaudio not installed. `pip install torchaudio` ({exc})"
        )

    if not wav_path.exists():
        raise SystemExit(f"ERROR: audio file not found: {wav_path}")
    if not wav_path.is_file():
        raise SystemExit(f"ERROR: not a file: {wav_path}")

    try:
        waveform, sr = torchaudio.load(str(wav_path))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: could not decode audio file {wav_path}.\n"
            "  Is it a valid WAV? Try converting: ffmpeg -i in.ext out.wav\n"
            f"  Underlying error: {type(exc).__name__}: {exc}"
        )

    n_channels, n_frames = waveform.shape
    print(f"[audio] path      : {wav_path}")
    print(f"[audio] loaded    : {n_channels} ch, {n_frames} frames, {sr} Hz")

    # Downmix to mono.
    if n_channels > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print("[audio] downmixed : -> 1 ch (channel mean)")

    # Resample to 16 kHz if needed.
    if sr != TARGET_SR:
        try:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"ERROR: resample from {sr} Hz to {TARGET_SR} Hz failed: "
                f"{type(exc).__name__}: {exc}"
            )
        print(f"[audio] resampled : {sr} Hz -> {TARGET_SR} Hz")
    else:
        print(f"[audio] sample rate already {TARGET_SR} Hz, no resample")

    signal = waveform.squeeze(0).to(torch.float32).numpy()
    dur_s = signal.shape[0] / TARGET_SR
    print(f"[audio] final     : {signal.shape[0]} samples, {dur_s:.2f} s")
    if signal.shape[0] == 0:
        raise SystemExit("ERROR: audio is empty after processing.")
    return signal


def run_inference(processor, model, signal: np.ndarray):
    """Run one forward pass; return (outputs, elapsed_ms).

    Raises SystemExit with a clear message on failure.
    """
    try:
        proc = processor(signal, sampling_rate=TARGET_SR)
        input_values = torch.from_numpy(
            np.asarray(proc["input_values"][0], dtype=np.float32).reshape(1, -1)
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: feature extraction failed (wrong sample rate / bad signal?): "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_values)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: forward pass failed: {type(exc).__name__}: {exc}"
        )
    return outputs, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the audeering age/gender wav2vec2 model and dump raw output for one WAV."
    )
    parser.add_argument(
        "wav",
        type=Path,
        nargs="?",
        help="Path to a WAV file to analyze (or use --file).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        dest="file",
        help="Path to a WAV file to analyze (alternative to the positional arg).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory to download model weights into (default: {DEFAULT_CACHE_DIR}).",
    )
    args = parser.parse_args()

    wav_path = args.file or args.wav
    if wav_path is None:
        parser.error("no WAV file given: pass it positionally or with --file/-f")
    if args.file is not None and args.wav is not None and args.file != args.wav:
        parser.error(
            f"conflicting paths: positional {args.wav} vs --file {args.file}"
        )

    processor, model = load_model(args.cache_dir)
    signal = load_audio(wav_path)
    outputs, elapsed_ms = run_inference(processor, model, signal)

    hidden_states, logits_age, logits_gender = outputs

    print("\n" + "=" * 60)
    print("RAW MODEL OUTPUT (unprocessed)")
    print("=" * 60)
    print(f"return type            : tuple of {len(outputs)}")
    print(f"[0] hidden_states shape: {tuple(hidden_states.shape)}  dtype={hidden_states.dtype}")
    print(f"[1] logits_age   shape : {tuple(logits_age.shape)}  dtype={logits_age.dtype}")
    print(f"[2] logits_gender shape: {tuple(logits_gender.shape)}  dtype={logits_gender.dtype}")
    print()

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)

    print("--- age (regression head, raw) ---")
    print(logits_age)
    print(f"as float: {logits_age.squeeze().item()!r}")
    print()

    print("--- gender (softmax head, raw) ---")
    print("index order per model card: (child, female, male)")
    print(logits_gender)
    print(f"as list : {logits_gender.squeeze().tolist()!r}")
    print()

    print("--- hidden_states (pooled embedding, first 16 dims) ---")
    print(hidden_states.squeeze()[:16])
    print()

    print("=" * 60)
    print(f"inference elapsed: {elapsed_ms:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
