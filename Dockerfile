# syntax=docker/dockerfile:1

# Slim Python base — small image, glibc (not musl) so the prebuilt
# torch / numpy / webrtcvad wheels install cleanly.
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependency: ffmpeg.
#
# app/audio/ingest.py shells out to the `ffmpeg` binary (over stdin/stdout
# pipes) to decode + resample uploads. It is NOT a pip package —
# `ffmpeg-python` is only a thin wrapper — so it has to come from apt.
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Runtime cache path the app reads (passed to HF as `cache_dir=`).
    # docker-compose mounts a named volume here so weights persist across
    # `docker compose down` / `up`.
    MODEL_CACHE_DIR=/app/.model_cache \
    # Where the build BAKES a pristine copy of the weights. The entrypoint
    # seeds MODEL_CACHE_DIR from here on first run if the volume is empty.
    MODEL_CACHE_SEED=/opt/model-seed \
    LOG_LEVEL=INFO

# ---------------------------------------------------------------------------
# Python deps first, as their own layer, so app-code edits don't bust the
# (slow) pip install cache.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Application code (before the weight download so `python -m
# app.inference.model` can run).
# ---------------------------------------------------------------------------
COPY app ./app

# ---------------------------------------------------------------------------
# Pre-download + cache the HuggingFace model weights AT BUILD TIME, into the
# SEED path (not the runtime path, which a volume will shadow).
#
# Tradeoff — bake at build vs. download on first start:
#
#   * Baking (what we do): the ~1.3 GB of weights land in the image during
#     build. First container start needs NO network and the /health
#     `model_loaded` flag flips within a second or two of boot — the
#     reliable choice for offline / air-gapped / flaky-egress deploys.
#     Cost: a bigger image and a longer, network-dependent build. This
#     layer is cached, so it only re-runs when requirements.txt or the
#     model code changes.
#   * Download on first start: smaller image, faster build, but the first
#     request (or healthcheck) after a cold start blocks on the
#     multi-second download and fails outright if the host can't reach
#     huggingface.co. The 3.4 s first-request latency seen locally was
#     exactly this.
# ---------------------------------------------------------------------------
RUN MODEL_CACHE_DIR="$MODEL_CACHE_SEED" python -m app.inference.model

# After the weights are baked, pin the HF client to offline mode: the
# runtime must never reach out to huggingface.co (no network dependency, no
# per-start HEAD-request latency, no noisy httpx INFO logs). The seed copy
# is complete, so offline resolution succeeds.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Everything else (tests, README, sample.wav, eval/, entrypoint) after the
# heavy layers.
COPY . .

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

# The entrypoint seeds the (possibly volume-mounted, possibly empty)
# MODEL_CACHE_DIR from MODEL_CACHE_SEED once, then execs the CMD.
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# One worker: the model is ~1.3 GB resident and CPU-bound per request; add
# replicas at the orchestrator layer, not with --workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
