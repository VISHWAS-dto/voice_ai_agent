# Voice Attribute Inference Service

> Placeholder — fill in as modules land.

Infers speaker **gender** and **age bracket** from short audio clips, for a
logistics call-center use case.

## Overview

_TODO_

## API

### `POST /analyze`

`multipart/form-data` upload of an audio clip (field name `audio`; wav / mp3 /
m4a / opus / 8 kHz telephony all accepted). Returns:

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male", "confidence": 0.87},
  "age_bracket": {"prediction": "31-45", "confidence": 0.63},
  "processing_ms": 142,
  "audio_quality": "good"
}
```

Pipeline: the bytes are read into memory (never written to disk — call audio
carries PII), decoded/resampled to 16 kHz mono, run through WebRTC VAD, and
graded `good` / `degraded` / `insufficient`. An `insufficient` grade (or a
decode failure) **skips the model** and returns `unknown` / `unknown` at
`0.0` confidence. `processing_ms` is end-to-end (ingest + VAD + inference).

The endpoint always responds **HTTP 200**: unusable audio, corrupt uploads,
a processing timeout, and unexpected internal errors all resolve to an
all-`unknown` body (with `audio_quality: "insufficient"`) rather than an
error status, so a failure here never breaks the calling voice AI system.
The model is loaded once at startup (FastAPI lifespan), not per-request.

### Reliability

- **Never a 500.** The handler has a top-level `try/except` that catches
  *any* exception and returns the all-`unknown` / `insufficient` body.
- **Never hangs.** The decode → VAD → inference pipeline runs in a worker
  thread under a hard 3 s wall-clock budget (`_PROCESSING_BUDGET_S` in
  `app/api/routes.py`). On overrun the request returns the safe body
  immediately; the orphaned thread finishes and is discarded (the pipeline
  holds no shared mutable state).
- **Model loaded once.** `app/main.py`'s lifespan builds the
  `AttributeInferencer` on startup and stashes it on `app.state`. The
  process is not marked ready until the ~1.3 GB of weights are resident, so
  no request pays the cold-start cost.

### `GET /health`

Container health/readiness probe. `{"status": "ok", "model_loaded": <bool>}`.
`model_loaded` is `true` only after the startup lifespan has finished
loading the model. (`GET /healthz` is a backwards-compatible alias.)

### Observability

Every `/analyze` request emits exactly **one structured JSON log line** on
stdout (`app/logging_config.py`) with: `contact_id`, `audio_quality`,
`gender_prediction`, `age_prediction`, `processing_ms`, `inference_ms`, and
`outcome` (`ok` / `degraded` / `error_fallback`). Levels:

| Level     | When                                                        |
|-----------|-------------------------------------------------------------|
| `INFO`    | normal request, `good` audio                               |
| `WARNING` | `degraded` / `insufficient` audio, decode failure, timeout |
| `ERROR`   | the exception fallback path                                 |

The **only** identifier logged is the generated `contact_id`. The upload's
filename, byte content, and any transcript are never logged. Set
`LOG_LEVEL` (default `INFO`) to change verbosity.

#### Limitations

- **Audio-quality thresholds are untuned.** `app/audio/quality.py` grades a
  clip purely on VAD speech-ratio (`>= 0.5` → `good`, `0.15–0.5` →
  `degraded`, `< 0.15` or `< 0.5 s` of speech → `insufficient`). These cut
  points are a hand-picked starting point, **not** calibrated against a set
  of human-labeled clips. They should be tuned once such a dataset exists;
  the current values will misgrade some borderline clips.
- No SNR / clipping / level signal feeds the grade yet — only speech ratio.

### `WS /ws/analyze`

_TODO_ — streaming variant.

## Architecture

_TODO_

- `app/api/` — HTTP + WebSocket routes
- `app/audio/` — ingestion (decode/resample) and quality gating
- `app/inference/` — model wrapper
- `app/schemas/` — Pydantic response contract

## Development

_TODO_

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose up --build
```

The image (`Dockerfile`):

- `python:3.11-slim` base.
- Installs **ffmpeg** via `apt-get` — it's a system binary the audio
  ingestion shells out to, not a pip package.
- Installs `requirements.txt`, then copies the app.
- **Pre-downloads the ~1.3 GB HF model weights at build time** into a seed
  path baked into the image, and pins `HF_HUB_OFFLINE=1` for the runtime.
  Tradeoff: bigger image + a network-dependent build, in exchange for a
  first container start that needs no network and reports
  `model_loaded: true` within a second or two. The alternative
  (download-on-first-start) makes the first request block on a multi-second
  download that fails outright with no egress — see the comment in the
  Dockerfile.
- `docker-entrypoint.sh` seeds the runtime cache (`/app/.model_cache`, a
  compose-mounted named volume) from the baked copy on first run, so
  restarts reuse the volume instead of re-downloading.
- Exposes `8000`, runs `uvicorn app.main:app` as `CMD`.

`docker-compose.yml` builds from the Dockerfile, maps `8000:8000`, mounts
the `model-cache` named volume at `/app/.model_cache` so weights persist
across `down`/`up`, and adds a `/health`-based healthcheck (the container
is only `healthy` once the model has loaded).

## Testing

_TODO_

```bash
pytest
```

## Evaluation

_TODO_ — see [`eval/`](eval/).
