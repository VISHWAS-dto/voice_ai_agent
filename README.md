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
and unexpected internal errors all resolve to an all-`unknown` body rather
than an error status, so a failure here never breaks the calling voice AI
system. The model is loaded once at startup (FastAPI lifespan), not
per-request.

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

_TODO_

```bash
docker compose up --build
```

## Testing

_TODO_

```bash
pytest
```

## Evaluation

_TODO_ — see [`eval/`](eval/).
