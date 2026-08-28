# Voice Attribute Inference Service

> Placeholder — fill in as modules land.

Infers speaker **gender** and **age bracket** from short audio clips, for a
logistics call-center use case.

## Overview

_TODO_

## API

### `POST /analyze`

_TODO_ — multipart audio upload, returns:

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male", "confidence": 0.87},
  "age_bracket": {"prediction": "31-45", "confidence": 0.63},
  "processing_ms": 142,
  "audio_quality": "good"
}
```

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
