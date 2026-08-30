"""Structured (JSON-lines) logging for the service.

Every log record is emitted as a single-line JSON object on stdout so a log
shipper (Loki, CloudWatch, Datadog, …) can parse it without a regex. The
formatter promotes a small, fixed set of "extra" fields onto the top level
of the JSON so ``/analyze`` telemetry — ``contact_id``, ``audio_quality``,
``gender_prediction``, ``age_prediction``, ``processing_ms`` and the
``outcome`` (``ok`` / ``degraded`` / ``error_fallback``) — is queryable
directly.

PII rule
--------
The ``/analyze`` handler must pass **only** the generated ``contact_id`` as
an identifier. Never the upload's filename, the raw bytes, a transcript, or
anything derived from the audio content. This formatter does not enforce
that (it just serialises what it is given) — the discipline lives at the
call sites in ``app/api/routes.py``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging

# "extra=" keys we lift to the top level of the JSON record. Anything else
# passed via extra is dropped (kept out of logs on purpose).
_PROMOTED_FIELDS = (
    "contact_id",
    "audio_quality",
    "gender_prediction",
    "gender_confidence",
    "age_prediction",
    "age_confidence",
    "processing_ms",
    "inference_ms",
    "outcome",
    "timed_out",
    "timeout_ms",
    "speech_ratio",
    "total_duration_s",
    "decode_error",
    "upload_bytes",
    "error_type",
)

# LogRecord attributes that are always present; used to detect ad-hoc extras.
_RESERVED = set(
    vars(
        logging.LogRecord("", 0, "", 0, "", (), None)
    )
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Promote known telemetry fields from record.__dict__ (populated by
        # `logger.info(..., extra={...})`).
        for key in _PROMOTED_FIELDS:
            if key in record.__dict__ and record.__dict__[key] is not None:
                payload[key] = record.__dict__[key]

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger's stdout handler.

    Idempotent: safe to call from both the FastAPI lifespan and a test
    fixture. Replaces any existing handlers so uvicorn's default plain-text
    handler does not double-log.

    Args:
        level: Root log level. INFO in normal operation.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()  # stdout
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # uvicorn installs its own handlers on these; let them propagate to root
    # so access/error lines are JSON too, and drop the duplicate handlers.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
