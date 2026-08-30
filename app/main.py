"""FastAPI application entrypoint.

Wires the REST + WebSocket routers together and owns the inference model's
lifecycle: the ~1.3 GB ``AttributeInferencer`` is loaded **once** in the
lifespan startup handler and stashed on ``app.state.inferencer`` so every
``POST /analyze`` reuses it. Loading per-request would add a multi-second
cold start to every call.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes, websocket
from app.inference.model import AttributeInferencer

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the inference model on startup, drop the reference on shutdown.

    The constructor downloads / reads ~1.3 GB of weights and can take
    several seconds. Doing it here means the process is not marked ready
    until the model is actually usable, and no request ever pays that
    cost.
    """
    logger.info("startup: loading AttributeInferencer (one-time model load)")
    app.state.inferencer = AttributeInferencer()
    logger.info("startup: model loaded, service ready")
    try:
        yield
    finally:
        # Let the GC reclaim the ~1.3 GB of weights promptly on shutdown.
        app.state.inferencer = None
        logger.info("shutdown: released inference model")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Returns:
        A FastAPI app with the REST and WebSocket routers registered and
        the model-loading lifespan attached.
    """
    app = FastAPI(title="Voice Attribute Inference Service", lifespan=lifespan)
    app.include_router(routes.router)
    app.include_router(websocket.router)
    return app


app = create_app()
