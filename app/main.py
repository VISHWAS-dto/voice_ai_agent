"""FastAPI application entrypoint.

Stub only. Wires the API routers together and will own model lifecycle
(load on startup) once ``app.inference.model`` is implemented.
"""

from fastapi import FastAPI

from app.api import routes, websocket


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Returns:
        A FastAPI app with the REST and WebSocket routers registered.
    """
    app = FastAPI(title="Voice Attribute Inference Service")
    app.include_router(routes.router)
    app.include_router(websocket.router)
    return app


app = create_app()
