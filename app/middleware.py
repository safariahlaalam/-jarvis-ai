"""Request ID tracing, safe request logging, auth enforcement, and CORS."""
import logging
import re
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.errors import error_response
from app.security import check_service_token

logger = logging.getLogger("jervis.access")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_PUBLIC_PATHS = {"/", "/api/health", "/docs", "/openapi.json"}


def resolve_request_id(incoming: str | None) -> str:
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return str(uuid.uuid4())


def register_middlewares(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def _request_context_and_auth(request: Request, call_next):
        request_id = resolve_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id

        path = request.url.path
        if request.method != "OPTIONS" and path.startswith("/api/") and path not in _PUBLIC_PATHS:
            auth_error = check_service_token(
                provided=request.headers.get("X-Service-Token"),
                expected=settings.service_token,
            )
            if auth_error is not None:
                status_code, code, message = auth_error
                response = error_response(status_code, code, message, request_id)
                response.headers["X-Request-ID"] = request_id
                logger.info("%s %s -> %s [request_id=%s]", request.method, path, response.status_code, request_id)
                return response

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s -> %s [request_id=%s]", request.method, path, response.status_code, request_id)
        return response

    # Registered last so it wraps everything as the outermost middleware.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list({*settings.cors_origins, "https://afariahlaalam.github.io"}),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["X-Service-Token", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
