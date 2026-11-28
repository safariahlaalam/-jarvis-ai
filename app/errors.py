"""Centralized error responses and exception handlers."""
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("jervis.errors")

_STATUS_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    503: "service_unavailable",
}


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    payload: dict = {
        "status": "error",
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if details is not None:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload, headers={"X-Request-ID": request_id})


def not_configured_response(feature: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "status": "not_configured",
            "feature": feature,
            "message": message,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = _get_request_id(request)
        # Never echo submitted input values back in validation errors.
        details = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())),
                "message": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        return error_response(
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            request_id=request_id,
            details=details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _get_request_id(request)
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) and exc.detail else "Request failed"
        return error_response(
            status_code=exc.status_code,
            code=code,
            message=message,
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _get_request_id(request)
        logger.exception("Unhandled exception [request_id=%s]", request_id)
        return error_response(
            status_code=500,
            code="internal_error",
            message="Internal server error",
            request_id=request_id,
        )
