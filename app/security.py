"""Service token verification and OpenAPI security scheme."""
import hmac
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_scheme = APIKeyHeader(name="X-Service-Token", auto_error=False)


async def declare_openapi_auth(_: str | None = Security(api_key_scheme)) -> None:
    """Declares X-Service-Token in OpenAPI schema; actual enforcement runs in middleware."""
    return None


def check_service_token(provided: str | None, expected: str | None) -> tuple[int, str, str] | None:
    if expected is None:
        return (503, "service_unavailable", "Service authentication is not configured")
    if not provided or not hmac.compare_digest(provided, expected):
        return (401, "unauthorized", "Invalid or missing service token")
    return None
