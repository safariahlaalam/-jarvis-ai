"""Files endpoints."""
from fastapi import APIRouter, Depends, Query, Request
from app.errors import not_configured_response
from app.security import declare_openapi_auth

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


@router.get("/api/files")
async def list_files(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return not_configured_response(
        feature="files",
        message="Object storage is not configured for files.",
        request_id=request.state.request_id,
    )
