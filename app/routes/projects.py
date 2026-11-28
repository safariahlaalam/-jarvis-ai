"""Projects endpoints."""
from fastapi import APIRouter, Depends, Request
from app.errors import not_configured_response
from app.schemas import ProjectCreateRequest
from app.security import declare_openapi_auth

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


@router.get("/api/projects")
async def list_projects(request: Request):
    return not_configured_response(
        feature="projects",
        message="Database is not configured for projects.",
        request_id=request.state.request_id,
    )


@router.post("/api/projects")
async def create_project(payload: ProjectCreateRequest, request: Request):
    return not_configured_response(
        feature="projects",
        message="Database is not configured for projects.",
        request_id=request.state.request_id,
    )
