"""Root, liveness, readiness, and auth verification endpoints."""
from fastapi import APIRouter, Depends
from app.config import Settings, get_settings
from app.database.connection import database_component_status
from app.security import declare_openapi_auth
from app.storage.cloud_storage import storage_component_status

public_router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


@public_router.get("/")
async def root_info(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
        "docs": "/docs" if settings.docs_enabled else None,
    }


@public_router.get("/api/health")
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
    }


@protected_router.get("/api/health/ready")
async def readiness_check(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ready",
        "components": {
            "database": database_component_status(settings),
            "storage": storage_component_status(settings),
        },
    }


@protected_router.get("/api/auth/verify")
async def verify_auth() -> dict:
    return {"authenticated": True}
