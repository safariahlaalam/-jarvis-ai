"""FastAPI application factory."""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_exception_handlers
from app.middleware import register_middlewares
from app.routes import chat, files, health, projects


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        docs_url="/docs" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        redoc_url=None,
        lifespan=_lifespan,
    )

    register_exception_handlers(app)
    register_middlewares(app, settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://afariahlaalam.github.io"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.public_router)
    app.include_router(health.protected_router)
    app.include_router(chat.router)
    app.include_router(projects.router)
    app.include_router(files.router)

    return app


app = create_app()
