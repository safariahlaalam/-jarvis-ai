"""Database URL normalization and readiness helpers."""
from app.config import Settings


def normalize_database_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned.startswith("postgres://"):
        return "postgresql+psycopg://" + cleaned[len("postgres://") :]
    if cleaned.startswith("postgresql://"):
        return "postgresql+psycopg://" + cleaned[len("postgresql://") :]
    return cleaned


def database_component_status(settings: Settings) -> str:
    return "configured" if settings.database_configured else "not_configured"
