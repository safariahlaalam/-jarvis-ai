"""Cloudflare R2 object key sanitization and readiness helpers."""
from pathlib import PurePosixPath
import re
from app.config import Settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def build_object_key(workspace_id: str, file_id: str, filename: str) -> str:
    if not workspace_id or not _SAFE_ID.match(workspace_id):
        raise ValueError("Invalid workspace_id")
    if not file_id or not _SAFE_ID.match(file_id):
        raise ValueError("Invalid file_id")

    raw_name = PurePosixPath(filename.replace("\\", "/")).name
    sanitized_name = _UNSAFE_FILENAME_CHARS.sub("_", raw_name).lstrip(".")
    if not sanitized_name:
        raise ValueError("Invalid filename")

    return f"workspaces/{workspace_id}/files/{file_id}/{sanitized_name}"


def storage_component_status(settings: Settings) -> str:
    return "configured" if settings.r2_configured else "not_configured"
