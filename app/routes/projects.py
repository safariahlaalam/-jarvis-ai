"""Projects endpoints."""
from datetime import datetime, timezone
import os
import uuid

from fastapi import APIRouter, Depends, Request
import psycopg
from psycopg.rows import dict_row

from app.errors import not_configured_response
from app.schemas import ProjectCreateRequest
from app.security import declare_openapi_auth

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


def _get_connection():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        return None
    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        prepare_threshold=None,
    )


def _ensure_projects_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()


@router.get("/api/projects")
async def list_projects(request: Request):
    conn = _get_connection()
    if conn is None:
        return not_configured_response(
            feature="projects",
            message="Database is not configured for projects.",
            request_id=request.state.request_id,
        )

    with conn:
        _ensure_projects_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, created_at
                FROM projects
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()

    projects = [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]
    return {"projects": projects}


@router.post("/api/projects", status_code=201)
async def create_project(payload: ProjectCreateRequest, request: Request):
    conn = _get_connection()
    if conn is None:
        return not_configured_response(
            feature="projects",
            message="Database is not configured for projects.",
            request_id=request.state.request_id,
        )

    project_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    description = getattr(payload, "description", None)

    with conn:
        _ensure_projects_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (id, name, description, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (project_id, payload.name, description, created_at),
            )
        conn.commit()

    return {
        "id": project_id,
        "name": payload.name,
        "description": description,
        "created_at": created_at.isoformat(),
    }
