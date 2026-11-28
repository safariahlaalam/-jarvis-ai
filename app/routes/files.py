"""Files endpoints."""
from datetime import datetime, timezone
import os
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.errors import not_configured_response
from app.security import declare_openapi_auth

# Prevent newer boto3 releases from sending AWS-only CRC32 headers to Backblaze B2
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


def _get_storage():
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket_name = os.getenv("R2_BUCKET_NAME", "").strip()
    endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip()

    if not all([access_key, secret_key, bucket_name, endpoint_url]):
        return None, None

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )
    return client, bucket_name


@router.get("/api/files")
async def list_files(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    client, bucket_name = _get_storage()
    if client is None:
        return not_configured_response(
            feature="files",
            message="Object storage is not configured for files.",
            request_id=request.state.request_id,
        )

    try:
        response = client.list_objects_v2(Bucket=bucket_name)
    except ClientError as exc:
        raise HTTPException(status_code=502, detail=f"Storage error: {exc}") from exc

    contents = response.get("Contents", [])
    contents.sort(key=lambda item: item["LastModified"], reverse=True)
    page = contents[offset : offset + limit]

    files = [
        {
            "key": item["Key"],
            "size_bytes": item["Size"],
            "last_modified": item["LastModified"].isoformat(),
        }
        for item in page
    ]
    return {
        "files": files,
        "total": len(contents),
        "limit": limit,
        "offset": offset,
    }


@router.post("/api/files", status_code=201)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    folder: str | None = Form(default=None),
):
    client, bucket_name = _get_storage()
    if client is None:
        return not_configured_response(
            feature="files",
            message="Object storage is not configured for files.",
            request_id=request.state.request_id,
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    safe_filename = os.path.basename(file.filename or "upload.bin")
    unique_name = f"{uuid.uuid4()}_{safe_filename}"
    clean_folder = (folder or "").strip().strip("/")
    object_key = f"{clean_folder}/{unique_name}" if clean_folder else unique_name
    content_type = file.content_type or "application/octet-stream"

    try:
        client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=raw_bytes,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}") from exc

    return {
        "key": object_key,
        "filename": safe_filename,
        "content_type": content_type,
        "size_bytes": len(raw_bytes),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/files/download")
async def get_download_url(
    request: Request,
    key: str = Query(..., min_length=1),
):
    client, bucket_name = _get_storage()
    if client is None:
        return not_configured_response(
            feature="files",
            message="Object storage is not configured for files.",
            request_id=request.state.request_id,
        )

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": key},
            ExpiresIn=3600,
        )
    except ClientError as exc:
        raise HTTPException(status_code=502, detail=f"Presign failed: {exc}") from exc

    return {"key": key, "download_url": url, "expires_in_seconds": 3600}
