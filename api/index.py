import os
import sys
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from app.main import app
except Exception as exc:
    app = FastAPI()

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
    async def startup_error_report(full_path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            },
        )
