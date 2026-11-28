import json
import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def _build_diagnostic_payload(exc: Exception) -> dict:
    app_dir = os.path.join(ROOT_DIR, "app")
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "root_files": sorted(os.listdir(ROOT_DIR)),
        "app_files": sorted(os.listdir(app_dir)) if os.path.isdir(app_dir) else "MISSING_APP_DIR",
        "has_service_token": bool(os.getenv("SERVICE_TOKEN")),
        "traceback": traceback.format_exc().splitlines()[-8:],
    }

try:
    from app.main import app
except Exception as main_exc:
    diag = _build_diagnostic_payload(main_exc)
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
        async def debug_view(path: str = ""):
            return JSONResponse(status_code=200, content=diag)
    except Exception:
        from http.server import BaseHTTPRequestHandler

        class handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(diag, ensure_ascii=False, indent=2).encode("utf-8"))
