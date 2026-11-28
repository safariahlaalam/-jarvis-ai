import json
import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from app.main import app
except Exception:
    from http.server import BaseHTTPRequestHandler

    ERROR_TRACE = traceback.format_exc()

    class handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            payload = json.dumps({"error": ERROR_TRACE}, ensure_ascii=False, indent=2)
            self.wfile.write(payload.encode("utf-8"))
