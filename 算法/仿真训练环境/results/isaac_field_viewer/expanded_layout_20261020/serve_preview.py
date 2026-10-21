"""Serve a staged HTML layout against the existing localhost viewer API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
UPSTREAM = "http://127.0.0.1:8770"
PORT = 8772


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def forward(self, body=None):
        request = urllib.request.Request(
            UPSTREAM + self.path,
            data=body,
            headers={"Content-Type": "application/json", "Origin": UPSTREAM},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                self.send(response.status, response.read(), response.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as error:
            self.send(error.code, error.read(), "application/json")
        except (urllib.error.URLError, TimeoutError):
            self.send(502, b'{"error":"viewer API unavailable"}', "application/json")

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            try:
                body = (ROOT / "index.html").read_text(encoding="utf-8")
                body = body.replace("__VIEWER_URL__", "http://127.0.0.1:8769/")
                self.send(200, body.encode("utf-8"), "text/html; charset=utf-8")
            except OSError:
                self.send(503, b"layout is being prepared", "text/plain")
        elif path == "/api/frame":
            self.forward()
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/control":
            self.send(404, b"not found", "text/plain")
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin != f"http://{host}":
            self.send(400, b"cross-origin request rejected", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("invalid body length")
            body = self.rfile.read(length)
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("command") not in {"run", "pause", "reset", "preview"}:
                raise ValueError("unsupported command")
        except (ValueError, UnicodeDecodeError):
            self.send(400, b"invalid control request", "text/plain")
            return
        self.forward(body)


if __name__ == "__main__":
    print(f"Staged expanded layout: http://127.0.0.1:{PORT}/; API: {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
