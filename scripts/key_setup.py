"""Optional local one-time credential form for setup before the desktop is installed."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.core.secrets import SecretStore

TOKEN = secrets.token_urlsafe(32)
DATA = Settings().data
STORE = SecretStore(DATA)
PAGE = """<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EMS · Provider setup</title><style>body{margin:0;background:#f2f6f8;color:#183244;font:16px Segoe UI,Arial,sans-serif;display:grid;place-items:center;min-height:100vh}.card{width:min(480px,85vw);background:white;border:1px solid #dce5ea;border-radius:18px;padding:38px;box-shadow:0 15px 60px #1233}.brand{letter-spacing:4px;color:#0b8f8c;font-weight:700}h1{font-size:26px}p{color:#667e8e;line-height:1.6}label{font-weight:600;font-size:13px;display:block;margin-top:25px}input{box-sizing:border-box;width:100%;margin:10px 0 22px;border:1px solid #beced8;border-radius:9px;padding:13px;font:inherit}button{width:100%;padding:13px;border:0;border-radius:9px;background:#0b8f8c;color:white;font:inherit;cursor:pointer}small{display:block;line-height:1.6;color:#738897;margin-top:20px}</style><div class="card"><span class="brand">EMS</span><h1>Connect Ollama Cloud</h1><p>Default model: <strong>GPT-OSS 120B</strong>.<br>This setup page runs only on your computer.</p><form method="post"><label for="key">Ollama API key</label><input id="key" name="key" type="password" autocomplete="off" required maxlength="4096" placeholder="Paste your API key"><button>Save key securely</button></form><small>The key is encrypted for your Windows account and stored outside the app. It is never included in source code, build files or this chat.</small></div></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def headers_for(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def valid(self):
        return self.path == "/" + TOKEN and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        if not self.valid():
            self.headers_for(404)
            return
        self.headers_for()
        self.wfile.write(PAGE.encode("utf8"))

    def do_POST(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if not self.valid() or self.headers.get("Origin") != origin:
            self.headers_for(403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length < 16384:
            self.headers_for(400)
            return
        value = parse_qs(self.rfile.read(length).decode("utf8")).get("key", [""])[0].strip()
        if not value or len(value) > 4096:
            self.headers_for(400)
            return
        STORE.save("ollama_cloud", value)
        self.headers_for()
        self.wfile.write(
            b'<html><title>EMS - Key saved</title><body style="font:20px Segoe UI;padding:50px;color:#123"><h1>Key saved securely</h1><p>You can close this page. EMS will use your key for Ollama Cloud.</p></body></html>'
        )
        (ROOT / "logs/key-setup-status.json").write_text(json.dumps({"saved": True}))
        threading.Thread(target=self.server.shutdown, daemon=True).start()


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}/{TOKEN}"
    (ROOT / "logs/key-setup-url.txt").write_text(url)
    timer = threading.Timer(900, server.shutdown)
    timer.daemon = True
    timer.start()
    print("Local key form is ready. No credential values will be logged.", flush=True)
    server.serve_forever()
    server.server_close()
    timer.cancel()
