"""Run the EMS AI core with a browser test console for chat and voice (development only).

Starts the same authenticated FastAPI backend the desktop app used, bound to 127.0.0.1,
and serves test-console/index.html from it at /console/. A fresh session token is
created on every start and passed in the URL fragment (#token=...), which browsers
never send to the server. Every API call from the page still needs that token.

    .venv\\Scripts\\python.exe scripts\\test_console.py            (Windows)
    .venv/bin/python scripts/test_console.py                     (macOS)
"""

import argparse
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="EMS AI core test console")
    parser.add_argument("--port", type=int, default=8765, help="Loopback port (default 8765).")
    parser.add_argument(
        "--data-dir",
        help="EMS data folder with settings, chats, documents and installed models. "
        "Default: $EMS_DATA_DIR, else ./.dev-data when present, else the installed app's data folder.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()

    # The backend resolves its data folder at import time, so set it first.
    data = args.data_dir or os.environ.get("EMS_DATA_DIR")
    if not data and (ROOT / ".dev-data").is_dir():
        data = str(ROOT / ".dev-data")
    if data:
        os.environ["EMS_DATA_DIR"] = str(Path(data).resolve())
    os.environ.setdefault("PYTHONUTF8", "1")

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    if os.name == "nt":
        # Close speech workers and the private Ollama together with this process.
        import win32api, win32job

        job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        win32job.AssignProcessToJobObject(job, win32api.GetCurrentProcess())

    sys.path.insert(0, str(ROOT / "backend"))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from app.core.settings import DATA
    from app.main import create_app

    token = secrets.token_hex(32)
    app = create_app(DATA, token)
    # A mounted static app is outside the API's token dependency; it holds no data.
    app.mount("/console", StaticFiles(directory=ROOT / "test-console", html=True), name="console")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", args.port))
    except OSError:
        raise SystemExit(f"Port {args.port} is in use. Close the other program or pass --port <number>.")
    listener.listen(128)
    url = f"http://127.0.0.1:{args.port}/console/#token={token}"

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                print(json.dumps({"event": "ems-ready", "port": args.port}), flush=True)
                print(f"\nEMS AI core is running. Data folder: {DATA}")
                print(f"Open the test console:\n  {url}\nPress Ctrl+C to stop.\n", flush=True)
                if not args.no_browser:
                    webbrowser.open(url)

    ReadyServer(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    ).run(sockets=[listener])


if __name__ == "__main__":
    main()
