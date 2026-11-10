"""Packaged backend entrypoint; Electron supplies a token and discovers the bound random port."""

import json
import logging
import os
from pathlib import Path
import socket
import sys
import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True)
    if os.name == "nt":
        import win32api, win32job

        owned_job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(owned_job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(owned_job, win32job.JobObjectExtendedLimitInformation, limits)
        win32job.AssignProcessToJobObject(owned_job, win32api.GetCurrentProcess())
    if not getattr(sys, "frozen", False):
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from app.main import create_app

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", int(os.environ.get("EMS_PORT", "0"))))
    listener.listen(128)
    port = listener.getsockname()[1]
    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                print(json.dumps({"event": "ems-ready", "port": port}), flush=True)

    ReadyServer(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning", access_log=False)
    ).run(sockets=[listener])
