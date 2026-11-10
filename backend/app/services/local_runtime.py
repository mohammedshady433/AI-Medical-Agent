"""Start only an explicitly configured private Ollama; no download or mode switch here."""

import asyncio
import logging
import signal
import os
from pathlib import Path
import subprocess
import psutil
from urllib.parse import urlparse
import httpx


class LocalRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.process = None

    async def start(self):
        cfg = self.settings.snapshot()
        runtime = cfg.get("local_runtime", {})
        executable = runtime.get("executable", "")
        if not executable or not Path(executable).is_file():
            return
        endpoint = cfg["providers"]["ollama_local"]["endpoint"].rstrip("/")
        async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
            try:
                response = await client.get(endpoint + "/api/version")
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
        parsed = urlparse(endpoint)
        env = dict(
            os.environ,
            OLLAMA_HOST=f"{parsed.hostname}:{parsed.port or 11434}",
            OLLAMA_MODELS=runtime["models_dir"],
            OLLAMA_NO_CLOUD="1",
            OLLAMA_NUM_PARALLEL="1",
            OLLAMA_MAX_LOADED_MODELS="2",
        )
        logs = self.settings.data / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "ollama.log").open("ab") as log:
            self.process = psutil.Popen(
                [executable, "serve"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt",
            )
        async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
            for _ in range(40):
                if self.process.poll() is not None:
                    logging.getLogger(__name__).warning("The configured local model service stopped during startup; see logs/ollama.log")
                    return
                try:
                    response = await client.get(endpoint + "/api/version")
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        logging.getLogger(__name__).warning("The local model service is still starting; see logs/ollama.log")

    async def close(self):
        owned, self.process = self.process, None
        if owned and owned.poll() is None:
            await asyncio.to_thread(self._stop_owned_tree, owned)

    @staticmethod
    def _stop_owned_tree(owned):
        """Windows does not kill Ollama's GPU worker when its parent exits.

        Capture only descendants of the process this instance started, before
        terminating the parent. psutil tracks process identity against PID reuse.
        An already-running external Ollama is never assigned to self.process.
        """
        if os.name != "nt" and os.getpgid(owned.pid) == owned.pid:
            try:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=5)
            except psutil.TimeoutExpired:
                os.killpg(owned.pid, signal.SIGKILL)
                owned.wait(timeout=5)
            except ProcessLookupError:
                pass
            return
        try:
            children = owned.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        try:
            owned.terminate()
        except psutil.NoSuchProcess:
            pass
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(children, timeout=5)
        for child in remaining:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            owned.wait(timeout=5)
        except psutil.TimeoutExpired:
            owned.kill()
            owned.wait(timeout=5)
        psutil.wait_procs(remaining, timeout=5)
