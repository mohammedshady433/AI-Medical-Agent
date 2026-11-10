"""User-triggered Ollama pulls, with explicit size consent and hardware checks."""

import asyncio
import json
import httpx
from .diagnostics import hardware


STARTER_MODELS = ("gpt-oss:20b", "qwen2.5:7b-instruct")


class ModelManager:
    def __init__(self, settings):
        self.settings = settings
        self.state = {"running": False, "model": "", "status": "idle", "completed": 0, "total": 0}
        self.task = None

    async def start(self, model, accepted):
        cfg = self.settings.snapshot()
        allowed = {
            *STARTER_MODELS,
            cfg["models"]["offline"]["id"],
            cfg["documents"]["embedding"]["model"],
            cfg["models"]["experimental"]["id"],
            cfg["assistant"]["voice_model"],
            cfg["assistant"]["vision"]["model"],
        }
        if model not in allowed or "cloud" in model.lower():
            raise ValueError("Only the configured local models can be installed here.")
        if not accepted:
            raise ValueError("Accept the model download before continuing.")
        if self.state["running"]:
            raise ValueError("A model download is already running.")
        experimental = "120b" in model.lower()
        if experimental and not cfg["experimental_local_120b_accepted"]:
            raise ValueError("Accept the experimental 120B hardware warning in Settings first.")
        specs = await asyncio.to_thread(hardware, self.settings.data)
        minimum = 80 if experimental else 20 if "20b" in model.lower() else 8 if "7b" in model.lower() else 6 if "3b" in model.lower() else 3
        if specs["free_disk_gb"] < minimum:
            raise ValueError(f"Allow at least {minimum} GB of free disk space before this download.")
        self.state = {
            "running": True,
            "model": model,
            "status": "connecting",
            "completed": 0,
            "total": 0,
            "hardware": specs,
        }
        self.task = asyncio.create_task(self.pull(model, cfg["providers"]["ollama_local"]["endpoint"]))
        return self.state

    async def pull(self, model, endpoint):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(180, connect=10), trust_env=False, follow_redirects=False
            ) as client:
                async with client.stream(
                    "POST", endpoint.rstrip("/") + "/api/pull", json={"model": model, "stream": True}
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("error"):
                            raise ValueError(
                                "Ollama could not finish the download. Retry when the connection is stable."
                            )
                        self.state.update(
                            {k: event[k] for k in ("status", "completed", "total") if k in event}
                        )
            self.state["status"] = "installed"
        except asyncio.CancelledError:
            self.state["status"] = "stopped; partial downloads can resume"
        except httpx.ConnectError:
            self.state["status"] = (
                "Ollama is not running. Install/start Ollama or run scripts/setup-models.ps1 -InstallOllama."
            )
        except (httpx.HTTPError, ValueError):
            self.state["status"] = "Download failed. Check Ollama and internet connectivity; retry to resume."
        finally:
            self.state["running"] = False

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await self.task
