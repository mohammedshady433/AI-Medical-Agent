"""Share local GPU capacity between the assistant and speech workers."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import httpx
from app.core.settings import local_url
from app.core.options import resident_voice


class GpuCoordinator:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport
        self.lock = asyncio.Lock()
        self.offload_speech = None
        self._voice_lease = ContextVar("ems_voice_gpu_lease", default=None)

    def _owns_voice_lease(self):
        lease = self._voice_lease.get()
        return lease is not None and lease[0]

    @asynccontextmanager
    async def voice_turn(self, enabled):
        """One resident voice turn may overlap its LLM and first TTS clip.

        Other turns, vision and memory release still take the exclusive lock.
        The caller must join its child tasks before leaving this scope.
        """
        if not enabled:
            yield
            return
        async with self.lock:
            lease = [True]
            token = self._voice_lease.set(lease)
            try:
                yield
            finally:
                lease[0] = False  # Inherited contexts cannot reuse an expired lease.
                self._voice_lease.reset(token)

    @asynccontextmanager
    async def assistant(self, config):
        if config["mode"] != "offline" or (config.get("_voice") and self._owns_voice_lease()):
            yield
            return
        async with self.lock:
            if not (config.get("_voice") and resident_voice(self.settings.snapshot())):
                if self.offload_speech: await self.offload_speech()
            yield

    @asynccontextmanager
    async def speech(self, uses_cuda):
        if not uses_cuda or self._owns_voice_lease():
            yield
            return
        async with self.lock:
            cfg = self.settings.snapshot()
            model = cfg["models"]["offline"]
            endpoint = cfg["providers"][model["provider"]]["endpoint"].rstrip("/")
            if not local_url(endpoint):
                raise ValueError("Local GPU coordination requires a loopback endpoint.")
            async with httpx.AsyncClient(timeout=30, trust_env=False, transport=self.transport) as client:
                try:
                    response = await client.get(endpoint + "/api/ps")
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    response = None  # Speech also works when no local assistant is running.
                if response is not None:
                    response.raise_for_status()
                    names = {item.get("name", item.get("model", "")).removesuffix(":latest") for item in response.json().get("models", [])}
                    voice = cfg["assistant"]["voice_model"] or model["id"]
                    for selected in {model["id"], voice}:
                        if resident_voice(cfg) and selected == voice:
                            continue
                        if selected.removesuffix(":latest") in names:
                            response = await client.post(endpoint + "/api/generate", json={"model": selected, "keep_alive": 0, "stream": False})
                            response.raise_for_status()
            yield

    async def release(self):
        async with self.lock:
            await self._release()

    @asynccontextmanager
    async def vision(self):
        # Image analysis must not compete with resident voice weights on smaller GPUs.
        async with self.lock:
            if self.offload_speech:
                await self.offload_speech()
            await self._release()
            yield

    async def _release(self):
        """Caller owns the GPU lock; never unload unrelated user models."""
        cfg = self.settings.snapshot()
        model = cfg["models"]["offline"]
        endpoint = cfg["providers"][model["provider"]]["endpoint"].rstrip("/")
        if not local_url(endpoint): raise ValueError("Memory release requires a local endpoint.")
        owned = {model["id"], cfg["assistant"]["voice_model"], cfg["assistant"]["vision"]["model"]} - {""}
        async with httpx.AsyncClient(timeout=20, trust_env=False, transport=self.transport) as client:
            try: response = await client.get(endpoint + "/api/ps")
            except (httpx.ConnectError, httpx.ConnectTimeout): return
            response.raise_for_status()
            loaded = {x.get("name", "").removesuffix(":latest") for x in response.json().get("models", [])}
            for name in owned:
                if name.removesuffix(":latest") in loaded:
                    result = await client.post(endpoint + "/api/generate", json={"model": name, "keep_alive": 0, "stream": False})
                    result.raise_for_status()
