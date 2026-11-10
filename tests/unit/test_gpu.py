import asyncio
import json
import httpx
from app.services.gpu import GpuCoordinator


async def test_speech_waits_for_assistant_then_unloads_only_selected_model(settings):
    calls, entered = [], asyncio.Event()
    def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}, {"name": "another-model"}]})
        assert json.loads(request.content) == {"model": "gpt-oss:20b", "keep_alive": 0, "stream": False}
        return httpx.Response(200, json={"done": True})
    gpu = GpuCoordinator(settings, httpx.MockTransport(respond))
    async def speech():
        async with gpu.speech(True):
            entered.set()
    config = settings.snapshot()
    config["mode"] = "offline"
    async with gpu.assistant(config):
        task = asyncio.create_task(speech())
        await asyncio.sleep(0)
        assert not entered.is_set() and not calls
    await task
    assert entered.is_set()
    assert calls == [("GET", "/api/ps"), ("POST", "/api/generate")]


async def test_cpu_speech_does_not_contact_ollama(settings):
    def respond(request):
        raise AssertionError("CPU speech must not unload models")
    async with GpuCoordinator(settings, httpx.MockTransport(respond)).speech(False):
        pass


async def test_uninstalled_assistant_is_not_loaded_for_speech(settings):
    calls = []
    def respond(request):
        calls.append(request.method)
        return httpx.Response(200, json={"models": []})
    async with GpuCoordinator(settings, httpx.MockTransport(respond)).speech(True):
        pass
    assert calls == ["GET"]


async def test_voice_lease_allows_own_speech_but_keeps_other_work_exclusive(settings):
    gpu = GpuCoordinator(settings)
    start, entered = asyncio.Event(), asyncio.Event()
    async def unrelated():
        await start.wait()
        async with gpu.lock:
            entered.set()
    other = asyncio.create_task(unrelated())  # Created outside the voice context.
    async with gpu.voice_turn(True):
        start.set()
        async with gpu.assistant({"mode": "offline", "_voice": True}):
            async with gpu.speech(True):
                await asyncio.sleep(0)
                assert not entered.is_set()
    await asyncio.wait_for(other, 1)
    assert entered.is_set() and not gpu._owns_voice_lease()
