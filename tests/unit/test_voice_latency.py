import asyncio
import base64
import json
from unittest.mock import AsyncMock
import httpx
import pytest
from app.agents.assistant import voice_config
from app.core.options import resident_voice
from app.services.gpu import GpuCoordinator
from app.speech.service import Speech
from app.main import create_app
from app.providers.client import Provider, ProviderError


@pytest.mark.parametrize("model,limit,resident", [("gpt-oss:20b", 512, True), ("qwen2.5:7b-instruct", 256, True)])
def test_voice_model_choices_are_local_and_memory_safe(settings, model, limit, resident):
    settings.update({"assistant": {"voice_model": model, "voice_low_latency": True}})
    config = voice_config(settings.snapshot())
    assert config["mode"] == "offline" and config["models"]["offline"]["id"] == model
    assert config["models"]["offline"]["max_response_tokens"] == limit
    assert resident_voice(config) == resident
    assert settings.snapshot()["mode"] == "online"


async def test_7b_not_evicted_when_also_selected_for_text(settings):
    settings.update({"models": {"offline": {"id": "qwen2.5:7b-instruct"}}, "assistant": {"voice_model": "qwen2.5:7b-instruct", "voice_low_latency": True}})
    def handle(request):
        assert request.method == "GET", "Do not evict a model that fits with speech"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b-instruct"}]})
    gpu = GpuCoordinator(settings, httpx.MockTransport(handle))
    gpu.offload_speech = AsyncMock()
    async with gpu.speech(True): pass
    async with gpu.assistant(voice_config(settings.snapshot())): pass
    gpu.offload_speech.assert_not_called()
    settings.update({"assistant": {"voice_model": "gpt-oss:20b", "voice_low_latency": False}})
    async with gpu.assistant(voice_config(settings.snapshot())): pass
    gpu.offload_speech.assert_awaited_once()


async def test_speech_cache_reuses_audio_but_respects_voice_settings(settings, secrets):
    speech = Speech(settings, secrets)
    speech.local = AsyncMock(return_value={"audio": base64.b64encode(b"fixture-wav").decode(), "warnings": []})
    assert await speech.synthesize("أهلاً", "ar") == await speech.synthesize("أهلاً", "ar")
    assert speech.local.await_count == 1
    settings.update({"speech": {"tts": {"speed": 1.1}}})
    await speech.synthesize("أهلاً", "ar"); assert speech.local.await_count == 2
    settings.update({"speech": {"tts": {"provider": "azure"}}})
    with pytest.raises(ValueError): await speech.synthesize("private", "en", local_only=True)
    await speech.close(); assert not speech.audio_cache and speech.cache_bytes == 0


async def test_chatterbox_cache_shares_only_equivalent_spoken_forms(settings, secrets):
    speech = Speech(settings, secrets)
    speech.local = AsyncMock(return_value={"audio": base64.b64encode(b"same-wave").decode(), "warnings": []})
    first = await speech.synthesize("بَعْدَ العَمَل الساعة 9.", "ar")
    assert await speech.synthesize("بعد  العمل الساعة 9.", "ar") == first
    assert speech.local.await_count == 1
    for different in ("بعد الغداء الساعة 9.", "قبل العمل الساعة 9.", "بعد العمل الساعة 10."):
        await speech.synthesize(different, "ar")
    assert speech.local.await_count == 4
    assert not speech.cache_locks


async def test_identical_concurrent_speech_runs_once_and_waiter_cancel_is_isolated(settings, secrets):
    speech = Speech(settings, secrets)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def generate(*args, **kwargs):
        entered.set()
        await finish.wait()
        return {"audio": base64.b64encode(b"same-wave").decode(), "warnings": []}
    speech.local = AsyncMock(side_effect=generate)
    first = asyncio.create_task(speech.synthesize("بعد العمل.", "ar"))
    await entered.wait()
    waiter = asyncio.create_task(speech.synthesize("بَعْدَ العَمَل.", "ar"))
    other = asyncio.create_task(speech.synthesize("بعد العمل.", "ar"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not first.done()
    finish.set()
    assert await first == await other
    assert speech.local.await_count == 1
    assert not speech.cache_locks


async def test_cancelled_generation_releases_the_cache_lock(settings, secrets):
    speech = Speech(settings, secrets)
    entered = asyncio.Event()
    async def generate(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    speech.local = generate
    task = asyncio.create_task(speech.synthesize("بعد العمل.", "ar"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not speech.cache_locks and not speech.audio_cache


async def test_cancel_stops_active_speech_and_rejects_late_stale_requests(tmp_path):
    app = create_app(tmp_path, "test")
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def pending(*args, **kwargs):
        entered.set()
        try: await asyncio.Event().wait()
        finally: stopped.set()
    app.state.speech.synthesize = pending
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", headers={"Authorization": "Bearer test"}) as client:
        request = asyncio.create_task(client.post("/speech/synthesize", json={"text": "مرحبا", "language": "ar", "generation": 1, "local_only": True}))
        await entered.wait()
        assert (await client.post("/speech/cancel", json={"generation": 2})).status_code == 200
        assert stopped.is_set() and (await request).status_code == 409
        assert (await client.post("/speech/synthesize", json={"text": "late", "language": "en", "generation": 1})).status_code == 409


async def test_release_only_unloads_configured_models(settings):
    settings.update({"assistant": {"voice_model": "qwen2.5:7b-instruct"}})
    calls = []
    def handle(request):
        if request.method == "GET": return httpx.Response(200, json={"models": [{"name": x} for x in ["gpt-oss:20b", "qwen2.5:7b-instruct", "unrelated"]]})
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"done": True})
    await GpuCoordinator(settings, httpx.MockTransport(handle)).release()
    assert set(calls) == {"gpt-oss:20b", "qwen2.5:7b-instruct"}


async def test_preload_uses_local_empty_prompt_and_reserves_speech_memory(settings, secrets):
    settings.update({"assistant": {"voice_model": "gpt-oss:20b", "voice_low_latency": True}, "speech": {"tts": {"device": "cuda"}}})
    calls = []
    def handle(request):
        calls.append(request.url.path)
        assert request.url.host == "127.0.0.1" and "authorization" not in request.headers
        if request.url.path == "/api/show": return httpx.Response(200, json={"capabilities": ["completion"]})
        body = json.loads(request.content)
        assert body["prompt"] == "" and body["options"]["num_gpu"] == 16
        assert body["options"]["num_ctx"] == 8192 and body["keep_alive"] == "10m"
        return httpx.Response(200, json={"done": True})
    cfg = voice_config(settings.snapshot())
    provider = Provider(cfg, secrets, httpx.MockTransport(handle))
    await provider.preload()
    assert calls == ["/api/show", "/api/generate"]
    cfg["_voice"] = False
    assert "num_gpu" not in Provider(cfg, secrets).payload([], [])["options"]
    provider = Provider(voice_config(settings.snapshot()), secrets, httpx.MockTransport(lambda _: httpx.Response(200, json={"remote_model": "remote"})))
    with pytest.raises(ProviderError): await provider.preload()


def test_cpu_speech_does_not_force_windows_gpu_layer_limit_on_mac(settings, secrets):
    settings.update({"assistant": {"voice_low_latency": True}, "speech": {"asr": {"device": "cpu"}, "tts": {"device": "cpu"}}})
    cfg = voice_config(settings.snapshot())
    assert "num_gpu" not in Provider(cfg, secrets).payload([], [])["options"]


async def test_voice_priming_is_one_local_token_without_tool_execution(settings, secrets):
    calls = []
    def handle(request):
        calls.append(request.url.path)
        assert request.url.host == "127.0.0.1" and "authorization" not in request.headers
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        body = json.loads(request.content)
        assert body["options"]["num_predict"] == 1 and not body["stream"]
        assert body["messages"] == [{"role": "system", "content": "Public instructions"}, {"role": "user", "content": ""}]
        return httpx.Response(200, json={"message": {"tool_calls": [{"function": {"name": "never_execute"}}]}, "done": True})
    provider = Provider(voice_config(settings.snapshot()), secrets, httpx.MockTransport(handle))
    await provider.preload(messages=[{"role": "system", "content": "Public instructions"}, {"role": "user", "content": ""}])
    assert calls == ["/api/show", "/api/chat"]


async def test_resident_voice_reserves_model_memory_before_loading_speech(tmp_path):
    order = []
    class PrimingProvider:
        def __init__(self, config, secrets):
            assert config["mode"] == "offline" and config["_voice"]
        async def preload(self, messages, tools): order.append("model")
    app = create_app(tmp_path, "test", provider_factory=PrimingProvider)
    app.state.settings.update({"assistant": {"voice_low_latency": True}})
    async def warmup(action):
        assert action == "warmup"
        order.append("speech")
    app.state.speech.local = warmup
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", headers={"Authorization": "Bearer test"}) as client:
        response = await client.post("/voice/prepare", json={"id": "22222222-2222-4222-8222-222222222222"})
        assert response.status_code == 200
    assert order == ["model", "speech"]


async def test_cancel_voice_preparation_and_reject_late_start(tmp_path):
    app = create_app(tmp_path, "test")
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def pending(*args, **kwargs):
        entered.set()
        try: await asyncio.Event().wait()
        finally: stopped.set()
    app.state.speech.local = pending
    body = {"id": "11111111-1111-4111-8111-111111111111"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", headers={"Authorization": "Bearer test"}) as client:
        request = asyncio.create_task(client.post("/voice/prepare", json=body))
        await asyncio.wait_for(entered.wait(), 2)
        assert (await client.patch("/settings", json={"language": "en"})).status_code == 400
        assert (await client.post("/voice/cancel", json=body)).status_code == 200
        assert stopped.is_set() and (await request).status_code == 409
        assert (await client.post("/voice/prepare", json=body)).status_code == 409


async def test_vision_releases_resident_speech_and_models(settings):
    settings.update({"assistant": {"voice_model": "gpt-oss:20b", "voice_low_latency": True}})
    calls = []
    def handle(request):
        if request.method == "GET": return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"done": True})
    gpu = GpuCoordinator(settings, httpx.MockTransport(handle))
    gpu.offload_speech = AsyncMock()
    async with gpu.vision():
        assert gpu.lock.locked()
        gpu.offload_speech.assert_awaited_once()
        assert calls == ["gpt-oss:20b"]
