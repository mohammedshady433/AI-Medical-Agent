import asyncio
import json
from unittest.mock import AsyncMock
import httpx
import pytest
from app.agents.assistant import Assistant, voice_config
from app.main import create_app
from app.providers.client import Provider
from app.speech.service import Speech


def test_online_voice_requires_explicit_choice_and_preserves_local_model(settings):
    assert voice_config(settings.snapshot())["mode"] == "offline"
    settings.update({"mode": "offline", "assistant": {"voice_mode": "online"}})
    cfg = voice_config(settings.snapshot())
    assert cfg["mode"] == "online"
    assert cfg["models"]["offline"]["id"] == "gpt-oss:20b"
    assert settings.snapshot()["mode"] == "offline"
    with pytest.raises(ValueError): settings.update({"assistant": {"voice_mode": "automatic"}})


async def test_cloud_voice_streaming_is_short_and_does_not_share_documents(settings, secrets, database, library):
    settings.update({"assistant": {"voice_mode": "online"}})
    seen = []
    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["stream"] and body["options"]["num_predict"] <= 512
        assert "search_documents" not in json.dumps(body["tools"])
        assert "PRIVATE" not in json.dumps(body["messages"])
        return httpx.Response(200, text='{"message":{"content":"A complete answer."},"done":true}\n')
    secrets.get = lambda *args: "fixture-key"
    factory = lambda cfg, keys: Provider(cfg, keys, httpx.MockTransport(respond))
    chat = database.new_chat()
    database.add_message(chat, "assistant", "PRIVATE FILE", {"document_context": True})
    assistant = Assistant(settings, secrets, database, library, factory)
    await assistant.prepare_voice(voice_config(settings.snapshot()))
    assert not seen, "Online preparation must not make paid warmup calls"
    events = [e async for e in assistant.reply(chat, "Say hello", asyncio.Event(), voice=True)]
    assert events[-1]["content"] == "A complete answer."
    assert events[-1]["meta"]["mode"] == "online" and len(seen) == 1


async def test_online_voice_missing_key_fails_before_loading_models(tmp_path):
    app = create_app(tmp_path, "fixture")
    app.state.settings.update({"assistant": {"voice_mode": "online"}})
    # Isolated test workspace has no real cloud credentials.
    from app.core.secrets import SecretStore
    from unittest.mock import patch
    app.state.speech.local = AsyncMock()
    with patch.object(SecretStore, "get", return_value=""):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", headers={"Authorization": "Bearer fixture"}) as client:
            result = await client.post("/voice/prepare", json={"id": "11111111-1111-4111-8111-111111111111"})
    assert result.status_code == 400 and "API key" in result.text
    app.state.speech.local.assert_not_awaited()


async def test_offline_typing_never_uses_azure_even_if_online_voice_is_enabled(settings, secrets):
    settings.update({"mode": "offline", "assistant": {"voice_mode": "online"}, "speech": {"tts": {"provider": "azure"}}})
    speech = Speech(settings, secrets)
    with pytest.raises(ValueError, match="online-only"):
        await speech.synthesize("Private typed answer", "en")
    with pytest.raises(ValueError, match="Offline voice"):
        await speech.synthesize("Private spoken answer", "en", local_only=True, voice_conversation=True)


async def test_azure_connection_reused_and_closed_with_fresh_keys_per_chunk(settings, secrets):
    import io
    import wave
    settings.update({"mode": "offline", "assistant": {"voice_mode": "online"}, "speech": {"tts": {"provider": "azure"}}})
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(b'\0\0' * 100)
    keys = iter(["first-key", "second-key"])
    secrets.get = lambda *_: next(keys)
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=target.getvalue())
    speech = Speech(settings, secrets)
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    speech.cloud_client = client
    for text in ["One.", "Two."]:
        audio, _ = await speech.synthesize(text, "en", voice_conversation=True)
        assert audio == target.getvalue() and speech.cloud_client is client
    assert [r.headers['Ocp-Apim-Subscription-Key'] for r in requests] == ["first-key", "second-key"]
    await speech.close()
    assert client.is_closed
