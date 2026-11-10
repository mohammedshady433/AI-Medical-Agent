"""The ten requested mode acceptance scenarios using controlled HTTP, not live provider proof."""

import json
import httpx
import pytest
from app.providers.client import Provider, ProviderError


async def request(settings, secrets, handler):
    config = settings.snapshot()
    return await Provider(config, secrets, httpx.MockTransport(handler)).test()


def good(request):
    if request.url.path == "/api/show":
        return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
    return httpx.Response(200, text=json.dumps({"message": {"content": "OK"}, "done": True}) + "\n")


async def test_01_online_routes_actual_120b(settings, secrets):
    def handler(req):
        assert str(req.url) == "https://ollama.com/api/chat"
        assert json.loads(req.content)["model"] == "gpt-oss:120b"
        return good(req)

    assert (await request(settings, secrets, handler))["mode"] == "online"


async def test_02_offline_never_calls_cloud(settings, secrets):
    settings.update({"mode": "offline"})

    def handler(req):
        assert req.url.host == "127.0.0.1"
        assert json.loads(req.content)["model"] == "gpt-oss:20b"
        assert "Authorization" not in req.headers
        return good(req)

    assert (await request(settings, secrets, handler))["mode"] == "offline"


def disconnected(req):
    raise httpx.ConnectError("Connection unavailable", request=req)


async def test_03_disconnect_recommends_offline(settings, secrets):
    with pytest.raises(ProviderError) as error:
        await request(settings, secrets, disconnected)
    assert error.value.kind == "connectivity" and error.value.suggest_offline


async def test_04_ignored_recommendation_keeps_online(settings, secrets):
    for _ in range(2):
        with pytest.raises(ProviderError):
            await request(settings, secrets, disconnected)
    assert settings.snapshot()["mode"] == "online"
    assert settings.snapshot()["models"]["online"]["id"] == "gpt-oss:120b"


async def test_05_manual_offline_works(settings, secrets):
    settings.update({"mode": "offline"})
    assert (await request(settings, secrets, good))["model"] == "gpt-oss:20b"


async def test_06_restored_connection_keeps_offline(settings, secrets):
    settings.update({"mode": "offline"})
    await request(settings, secrets, good)
    assert settings.snapshot()["mode"] == "offline"


async def test_07_manual_online_resumes(settings, secrets):
    settings.update({"mode": "offline"})
    settings.update({"mode": "online"})
    assert (await request(settings, secrets, good))["model"] == "gpt-oss:120b"


async def test_08_missing_model_does_not_download(settings, secrets):
    settings.update({"mode": "offline"})
    paths = []

    def handler(req):
        paths.append(req.url.path)
        return httpx.Response(404)

    with pytest.raises(ProviderError) as error:
        await request(settings, secrets, handler)
    assert error.value.kind == "model" and paths == ["/api/show"]


async def test_local_alias_cannot_proxy_cloud(settings, secrets):
    settings.update({"mode": "offline"})
    paths = []

    def handler(req):
        paths.append(req.url.path)
        return httpx.Response(200, json={"remote_host": "https://ollama.com", "remote_model": "gpt-oss:120b"})

    with pytest.raises(ProviderError) as error:
        await request(settings, secrets, handler)
    assert error.value.kind == "privacy" and paths == ["/api/show"]


async def test_09_expired_key_is_auth_error(settings, secrets):
    with pytest.raises(ProviderError) as error:
        await request(settings, secrets, lambda req: httpx.Response(401))
    assert error.value.kind == "authentication" and not error.value.suggest_offline


async def test_10_provider_failure_is_not_internet_claim(settings, secrets):
    with pytest.raises(ProviderError) as error:
        await request(settings, secrets, lambda req: httpx.Response(503))
    assert error.value.kind == "provider" and settings.snapshot()["mode"] == "online"


async def test_redirects_do_not_leak_credentials(settings, secrets):
    with pytest.raises(ProviderError):
        await request(
            settings,
            secrets,
            lambda req: httpx.Response(302, headers={"Location": "https://untrusted.example"}),
        )


async def test_reasoning_never_shown(settings, secrets):
    def handler(req):
        return httpx.Response(
            200,
            text=json.dumps({"message": {"thinking": "private reasoning", "content": "Visible answer"}})
            + "\n",
        )

    result = [
        e
        async for e in Provider(settings.snapshot(), secrets, httpx.MockTransport(handler)).stream(
            [{"role": "user", "content": "Hi"}]
        )
    ]
    assert result == [{"type": "token", "text": "Visible answer"}]


async def test_compatible_tool_stream_assembled(settings, secrets):
    settings.update({"models": {"online": {"provider": "compatible", "id": "user-chosen-model"}}})

    def handler(req):
        assert req.url.path == "/v1/chat/completions"
        parts = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call1",
                                    "function": {"name": "calculator", "arguments": '{"expression":'},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"2+3"}'}}]}}]},
        ]
        return httpx.Response(
            200, text="\n\n".join("data: " + json.dumps(p) for p in parts) + "\n\ndata: [DONE]\n"
        )

    result = [
        e async for e in Provider(settings.snapshot(), secrets, httpx.MockTransport(handler)).stream([])
    ]
    assert json.loads(result[0]["call"]["function"]["arguments"]) == {"expression": "2+3"}
