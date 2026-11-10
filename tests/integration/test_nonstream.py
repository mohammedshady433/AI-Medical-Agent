import httpx
import json
import pytest
from app.providers.client import Provider


@pytest.mark.parametrize("adapter", ["ollama_cloud", "compatible"])
async def test_nonstreaming_setting_applies(settings, secrets, adapter):
    settings.update({"models": {"online": {"provider": adapter, "streaming": False}}})

    def handler(request):
        assert json.loads(request.content)["stream"] is False
        message = {
            "content": "A complete answer.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "calculator", "arguments": {"expression": "2+3"}}}
            ],
        }
        return httpx.Response(
            200,
            json={"message": message} if adapter == "ollama_cloud" else {"choices": [{"message": message}]},
        )

    events = [
        e async for e in Provider(settings.snapshot(), secrets, httpx.MockTransport(handler)).stream([])
    ]
    assert events[0] == {"type": "token", "text": "A complete answer."}
    assert events[1]["call"]["function"]["name"] == "calculator"
