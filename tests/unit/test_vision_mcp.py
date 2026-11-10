import asyncio
import base64
from functools import partial
from io import BytesIO
import json
import httpx
from PIL import Image
import pytest
from app.extensions.vision import Vision, prepare_image
from app.extensions.mcp import Mcp, connection, validate_arguments


def png():
    data = BytesIO(); Image.new("RGB", (2000, 1000), "white").save(data, "PNG")
    return base64.b64encode(data.getvalue()).decode()


async def test_vision_local_resize_and_language(settings):
    settings.update({"assistant": {"vision": {"enabled": True, "model": "test-vision"}}})
    requests = []
    def handle(request):
        assert request.url.host in {"127.0.0.1", "localhost"}
        data = json.loads(request.content); requests.append(data)
        if request.url.path == "/api/show": return httpx.Response(200, json={"capabilities": ["vision"]})
        assert "Egyptian Arabic" in data["messages"][0]["content"]
        image = Image.open(BytesIO(base64.b64decode(data["messages"][1]["images"][0])))
        assert image.size == (1280, 640) and data["keep_alive"] == 0
        return httpx.Response(200, json={"message": {"content": "صورة بيضاء"}})
    result = await Vision(settings, transport=httpx.MockTransport(handle)).analyze(png(), "اشرح الصورة")
    assert result["local"] and result["text"] == "صورة بيضاء" and len(requests) == 2


@pytest.mark.parametrize("info", [{"capabilities": []}, {"capabilities": ["vision"], "remote_host": "https://cloud.invalid"}])
async def test_vision_rejects_nonvision_cloud_alias(settings, info):
    settings.update({"assistant": {"vision": {"enabled": True, "model": "test-vision"}}})
    def handle(request):
        assert request.url.path == "/api/show"
        return httpx.Response(200, json=info)
    with pytest.raises(ValueError): await Vision(settings, transport=httpx.MockTransport(handle)).analyze(png(), "describe")


def test_invalid_image():
    for value in ["bad base64", base64.b64encode(b"not an image").decode(), "x" * 12000001]:
        with pytest.raises(ValueError): prepare_image(value)


def mcp_config(settings, remote=False):
    row = {"id": "test", "name": "Test service", "url": "https://example.com/mcp" if remote else "http://127.0.0.1:8765/mcp", "enabled": True, "model_access": True, "allowed_tools": ["echo"]}
    settings.update({"assistant": {"mcp_servers": [row]}})


def fake_protocol(calls):
    def handle(request):
        if request.method == "GET": return httpx.Response(405)
        if request.method == "DELETE": return httpx.Response(200)
        body = json.loads(request.content)
        method = body["method"]
        if "id" not in body: return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo Arabic text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}]}
        elif method == "tools/call":
            calls.append(body["params"])
            result = {"content": [{"type": "text", "text": body["params"]["arguments"]["text"]}], "isError": False}
        else: raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
    return httpx.MockTransport(handle)


async def test_official_mcp_protocol_only_executes_after_one_use_approval(settings, secrets):
    mcp_config(settings); calls = []
    mcp = Mcp(settings, secrets, partial(connection, transport=fake_protocol(calls)))
    assert mcp.definitions()[0] == []
    assert (await mcp.connect("test"))["tools"][0]["name"] == "echo"
    assert len(mcp.definitions()[0]) == 1
    request = mcp.prepare("test", "echo", {"text": "أهلاً بالعربي"})
    assert not calls
    result = await mcp.confirm(request["id"], True)
    assert result["text"] == "أهلاً بالعربي" and len(calls) == 1
    with pytest.raises(ValueError): await mcp.confirm(request["id"], True)
    request = mcp.prepare("test", "echo", {"text": "do not run"})
    await mcp.confirm(request["id"], False)
    assert len(calls) == 1


async def test_model_mcp_cancellation_and_changed_permissions(settings, secrets):
    mcp_config(settings); calls = []; mcp = Mcp(settings, secrets, partial(connection, transport=fake_protocol(calls)))
    await mcp.connect("test")
    request = mcp.prepare("test", "echo", {"text": "cancel"}, model=True)
    cancel = asyncio.Event(); cancel.set()
    with pytest.raises(asyncio.CancelledError): await mcp.wait(request["id"], cancel)
    assert not calls and not mcp.pending
    request = mcp.prepare("test", "echo", {"text": "approved"}, model=True)
    await mcp.confirm(request["id"], True)
    result = await mcp.wait(request["id"])
    assert result["text"] == "approved" and len(calls) == 1
    request = mcp.prepare("test", "echo", {"text": "changed"})
    servers = settings.snapshot()["assistant"]["mcp_servers"]; servers[0]["url"] = "http://127.0.0.1:9999/mcp"
    settings.update({"assistant": {"mcp_servers": servers}})
    with pytest.raises(ValueError): await mcp.confirm(request["id"], True)
    assert len(calls) == 1


async def test_offline_remote_mcp_and_external_schema_rejected(settings, secrets):
    mcp_config(settings, remote=True); settings.update({"mode": "offline"})
    with pytest.raises(ValueError): Mcp(settings, secrets).server("test")
    with pytest.raises(ValueError): validate_arguments({"$ref": "https://example.com/schema.json"}, {})
    with pytest.raises(ValueError): validate_arguments({"type": "object", "required": ["text"]}, {})
