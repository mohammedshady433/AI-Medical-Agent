"""Opt-in MCP HTTP client. Discovery is explicit; every tool call needs approval."""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import hashlib
import json
import secrets as random_secrets
import time
import httpx
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import NoSuchResource
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from app.core.settings import local_url


def no_remote_schema(uri):
    raise NoSuchResource(ref=uri)


def validate_arguments(schema, arguments):
    if not isinstance(arguments, dict) or len(json.dumps(arguments)) > 16000:
        raise ValueError("Tool arguments must be a JSON object under 16,000 characters.")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, registry=Registry(retrieve=no_remote_schema)).validate(arguments)
    except Exception as exc:
        raise ValueError("Arguments do not match this tool's schema. Remote schema references are not supported.") from exc


@asynccontextmanager
async def connection(server, key, transport=None):
    class BoundedStream(httpx.AsyncByteStream):
        def __init__(self, stream): self.stream = stream
        async def __aiter__(self):
            size = 0
            async for chunk in self.stream:
                size += len(chunk)
                if size > 2_000_000: raise ValueError("MCP response exceeds 2 MB.")
                yield chunk
        async def aclose(self): await self.stream.aclose()
    async def bound_response(response):
        response.stream = BoundedStream(response.stream)
    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False, follow_redirects=False, transport=transport, event_hooks={"response": [bound_response]})
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with streamablehttp_client(server["url"], headers=headers, timeout=15, sse_read_timeout=45,
                                    httpx_client_factory=client_factory) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=40)) as session:
            await session.initialize()
            yield session


class Mcp:
    def __init__(self, settings, secrets, connector=connection):
        self.settings, self.secrets, self.connector = settings, secrets, connector
        self.catalogs, self.pending = {}, {}

    def server(self, sid):
        cfg = self.settings.snapshot()
        server = next((s for s in cfg["assistant"]["mcp_servers"] if s["id"] == sid and s["enabled"]), None)
        if not server: raise ValueError("This MCP server is not enabled.")
        if cfg["mode"] == "offline" and not local_url(server["url"]):
            raise ValueError("Remote MCP is blocked in Offline mode. Select Online explicitly or use a loopback MCP server.")
        return server

    def catalog(self, sid):
        server = self.server(sid)
        cached = self.catalogs.get(sid)
        if not cached or cached["url"] != server["url"]: raise ValueError("Connect to this server to load its tools first.")
        return cached["tools"]

    async def connect(self, sid):
        server = self.server(sid)
        self.catalogs.pop(sid, None)
        try:
            async with asyncio.timeout(45):
                async with self.connector(server, self.secrets.get("mcp:" + sid)) as session:
                    result = await session.list_tools()
            tools = []
            for tool in result.tools[:100]:
                schema = tool.inputSchema
                if len(json.dumps(schema)) > 6000: continue
                tools.append({"name": tool.name[:128], "description": (tool.description or "")[:600], "schema": schema})
            self.catalogs[sid] = {"url": server["url"], "tools": tools}
            return {"tools": tools, "truncated": bool(result.nextCursor) or len(result.tools) > 100}
        except asyncio.CancelledError: raise
        except Exception as exc:
            # Never forward a remote error containing credentials or raw headers.
            raise ValueError("MCP connection failed. Check the endpoint, key and server availability (" + type(exc).__name__ + ").") from None

    def definitions(self, local_only=False):
        definitions, mapping, size = [], {}, 0
        for server in self.settings.snapshot()["assistant"]["mcp_servers"]:
            if not server["enabled"] or not server["model_access"]: continue
            if local_only and not local_url(server["url"]): continue
            try: catalog = self.catalog(server["id"])
            except ValueError: continue
            for tool in catalog:
                if tool["name"] not in server["allowed_tools"]: continue
                name = "mcp_" + hashlib.sha256((server["id"] + ":" + tool["name"]).encode()).hexdigest()[:20]
                item = {"type": "function", "function": {"name": name,
                    "description": "External tool from " + server["name"] + "; requires user confirmation. " + tool["description"], "parameters": tool["schema"]}}
                size += len(json.dumps(item))
                if len(definitions) >= 12 or size > 24000: return definitions, mapping
                definitions.append(item); mapping[name] = (server["id"], tool["name"])
        return definitions, mapping

    def prepare(self, sid, name, arguments, model=False, local_only=False):
        server = self.server(sid)
        if local_only and not local_url(server["url"]): raise ValueError("Voice conversations only use local MCP services.")
        tool = next((t for t in self.catalog(sid) if t["name"] == name), None)
        if not tool: raise ValueError("Unknown MCP tool. Refresh the server catalog.")
        if model and (not server["model_access"] or name not in server["allowed_tools"]): raise ValueError("The tool is not enabled for this assistant.")
        if isinstance(arguments, str):
            try: arguments = json.loads(arguments)
            except ValueError: raise ValueError("Tool arguments must be valid JSON.") from None
        validate_arguments(tool["schema"], arguments)
        self.expire()
        if len(self.pending) >= 16: raise ValueError("Too many pending tool approvals.")
        rid = random_secrets.token_hex(16)
        preview = {"id": rid, "label": server["name"] + " → " + name, "server": sid, "url": server["url"], "tool": name, "arguments": arguments}
        self.pending[rid] = {"preview": preview, "expires": time.monotonic() + 120, "schema": tool["schema"], "model": model,
                             "future": asyncio.get_running_loop().create_future() if model else None}
        return preview

    def expire(self):
        for rid, item in list(self.pending.items()):
            if item["expires"] < time.monotonic():
                if item["future"] and not item["future"].done(): item["future"].set_result(False)
                self.pending.pop(rid, None)

    async def confirm(self, rid, approve):
        self.expire(); item = self.pending.get(rid)
        if not item: raise ValueError("This approval expired or was already used.")
        if item["future"]:
            if item["future"].done(): raise ValueError("This approval was already answered.")
            item["future"].set_result(approve)
            return {"approved": approve}
        self.pending.pop(rid)
        return await self.invoke(item) if approve else {"cancelled": True}

    async def wait(self, rid, cancel=None):
        item = self.pending.get(rid)
        if not item: raise ValueError("Approval expired.")
        future = item["future"]
        try:
            while not future.done():
                if cancel and cancel.is_set(): raise asyncio.CancelledError()
                if time.monotonic() >= item["expires"]: return {"error": "User approval timed out."}
                await asyncio.wait({future}, timeout=0.2)
            if not future.result(): return {"cancelled": True}
            if cancel and cancel.is_set(): raise asyncio.CancelledError()
            return await self.invoke(item)
        finally:
            self.pending.pop(rid, None)
            if not future.done(): future.cancel()

    async def invoke(self, item):
        preview = item["preview"]; server = self.server(preview["server"])
        if server["url"] != preview["url"]: raise ValueError("Server changed after this request. Review a new request.")
        if item["model"] and (not server["model_access"] or preview["tool"] not in server["allowed_tools"]):
            raise ValueError("Tool permission changed before execution.")
        validate_arguments(item["schema"], preview["arguments"])
        try:
            async with asyncio.timeout(45):
                async with self.connector(server, self.secrets.get("mcp:" + server["id"])) as session:
                    result = await session.call_tool(preview["tool"], preview["arguments"])
            text = "\n".join(part.text for part in result.content if part.type == "text")[:12000]
            return {"text": text, "is_error": bool(result.isError), "untrusted_external_content": True}
        except asyncio.CancelledError: raise
        except Exception as exc:
            raise ValueError("MCP tool failed (" + type(exc).__name__ + "). It was not retried; check the external service before retrying a change.") from None
