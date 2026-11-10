"""Streaming adapters; an immutable request snapshot prevents mode changes mid-turn."""

from __future__ import annotations
import json
import httpx
from app.core.settings import local_url
from app.core.options import resident_voice


class ProviderError(Exception):
    def __init__(self, kind, message, suggest_offline=False):
        super().__init__(message)
        self.kind, self.message, self.suggest_offline = kind, message, suggest_offline

    def event(self):
        return {
            "type": "error",
            "kind": self.kind,
            "message": self.message,
            "suggest_offline": self.suggest_offline,
        }


def http_error(status, mode):
    if status in (401, 403):
        return ProviderError(
            "authentication",
            "The provider rejected your credentials or access permissions. Check the key and account in Settings.",
        )
    if status == 404:
        return ProviderError(
            "model",
            "The selected local model is not installed. Open Settings → Offline → Download local model, or select ONLINE in the header to use your cloud API key."
            if mode == "offline"
            else "The cloud endpoint or selected model was not found. Check the provider endpoint and exact model identifier in Settings.",
        )
    if status == 429:
        return ProviderError(
            "rate_limit",
            "The provider rate or account limit was reached. Wait before retrying or check your plan.",
        )
    if status >= 500:
        return ProviderError(
            "provider",
            "The provider is temporarily unavailable. Retry later or manually select Offline.",
            mode == "online",
        )
    return ProviderError(
        "request",
        f"The provider rejected the request (HTTP {status}). Check its supported settings and model.",
    )


class Provider:
    def __init__(self, config, secrets, transport=None):
        self.config, self.mode = config, config["mode"]
        self.model = config["models"][self.mode]
        self.provider = config["providers"][self.model["provider"]]
        self.base = self.provider["endpoint"].rstrip("/")
        self.adapter = self.provider["adapter"]
        if self.mode == "offline" and not local_url(self.base):
            raise ProviderError("privacy", "Offline requests require a loopback endpoint.")
        self.key = (
            secrets.get(self.model["provider"], self.provider.get("secret_env", ""))
            if self.mode == "online"
            else ""
        )
        self.transport = transport

    def client(self):
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.model["timeout"], connect=10),
            trust_env=False,
            follow_redirects=False,
            transport=self.transport,
        )

    def headers(self):
        if self.mode == "online" and not self.key:
            raise ProviderError(
                "authentication", "Add your provider API key in Settings, then test the connection."
            )
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def payload(self, messages, tools):
        m = self.model
        if self.adapter == "ollama":
            data = {
                "model": m["id"],
                "messages": messages,
                "stream": m.get("streaming", True),
                "options": {
                    "num_ctx": m["context_window"],
                    "num_predict": m["max_response_tokens"],
                    "temperature": m["temperature"],
                },
                "keep_alive": "10m",
            }
            if m.get("reasoning_effort"):
                data["think"] = m["reasoning_effort"]
            cuda_speech = any(self.config["speech"][stage]["device"] == "cuda" for stage in ("asr", "tts"))
            if self.mode == "offline" and self.config.get("_voice") and cuda_speech and resident_voice(self.config) and m["id"] == "gpt-oss:20b":
                # Keep space for local ASR + the approved Arabic voice on 16 GB GPUs.
                # Measured faster than unloading/reloading all 20B weights each turn.
                data["options"]["num_gpu"] = 16
        else:
            data = {
                "model": m["id"],
                "messages": messages,
                "stream": m.get("streaming", True),
                "max_tokens": m["max_response_tokens"],
                "temperature": m["temperature"],
            }
        if tools:
            data["tools"] = tools
        return data

    async def preload(self, messages=None, tools=()):
        """Load an installed local model without generating a disposable answer."""
        if self.mode != "offline" or self.adapter != "ollama":
            raise ProviderError("privacy", "Voice preparation requires local Ollama.")
        async with self.client() as client:
            show = await client.post(self.base + "/api/show", json={"model": self.model["id"]})
            if show.status_code >= 300:
                raise http_error(show.status_code, self.mode)
            info = show.json()
            if info.get("remote_host") or info.get("remote_model") or "cloud" in info.get("capabilities", []):
                raise ProviderError("privacy", "A cloud model cannot be used for local voice.")
            options = self.payload([], [])["options"]
            if messages is None:
                result = await client.post(self.base + "/api/generate", json={"model": self.model["id"], "prompt": "", "stream": False, "keep_alive": "10m", "options": options})
            else:
                # One discarded LOCAL token warms prompt evaluation and decoder kernels.
                # This never executes returned tools or enters conversation history.
                payload = self.payload(messages, tools)
                payload["stream"] = False
                payload["options"].update(num_predict=1, temperature=0)
                result = await client.post(self.base + "/api/chat", json=payload)
            if result.status_code >= 300:
                raise http_error(result.status_code, self.mode)
            if result.json().get("error"):
                raise ProviderError("provider", "The local model could not be prepared.")

    async def stream(self, messages, tools=()):
        path = "/api/chat" if self.adapter == "ollama" else "/chat/completions"
        calls = {}
        try:
            async with self.client() as client:
                if self.mode == "offline":
                    show = await client.post(self.base + "/api/show", json={"model": self.model["id"]})
                    if show.status_code >= 300:
                        raise http_error(show.status_code, self.mode)
                    details = show.json()
                    if (
                        details.get("remote_host")
                        or details.get("remote_model")
                        or "cloud" in details.get("capabilities", [])
                    ):
                        raise ProviderError(
                            "privacy",
                            "This local tag points to a cloud model. Choose an installed local model for Offline mode.",
                        )
                payload = self.payload(messages, tools)
                if not payload["stream"]:
                    response = await client.post(self.base + path, headers=self.headers(), json=payload)
                    if response.status_code >= 300:
                        raise http_error(response.status_code, self.mode)
                    try:
                        obj = response.json()
                        if obj.get("error"):
                            raise ProviderError("provider", "The provider could not complete inference.")
                        message = (
                            obj.get("message", {})
                            if self.adapter == "ollama"
                            else (obj.get("choices") or [{}])[0].get("message", {})
                        )
                        if message.get("content"):
                            yield {"type": "token", "text": message["content"]}
                        for call in message.get("tool_calls") or []:
                            yield {"type": "tool_call", "call": call}
                    except (ValueError, AttributeError, TypeError):
                        raise ProviderError("protocol", "The provider returned invalid response data.")
                    return
                async with client.stream(
                    "POST", self.base + path, headers=self.headers(), json=payload
                ) as response:
                    if response.status_code >= 300:
                        raise http_error(response.status_code, self.mode)
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if self.adapter != "ollama":
                            if not line.startswith("data:"):
                                continue
                            line = line[5:].strip()
                            if line == "[DONE]":
                                break
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            raise ProviderError("protocol", "The provider returned invalid streaming data.")
                        if obj.get("error"):
                            raise ProviderError(
                                "provider",
                                "The provider could not complete inference. Check the model and server logs.",
                            )
                        delta = (
                            obj.get("message", {})
                            if self.adapter == "ollama"
                            else (obj.get("choices") or [{}])[0].get("delta", {})
                        )
                        if delta.get("content"):
                            yield {"type": "token", "text": delta["content"]}
                        for i, call in enumerate(delta.get("tool_calls") or []):
                            if self.adapter == "ollama":
                                yield {"type": "tool_call", "call": call}
                            else:
                                index = call.get("index", i)
                                target = calls.setdefault(
                                    index,
                                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                                )
                                if call.get("id"):
                                    target["id"] = call["id"]
                                fn = call.get("function", {})
                                for field in ("name", "arguments"):
                                    target["function"][field] += fn.get(field, "")
                    for call in calls.values():
                        yield {"type": "tool_call", "call": call}
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # A failed provider connection cannot prove that the whole internet is down.
            message = (
                "Cannot reach the cloud provider. Check your connection or select Offline manually."
                if self.mode == "online"
                else "Cannot reach local Ollama. Start Ollama and install the selected model in Settings."
            )
            raise ProviderError(
                "connectivity" if self.mode == "online" else "local_unavailable",
                message,
                self.mode == "online",
            )
        except httpx.TimeoutException:
            raise ProviderError(
                "timeout",
                "The selected model took too long to respond. Try again or reduce the context size.",
                self.mode == "online",
            )
        except httpx.HTTPError:
            raise ProviderError(
                "transport",
                "The provider connection ended unexpectedly. Retry the request.",
                self.mode == "online",
            )

    async def models(self):
        path = "/api/tags" if self.adapter == "ollama" else "/models"
        try:
            async with self.client() as client:
                response = await client.get(self.base + path, headers=self.headers())
                if response.status_code >= 300:
                    raise http_error(response.status_code, self.mode)
                data = response.json()
                return [x.get("name", x.get("id", "")) for x in data.get("models", data.get("data", []))]
        except httpx.HTTPError:
            raise ProviderError(
                "connectivity" if self.mode == "online" else "local_unavailable",
                "Cannot reach the selected provider.",
                self.mode == "online",
            )

    async def test(self):
        parts = []
        async for event in self.stream([{"role": "user", "content": "Reply with OK only."}]):
            if event["type"] == "token":
                parts.append(event["text"])
        if not "".join(parts).strip():
            raise ProviderError(
                "empty", "The model returned no visible answer. It may need a larger response token limit."
            )
        return {"ok": True, "model": self.model["id"], "mode": self.mode}
