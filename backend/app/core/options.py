from pydantic import BaseModel, ConfigDict, Field
from urllib.parse import urlsplit
from typing import Literal


def resident_voice(config):
    selected = config["assistant"]["voice_model"] or config["models"]["offline"]["id"]
    return config["assistant"]["voice_low_latency"] and selected in {"qwen2.5:7b-instruct", "gpt-oss:20b"}


class ResearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = False
    max_searches: int = Field(default=3, ge=1, le=6)
    max_sources: int = Field(default=5, ge=1, le=10)
    max_model_calls: int = Field(default=3, ge=1, le=6)
    page_chars: int = Field(default=6000, ge=1000, le=12000)
    output_tokens: int = Field(default=1200, ge=128, le=3000)
    planner_model: str = Field(default="", max_length=160)
    worker_model: str = Field(default="", max_length=160)


class VisionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = False
    model: str = Field(default="", max_length=160)


class McpServer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    name: str = Field(min_length=1, max_length=80)
    url: str = Field(min_length=1, max_length=500)
    enabled: bool = False
    model_access: bool = False
    allowed_tools: list[str] = Field(default_factory=list, max_length=12)


class AssistantOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    desktop_actions: bool = True
    voice_activation: bool = False
    wake_phrase: str = Field(default="يا إي إم إس", min_length=2, max_length=50, pattern=r"^[^\r\n]+$")
    barge_in: bool = True
    voice_detection_version: int = Field(default=1, ge=1, le=1)
    follow_up_seconds: int = Field(default=20, ge=5, le=60)
    hotkeys: bool = True
    voice_model: str = Field(default="", max_length=160)
    voice_mode: Literal["offline", "online"] = "offline"
    voice_low_latency: bool = False
    voice_pause_ms: int = Field(default=500, ge=400, le=1500)
    workspace: list[str] = Field(default_factory=list, max_length=8)
    research: ResearchOptions = Field(default_factory=ResearchOptions)
    vision: VisionOptions = Field(default_factory=VisionOptions)
    mcp_servers: list[McpServer] = Field(default_factory=list, max_length=8)
    clipboard_history: bool = False
    weather_city: str = Field(default="", max_length=100)
    weather_enabled: bool = False
    first_run_completed: bool = False


def validate_options(value):
    options = AssistantOptions.model_validate(value)
    if any(not name.strip() or len(name) > 160 for name in options.workspace):
        raise ValueError("Workspace entries must be installed app names, up to 160 characters.")
    ids = set()
    for server in options.mcp_servers:
        url = urlsplit(server.url)
        local = url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"}
        if not (local or (url.scheme == "https" and url.hostname)) or url.username or url.password or url.query or url.fragment:
            raise ValueError("MCP endpoints require HTTPS or loopback HTTP, with keys stored separately.")
        if server.id in ids or any(not tool or len(tool) > 128 for tool in server.allowed_tools):
            raise ValueError("Use unique MCP server IDs and valid tool names.")
        ids.add(server.id)
    if "cloud" in options.vision.model.casefold():
        raise ValueError("Screen vision requires an installed local vision model, not a cloud tag.")
    if "cloud" in options.voice_model.casefold():
        raise ValueError("Voice conversations require an installed local model.")
