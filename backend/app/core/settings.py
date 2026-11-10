"""Versioned defaults plus atomic per-user runtime overrides. Secrets live elsewhere."""

from __future__ import annotations
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
import yaml
from app.core.options import validate_options
from .platforms import data_directory

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
DATA = Path(
    os.environ.get(
        "EMS_DATA_DIR", str(data_directory())
    )
)
LOCK = threading.RLock()


def replace_settings_file(source, target):
    """Windows readers/scanners may briefly deny replacement; keep the write atomic."""
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * 2**attempt)


def local_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "http"
        and parsed.hostname in ("127.0.0.1", "localhost", "::1")
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def validate(config: dict) -> None:
    validate_options(config["assistant"])
    if config["mode"] not in ("online", "offline"):
        raise ValueError("Mode must be online or offline.")
    if config["language"] not in ("ar", "en"):
        raise ValueError("Language must be ar or en.")
    for key, provider in config["providers"].items():
        url = urlparse(provider["endpoint"])
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Provider endpoints must not contain credentials, queries, or fragments.")
        if provider["mode"] == "offline" and not local_url(provider["endpoint"]):
            raise ValueError("Offline providers must use a loopback HTTP endpoint.")
        if provider["mode"] == "offline" and provider["adapter"] != "ollama":
            raise ValueError(
                "Offline providers currently require the Ollama adapter so local model residency can be checked."
            )
        if provider["mode"] == "online" and (url.scheme != "https" or not url.hostname):
            raise ValueError("Online providers require HTTPS.")
        if provider["adapter"] not in ("ollama", "openai_compatible"):
            raise ValueError("Unknown provider adapter.")
    for mode in ("online", "offline"):
        model = config["models"][mode]
        provider = config["providers"].get(model["provider"])
        if not provider or provider["mode"] != mode:
            raise ValueError("The model provider does not match its operating mode.")
        if not model["id"].strip() or len(model["id"]) > 160:
            raise ValueError("Enter a model identifier.")
        if mode == "offline" and ("cloud" in model["id"].lower()):
            raise ValueError("Cloud model tags cannot run in Offline mode.")
        if (
            mode == "offline"
            and "120b" in model["id"].lower()
            and not config.get("experimental_local_120b_accepted")
        ):
            raise ValueError("Local 120B requires explicit hardware-warning acceptance in Settings.")
        if not 512 <= model["context_window"] <= 131072 or not 64 <= model["max_response_tokens"] <= 16384:
            raise ValueError("Invalid context or response token limit.")
        if not 5 <= model["timeout"] <= 600 or not 0 <= model["temperature"] <= 2:
            raise ValueError("Invalid timeout or temperature.")
    if not local_url(config["documents"]["embedding"]["endpoint"]):
        raise ValueError("Embeddings must run locally in both modes.")
    if config["speech"]["tts"]["provider"] not in ("chatterbox", "kemetone", "azure", "none"):
        raise ValueError("Unknown speech provider.")
    if config["speech"]["tts"].get("chatterbox_version", "v3") not in ("v2", "v3"):
        raise ValueError("Chatterbox version must be v2 or v3.")
    if config["speech"]["tts"]["provider"] == "chatterbox" and config["speech"]["tts"].get("voice", "builtin") not in ("builtin", "egyptian"):
        raise ValueError("Choose the Egyptian or default multilingual Arabic voice.")
    if any(config["speech"][stage]["device"] not in ("cpu", "cuda") for stage in ("asr", "tts")):
        raise ValueError("Speech device must be CPU or CUDA.")
    if not 0.5 <= config["speech"]["tts"]["speed"] <= 2 or not 0 <= config["speech"]["tts"]["volume"] <= 1:
        raise ValueError("Invalid voice speed or volume.")


class Settings:
    def __init__(self, data: Path = DATA, root: Path = ROOT):
        self.data, self.root = Path(data), Path(root)
        self.data.mkdir(parents=True, exist_ok=True)
        self.path = self.data / "settings.json"
        self.config = self.defaults()
        self.reload()
        validate(self.config)

    def reload(self):
        """Read saved settings directly; do not silently treat an unreadable file as a new install."""
        with LOCK:
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                return False
            candidate = self.merge(self.defaults(), stored)
            # Upgrade the old hidden/off RMS interruption switch once. Explicit
            # preferences saved by the new Settings UI carry this marker.
            if "voice_detection_version" not in stored.get("assistant", {}):
                candidate["assistant"]["barge_in"] = True
                if candidate["assistant"]["voice_pause_ms"] == 700:
                    candidate["assistant"]["voice_pause_ms"] = 500
            validate(candidate)
            changed = candidate != self.config
            self.config = candidate
            return changed

    def defaults(self):
        def read(name):
            return yaml.safe_load((self.root / "config" / (name + ".yaml")).read_text(encoding="utf8"))

        app = read("app")
        return {
            "schema_version": app["schema_version"],
            "mode": app["default_mode"],
            "language": app["language"],
            "privacy": app["privacy"],
            "models": read("models"),
            "providers": read("providers"),
            "speech": read("speech"),
            "documents": read("documents"),
            "assistant": read("assistant"),
            "experimental_local_120b_accepted": False,
        }

    @staticmethod
    def merge(base, patch):
        result = copy.deepcopy(base)
        for key, value in patch.items():
            result[key] = (
                Settings.merge(result[key], value)
                if isinstance(value, dict) and isinstance(result.get(key), dict)
                else copy.deepcopy(value)
            )
        return result

    def snapshot(self):
        with LOCK:
            return copy.deepcopy(self.config)

    def update(self, patch):
        with LOCK:
            candidate = self.merge(self.config, patch)
            validate(candidate)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf8")
            replace_settings_file(temp, self.path)
            self.config = candidate
            return self.snapshot()
