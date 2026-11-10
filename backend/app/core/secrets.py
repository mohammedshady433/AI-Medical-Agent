"""DPAPI binds saved credentials to the current Windows account; never return keys to UI."""

import sys
import hashlib
import base64
import json
import os
import re
import threading
from pathlib import Path


class SecretStore:
    def __init__(self, data: Path):
        self.path = data / "credentials.dpapi.json"
        self.lock = threading.RLock()
        # Keep development/test workspaces separate from installed credentials.
        scope = hashlib.sha256(str(data.resolve()).encode()).hexdigest()[:24]
        self.service = "com.ems.assistant." + scope

    @staticmethod
    def _keychain():
        # Explicit native backend: never fall back to plaintext storage.
        from keyring.backends.macOS import Keyring
        return Keyring()

    def get(self, name: str, env: str = "") -> str:
        with self.lock:
            if sys.platform == "darwin":
                value = self._keychain().get_password(self.service, name)
                return value if value is not None else (os.environ.get(env, "") if env else "")
            values = json.loads(self.path.read_text()) if self.path.exists() else {}
            if name in values:
                import win32crypt

                return win32crypt.CryptUnprotectData(base64.b64decode(values[name]), None, None, None, 0)[
                    1
                ].decode("utf8")
            return os.environ.get(env, "") if env else ""

    def save(self, name: str, value: str):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or len(value) > 4096:
            raise ValueError("Invalid credential name or size.")
        if sys.platform == "darwin":
            with self.lock:
                keychain = self._keychain()
                if value.strip():
                    keychain.set_password(self.service, name, value.strip())
                elif keychain.get_password(self.service, name) is not None:
                    keychain.delete_password(self.service, name)
            return
        import win32crypt

        with self.lock:
            values = json.loads(self.path.read_text()) if self.path.exists() else {}
            if value.strip():
                values[name] = base64.b64encode(
                    win32crypt.CryptProtectData(
                        value.strip().encode("utf8"), "EMS provider credential", None, None, None, 0
                    )
                ).decode("ascii")
            else:
                values.pop(name, None)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(values), encoding="utf8")
            os.replace(tmp, self.path)
