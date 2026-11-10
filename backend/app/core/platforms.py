"""Native paths shared by backend setup and command adapters."""
import os
from pathlib import Path
import sys


def data_directory():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/EMS-AI-Assistant"
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EMS-AI-Assistant"


def venv_python(folder):
    return Path(folder) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ollama_name():
    return "ollama.exe" if os.name == "nt" else "ollama"
