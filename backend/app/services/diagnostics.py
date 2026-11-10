import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import psutil
import yaml
from app.core.settings import ROOT


def hardware(path):
    gpu = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in result.stdout.splitlines():
            name, total, free, driver = [p.strip() for p in line.split(",")]
            gpu.append({"name": name, "vram_mb": int(total), "free_vram_mb": int(free), "driver": driver})
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    disk = shutil.disk_usage(path)
    return {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "available_ram_gb": round(psutil.virtual_memory().available / 2**30, 1),
        "free_disk_gb": round(disk.free / 2**30, 1),
        "gpus": gpu,
    }


def build_info():
    path = ROOT / "config/build.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    config = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf8"))
    return {
        "version": config["version"],
        "build_timestamp": "development",
        "source_commit": "unbuilt",
        "schema_version": config["schema_version"],
        "dependencies": {name: importlib.metadata.version(name) for name in ("fastapi", "httpx", "pydantic")},
    }
