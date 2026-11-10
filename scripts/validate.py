from pathlib import Path
import sys
import tempfile
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings

with tempfile.TemporaryDirectory(prefix="ems-config-") as tmp:
    config = Settings(Path(tmp)).snapshot()
    print("Configured defaults:", config["models"]["online"]["id"], "/", config["models"]["offline"]["id"])
version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
assert yaml.safe_load((ROOT / "config/app.yaml").read_text())["version"] == version, (
    "Synchronize pyproject.toml and config/app.yaml versions."
)
print("Configuration and version validation passed.")
