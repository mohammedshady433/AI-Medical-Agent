import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.diagnostics import hardware

parser = argparse.ArgumentParser()
profile = parser.add_mutually_exclusive_group()
profile.add_argument("--experimental", action="store_true")
profile.add_argument("--recommended", action="store_true")
profile.add_argument("--embeddings-only", action="store_true")
args = parser.parse_args()
info = hardware(Path(os.environ.get("LOCALAPPDATA", ".")))
print(json.dumps(info, indent=2))
required = 8 if args.embeddings_only else 80 if args.experimental else 24
if info["free_disk_gb"] < required:
    raise SystemExit(f"Insufficient free disk: allow at least {required} GB for this setup.")
if args.embeddings_only:
    print("Embeddings-only setup: local document search with cloud answers; no GPT-OSS assistant download.")
elif args.experimental:
    print(
        "Experimental 120B: this laptop cannot provide fast fully GPU-resident inference. 20B is recommended. No model has been downloaded by this check."
    )
elif info["ram_gb"] < 24:
    print("Warning: GPT-OSS 20B may be slow or run out of memory on this computer.")
