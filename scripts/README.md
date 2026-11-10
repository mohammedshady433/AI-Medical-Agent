# Scripts

Run these commands from the repository root. The [developer guide](../docs/MAINTAINER_GUIDE.md) explains prerequisites and data locations.

| Task | Mac | Windows PowerShell |
|---|---|---|
| Set up development | `bash scripts/setup-macos.sh` | `./scripts/setup.ps1` |
| Start the core + test console | `.venv/bin/python scripts/test_console.py` | `.venv\Scripts\python.exe scripts\test_console.py` |
| Run tests | `bash scripts/test-macos.sh` | `./scripts/test.ps1` |

The other scripts support these commands or run specific checks:

- `speech_worker.py`, `chatterbox_worker.py`, `chatterbox_cpu.py`, `chatterbox_graph.py` and the `speech_*` helpers run local recognition and speech. The backend starts them as worker processes.
- `download_runtime.py`, `configure_runtime.py` and `setup-models.ps1` / `setup-speech.ps1` / `setup-chatterbox.ps1` handle model setup and saved paths.
- `key_setup.py` is a one-page local form for saving an Ollama Cloud key.
- `verify_*`, `smoke_*`, `live_check.py` and `benchmark_*` are developer diagnostics. Read their arguments before running them: voice checks can load large models or play audio, and live checks can use provider quota.

Generated logs and caches are ignored by Git. Keep the data folder, runtime models and Python environments when cleaning.
