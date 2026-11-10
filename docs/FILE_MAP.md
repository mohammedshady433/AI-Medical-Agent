# Which file should I change?

Start with one row. Saved settings in the data folder override files in `config/`.

| Change | File or folder | What it controls |
|---|---|---|
| API routes | `backend/app/main.py` | Local requests, cancellation and status |
| Answer generation | `backend/app/agents/assistant.py` | Prompts, history, the tool loop and the final answer |
| Before/after consistency | `backend/app/agents/voice_facts.py` | Detects changed relative times in spoken restatements |
| Tools the model may call | `backend/app/agents/tools.py` | Document search, overview, calculator, optional web search |
| Model connection | `backend/app/providers/client.py` | Ollama and OpenAI-compatible protocols, streaming, error mapping |
| Settings and validation | `backend/app/core/settings.py`, `core/options.py` | Defaults plus saved overrides, option schema, `resident_voice` |
| Provider keys | `backend/app/core/secrets.py` | DPAPI (Windows) or Keychain (Mac) storage |
| New-install model defaults | `config/models.yaml`, `providers.yaml` | Answer models, limits and endpoints |
| Personality | `config/personality.txt` | General behavior; voice-specific rules live in `assistant.py` |
| Document reading | `backend/app/documents/` | Format readers, OCR, chunking |
| Search and saved knowledge | `backend/app/rag/` | Embeddings, hybrid search, owner instructions, export/import |
| Storage | `backend/app/storage/database.py` | SQLite schema, migration and saved messages |
| Speech service | `backend/app/speech/service.py`, `reply.py` | Worker lifecycle, audio cache, first-clip preparation, cancellation |
| Recognition | `scripts/speech_worker.py` | Whisper decoding (no phrase substitution) and the English OS voice |
| Arabic speech | `scripts/chatterbox_worker.py`, `chatterbox_cpu.py`, `chatterbox_graph.py` | Voice weights, CPU/GPU execution and waveform generation |
| GPU sharing | `backend/app/services/gpu.py` | Keeps the answer model and speech models from overloading the GPU |
| Setup and downloads | `backend/app/services/setup.py`, `model_manager.py`, `scripts/download_runtime.py` | Explicit setup, progress, checksums and resume |
| Private Ollama | `backend/app/services/local_runtime.py` | Starts and stops the loopback-only model engine |
| Optional extensions | `backend/app/extensions/` | Research, Vision and MCP |
| Test console | `test-console/index.html`, `scripts/test_console.py` | Browser test page and the launcher that serves it |
| Tests | `tests/` | Unit and integration checks |

`document-runtime/` contains required document-reader dependencies (Java + Tika, Tesseract). Don't remove it as clutter: the document readers use it. `build/`, `.runtime/`, `.dev-data/` and local environments are ignored outputs.

Never commit installed data, keys, model downloads or private recordings.
