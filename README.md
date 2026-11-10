# EMS AI core

The AI engine behind the EMS assistant, extracted from the desktop app so it can be reused in other projects. It answers questions in Arabic and English, searches your documents with citations, and holds spoken conversations, all on your own computer when you choose Offline.

| Capability | Local (Offline) | Cloud (Online, optional) |
|---|---|---|
| Answers | GPT-OSS 20B or Qwen 7B through a private Ollama | GPT-OSS 120B on Ollama Cloud, or any OpenAI-compatible API |
| Document search (RAG) | BGE-M3 embeddings + SQLite FTS5 hybrid search | same, local |
| Speech to text | Faster-Whisper large-v3-turbo | same, local |
| Text to speech | Chatterbox Arabic (Standard or Egyptian); OS voice for English | Azure Speech |

The desktop interface (React + Electron), installers and desktop commands were removed. The core is a Python FastAPI service on `127.0.0.1` that requires a session token on every request.

## Quick start (Windows)

```powershell
./scripts/setup.ps1                               # Python 3.12 environment + dependencies
.venv\Scripts\python.exe scripts\test_console.py   # start the AI core and open the test page
```

On a Mac, use `bash scripts/setup-macos.sh`, then `.venv/bin/python scripts/test_console.py`.

The test console opens in your browser at `http://127.0.0.1:8765/console/`. Use it to:

- **Chat:** stream answers, see tool steps and `[D…]` citations, stop a reply.
- **Voice:** press **Start talking** and pause. EMS transcribes with Whisper, answers in spoken style, checks before/after timings, then speaks with Chatterbox. Each stage is timed.
- **Setup & keys:** download the offline models or the voice, save an online API key, test connections.
- **Documents:** add files and watch them get indexed.

By default the console uses `./.dev-data` when present, otherwise the installed app's data folder. Pass `--data-dir <folder>` to choose another, or `--port <number>` to change the port.

## Use the core from your own code

Start the backend the same way the console does (see [scripts/test_console.py](scripts/test_console.py)), or call `create_app(data_dir, token)` from [backend/app/main.py](backend/app/main.py) and serve it with uvicorn. Send `Authorization: Bearer <token>` on every request. The main routes:

| Route | Purpose |
|---|---|
| `POST /chats` → `POST /chat {chat_id, text, voice}` | Stream an answer as NDJSON events: `token`, `status`, `approval`, `done`, `cancelled`, `error` |
| `POST /chat/cancel` | Stop the running answer |
| `POST /speech/transcribe?language=ar\|en\|auto` (multipart `file`) | Whisper speech to text |
| `POST /speech/synthesize {text, language, generation, local_only, voice_conversation}` | Base64 WAV clip |
| `POST /voice/prepare {id}` | Preload the answer and speech models before a conversation |
| `POST /documents` (multipart `file`), `GET /documents` | Add and list library files |
| `GET/PATCH /settings`, `POST /credentials` | Read or change settings; store a provider key |
| `GET /setup/status`, `POST /setup/install {component, accepted_download}` | Model and voice setup |

## Repository

| Folder | Contents |
|---|---|
| [backend/app](backend/app/) | The AI core: agent loop, providers, RAG, speech, storage, setup, optional Research/Vision/MCP |
| [scripts](scripts/README.md) | Setup, tests, the test console, speech workers, model downloads, benchmarks |
| [test-console](test-console/) | The single-file browser test page |
| [config](config/) | Defaults for new data folders, including the assistant's personality |
| [document-runtime](document-runtime/) | Bundled Java + Apache Tika and Tesseract OCR used to read old Office files and scans |
| [assets/voices](assets/voices/) | Synthetic Egyptian voice reference and the speech warm-up clip |
| [tests](tests/) | Unit and integration tests (`./scripts/test.ps1`) |
| [docs](docs/README.md) | Architecture, speech, documents, model configuration and test results |

[Developer guide](docs/MAINTAINER_GUIDE.md) · [Which file to edit](docs/FILE_MAP.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)

Chats, settings, documents and downloaded models live in the data folder, never in the source tree. Setup needs internet; completed offline conversations run on your computer. License: [GPL-3.0](LICENSE). [Third-party notices](docs/THIRD_PARTY.md).
