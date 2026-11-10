# Develop and maintain the EMS AI core

Start with the [README](../README.md).

## Run the source

Use Python 3.12. In VS Code, open the folder and use **Terminal → Run Task**, or run these commands from the repository folder:

| Action | macOS | Windows PowerShell |
|---|---|---|
| Install dependencies | `bash scripts/setup-macos.sh` | `./scripts/setup.ps1` |
| Start the core + test console | `.venv/bin/python scripts/test_console.py` | `.venv\Scripts\python.exe scripts\test_console.py` |
| Run the tests | `bash scripts/test-macos.sh` | `./scripts/test.ps1` |

Development uses `.dev-data`, separate from any installed EMS data. On a new machine, open **Setup & keys** in the test console and install the offline models and the voice. The scripts `setup-models.ps1`, `setup-speech.ps1` and `setup-chatterbox.ps1` remain for scripted setup. See [Mac details](reference/MACOS_SETUP.md) if a prerequisite is missing.

## How a request runs

1. A client (the test console or your own program) calls the API with the session token.
2. `backend/app/main.py` authenticates the request and selects the service.
3. `backend/app/agents/assistant.py` freezes the settings, builds the prompt and permitted history, and runs the bounded tool loop through `providers/client.py`. It then removes unknown citations and saves the answer.
4. For voice, `scripts/speech_worker.py` transcribes the recording. `speech/reply.py` prepares the first clip while the answer streams, and `scripts/chatterbox_worker.py` generates the Arabic audio. The client plays the clips in order.

Spoken restatements also go through `agents/voice_facts.py`, which checks the before/after timings the user supplied. A conflicting draft gets one correction attempt; if it still conflicts, EMS asks for clarification. This check does not prove every fact in an answer. Recognized text is never silently replaced.

Defaults live in `config/`. Existing data folders use their saved settings, so changing YAML does not override a user's choices. Use the [File map](FILE_MAP.md) for specific edits and [Model configuration](reference/MODEL_CONFIGURATION.md) for model changes.

## Verify a change

Add a regression test for the failure, run the relevant tests, then run the full test command. Use invented examples, and don't copy personal chats, documents, recordings or keys into fixtures.

For voice, record the recognized words, the complete final answer, first audio, playback completion and gaps; the test console's timing table shows these. Validate meaning separately from audio validity. Keep cold model preparation separate from turn latency.

If you change `speech_chunks()` in `backend/app/speech/reply.py`, make the same change to `speechChunks()` in `test-console/index.html`. If they differ, the server's prepared first clip is never used. `tests/fixtures/voice/chunks.json` holds the shared cases.

## Data and recovery

The installed data folder is `~/Library/Application Support/EMS-AI-Assistant` on Mac and `%LOCALAPPDATA%\EMS-AI-Assistant` on Windows; development uses `.dev-data`. It holds chats, documents, settings and runtime files. Stop the core before copying the database. Never delete this folder to repair setup.

`POST /knowledge/export` and `/knowledge/import` move the document library between machines. Each device needs its own runtime and provider keys: Windows uses DPAPI and Mac uses Keychain. Keep authentication, loopback restrictions, source citations and explicit tool approvals intact.

## Update the guides

Run `.venv\Scripts\python.exe scripts\verify_docs.py` (or `.venv/bin/python` on Mac) to check local documentation links.
