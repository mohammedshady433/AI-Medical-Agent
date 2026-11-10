# Troubleshooting

Start the core with `scripts/test_console.py` and open **Backend status** and **Setup & keys** in the test console. For measured voice performance and known limits, see [Test results](CURRENT_STATE.md).

Start with the exact error and the version from `GET /health`. Preserve the selected mode while diagnosing. `logs/*.log` in the source tree records development/build commands; the data folder's `logs` directory records the speech, Ollama and setup processes. The core's own errors print in the terminal that runs it. Do not send credentials, private documents or full conversation contents in a support report.

| Problem | Check | Repair and verification |
|---|---|---|
| Python is missing | Windows: `py -3.12 --version`; Mac: `python3.12 --version` | Install native Python 3.12 for that computer and rerun `scripts/setup.ps1` or `setup-macos.sh` |
| Dependency import error | `pip check`, Python interpreter path | Use `.venv` for API/build and the separate speech environment for Torch/Transformers; reinstall the appropriate pins |
| Ollama not responding | Endpoint, `/api/version`, running process | Start Ollama or configure the private runtime; no model is installed automatically by a conversation |
| Missing local model | `GET /setup/status` (installed models) | Use **Install offline models** in the test console; do not select a cloud tag for Offline |
| 120B is rejected | Hardware warning acceptance and free disk | Use 20B on this laptop; 120B is an explicit slow experiment, not a normal setup requirement |
| GPU out of memory | `GET /diagnostics` VRAM, other GPU apps, context size | Close competing jobs, reduce context, keep one generation, use CPU embeddings/TTS, and unload CUDA ASR after recognition |
| CUDA DLL missing | Torch CUDA wheel, driver, ASR compute type | Install the matching isolated CUDA runtime; test CPU/int8 to separate model from GPU problems |
| API authentication failure | Provider-specific saved key and account | Replace expired/restricted key and run a real test; model switching does not cure every authorization problem |
| Empty provider model list | Endpoint and key account | Verify the provider's documented list API; enter an exact accessible ID and test inference |
| Provider/network failure | Error category and retry | Check connection or provider; choose Offline manually if wanted; mode never changes itself |
| Microphone not available | Browser and operating-system microphone permission, selected input | Open the console at `http://127.0.0.1` (not a LAN address), allow the microphone in the browser, choose a working input; use short recordings |
| Transcript contains wrong words | Noise, selected language, exact sample | Review before sending, use a better microphone; never fix this by injecting document vocabulary into ASR |
| Arabic voice sounds broken or cuts off | Selected voice and installed source revision | Use **Check & repair** and **Install voice** in the test console for missing components. Keep Chatterbox V3 and choose Standard Arabic for the current Mac profile. Do not replace pinned dependencies with an unreviewed pip upgrade. |
| English voice unavailable | Installed system English voice | Add a Windows English voice or macOS Samantha in the system speech settings. Azure is an optional Online voice alternative. |
| Speech worker times out | `speech.log`, model files, paths | Verify the model manifest, restart after changing paths, test a short sentence through `smoke_speech.py` |
| Long speech stalls | Transport buffer and chunk size | Use the current bounded WAV transport and chunked playback; do not revert to a 64 KB JSON-line limit |
| PDF has no text | Is it scanned/encrypted? | Decrypt an authorized copy; configure local OCR and language data for scans |
| Office file extraction fails | File integrity, legacy runtime files | Test a known small fixture; preserve Java/Tika policy and all runtime files; save to a modern format if necessary |
| `keyword_only` status | BGE-M3 server/model availability | Start local embeddings and Retry; successful extracted text is reused |
| Cross-language search misses | Semantic indexing and embedding model | Complete BGE-M3 indexing; test known Arabic/English facts; changing embeddings requires re-indexing |
| All formats report extraction failure | Full local traceback and whether chunks exist | A shared indexing/runtime failure can affect every format. Verify the packaged SSL policy file, embedding endpoint and current build before blaming each parser. |
| Cloud document setup downloads too much | Provisioning command | Use `setup-models.ps1 -InstallOllama -EmbeddingsOnly` for local search with cloud answers; this skips the offline assistant. |
| Development cannot see installed keys or runtimes | `.dev-data` versus the installed application-data workspace | Configure development paths separately using the platform setup guide; enter your own development key. |
| Settings file is invalid | JSON syntax and schema validation | Close EMS, back up settings.json, repair the invalid field from defaults; do not delete the database or key store |
| Database corruption | Disk health and a copy of the database | Stop EMS, preserve the damaged file, restore a known backup; never perform destructive repairs on the only copy |

## Developer checks on Mac

```bash
bash scripts/test-macos.sh
.venv/bin/python -m pip check
```

Use the test console's setup controls for repairs; these commands check the source checkout.

## Developer checks on Windows

```powershell
.\scripts\test.ps1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\hardware_check.py --recommended
```

The ten connectivity acceptance tests use controlled HTTP responses. To test a real internet loss, first configure working online credentials and a local model. In the test console, select Online for typed chat and send a question; disconnect the network and send again. Confirm the error suggests Offline and the mode stays Online. Then select Offline yourself, reconnect, and confirm the mode stays Offline. Do not use a fake response as evidence that the real providers worked.
