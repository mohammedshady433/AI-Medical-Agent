# macOS setup

The EMS AI core runs natively on Apple Silicon Macs (M1 or newer). Intel Macs have no validated local voice dependency set; typed chat and documents may work there but are untested. See [the test results](../CURRENT_STATE.md) for checks actually completed.

## Set up the source

Install Git and native **Python 3.12** from [python.org](https://www.python.org/downloads/macos/) or `brew install python@3.12`. Don't use Rosetta Python on an arm64 Mac. Then:

```bash
git clone https://github.com/Inegm03/EMS-AI-Assistant.git
cd EMS-AI-Assistant
bash scripts/setup-macos.sh
.venv/bin/python scripts/test_console.py
```

If `python3.12` is not on PATH, use `EMS_PYTHON312=/absolute/path/to/python3.12 bash scripts/setup-macos.sh`. The console uses `.dev-data` when it exists, otherwise `~/Library/Application Support/EMS-AI-Assistant`. Pass `--data-dir` to choose another. Keep the same data folder throughout setup and repair.

Provider keys use the current user's macOS Keychain. macOS can ask you to authorize access to that Keychain item. A denied or locked Keychain produces an error; EMS doesn't fall back to a plaintext credential file.

## Local models and document questions

1. EMS reuses native [Ollama for macOS](https://ollama.com/download/mac) when it is installed, or downloads its checksum-verified engine during setup. It runs a separate loopback-only engine on port **11435** with its own models folder. Ollama also uses its standard `~/.ollama` key directory; keep those keys private.
2. In the test console, open **Setup & keys** and click **Install offline models**. This downloads GPT-OSS 20B, Qwen 7B and BGE-M3 embeddings (about 25 GB). Completed downloads are reused.
3. Select **Offline** for typed chat and ask a short question.
4. Add a document under **Documents**. Wait for `ready`, then ask a specific question and check the citation. `keyword_only` means semantic embeddings aren't available yet.

Allow at least 16 GB unified memory for a practical 7B setup; 32 GB or more is better when combining larger models with voice. These are planning targets, not certified minimums. Ollama uses Metal acceleration; Mac setup never installs CUDA.

OCR is optional: run `brew install tesseract tesseract-lang`. Arabic OCR needs the `ara` language data and English needs `eng`; check with `tesseract --list-langs`.

Older DOC/PPT/XLS files and additional Tika formats need a native **Java 17**, discoverable by `/usr/libexec/java_home -v 17`, such as [Eclipse Temurin 17](https://adoptium.net/temurin/releases/?version=17). Otherwise convert these files to DOCX/PPTX/XLSX. Don't point the reader at a newer Java: it relies on Java 17's restricted SecurityManager.

## Local Arabic and English voice

1. In **Setup & keys**, click **Install voice**. Recognition and synthesis use separate Python environments.
2. Keep CPU for recognition and Arabic synthesis; CUDA isn't available on Macs. Chatterbox V3 provides the Egyptian and Standard Arabic profiles. Standard Arabic uses the Mac CPU decoder optimization.
3. Press **Start talking** in the console and allow microphone access when the browser asks. Use `http://127.0.0.1`, because browsers only allow the microphone on local or secure pages.
4. English replies use the local **Samantha** voice through `/usr/bin/say`. If it's missing, install Samantha in the macOS accessibility speech settings.

Voice defaults to Offline and is independent of the typed-chat mode. CPU Arabic synthesis adds latency; Windows CUDA timings don't describe Mac performance.

## Data and troubleshooting

Stop the core before backing up the data folder (database, documents and runtime paths). Keychain credentials are separate; enter them again on another Mac or account. Windows DPAPI files can't be decrypted on macOS, and a Windows settings file has incompatible runtime paths.

Setup output is in `logs/setup.log` inside the data folder, and speech worker logs are next to it. Check CPU selection, native architecture, free memory and disk, and microphone permission before retrying. Failures never silently switch between Online and Offline.
