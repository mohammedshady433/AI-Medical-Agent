# Third-party components and attribution

The EMS first-party repository retains the prior project's GPL-3.0 license (`LICENSE`). Third-party components are not relicensed by EMS. Preserve their notices when redistributing their binaries or source.

| Component | Role | Upstream / license |
|---|---|---|
| FastAPI, Uvicorn | Local API service | Their MIT/BSD-licensed upstream distributions |
| Ollama | Local and cloud model serving | https://github.com/ollama/ollama · MIT |
| GPT-OSS | Language model | OpenAI GPT-OSS model repositories · Apache-2.0 |
| BGE-M3 | Multilingual embeddings | https://huggingface.co/BAAI/bge-m3 · MIT |
| Faster Whisper | Speech recognition runtime | https://github.com/SYSTRAN/faster-whisper · MIT |
| Whisper large-v3-turbo converted weights | Local ASR | https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo |
| KemeTone | Egyptian female TTS and G2P | https://huggingface.co/Rabe3/kemetone · Apache-2.0 model card; pinned revision in runtime-lock.json |
| Chatterbox Multilingual V3 | Replacement local Arabic TTS | https://huggingface.co/ResembleAI/chatterbox and https://github.com/resemble-ai/chatterbox · MIT; checkpoint revision pinned in runtime-lock.json |
| Resemble PerTh | Chatterbox audio watermarking | https://github.com/resemble-ai/Perth · retain upstream notices and watermarking |
| Kokoro / Misaki | KemeTone inference / phonemization | https://github.com/hexgrad/kokoro · Apache-2.0 |
| eSpeak NG | Local phoneme generation | https://github.com/espeak-ng/espeak-ng · GPL-3.0 |
| Apache Tika | Legacy document extraction | https://tika.apache.org · Apache-2.0 |
| Portable Java runtime | Tika runtime | Bundled legal files in document-runtime/jre/legal |
| Docling | Optional structured document parsing | https://github.com/docling-project/docling · MIT; model licenses are separate |
| Tesseract / tessdata | Bundled Arabic/English OCR | https://github.com/tesseract-ocr · Apache-2.0 |

The document runtime is carried forward from the existing EMS package with its Java legal files, parser source, policy and manifest. The legacy Office fixtures are public Apache Tika test documents; see `tests/fixtures/NOTICE.md`. Downloaded model manifests record exact repository revisions and file hashes. Large language/speech model files are not included in the source tree or Git history.

KemeTone is a synthetic voice model. EMS does not claim that generated speech is a recording of a real person or provide voice impersonation features. Local English synthesis uses a voice already installed and licensed with Windows. Azure and Brave services are governed by the user's own provider account terms.

The bundled Tesseract 5.5.3 runtime includes its upstream `doc/LICENSE` and language data. Exact runtime and Arabic/English trained-data download revisions/hashes are recorded in `config/runtime-lock.json`. The small OCR language files are included so scans and text images work without a separate OCR installation.
