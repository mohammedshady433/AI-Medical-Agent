# Speech

EMS voice conversations use three components: Faster Whisper recognizes the recording, the selected model writes an answer, and Chatterbox speaks Arabic. English uses the operating system's voice. The test console (`test-console/index.html`) exercises the whole loop; this page describes the implementation.

## Setup and profiles

Use **Install voice** in the test console (`POST /setup/install` with `component: "voice"`). Recognition and synthesis have separate Python environments. Model revisions and hashes are pinned in `config/runtime-lock.json`; ordinary inference uses installed local files.

Chatterbox V3 provides **Standard Arabic** and **Egyptian** profiles. The Egyptian profile uses the synthetic reference in `assets/voices/egyptian-reference.wav`; its provenance and checksum are stored beside it. Speaker conditioning is cached for the worker's lifetime. Changing the profile restarts the worker. Personal recordings and documents are never used as voice references.

The Mac uses CPU recognition and Arabic synthesis. A compatible Windows NVIDIA installation can use CUDA after testing its drivers and memory limits. Mac English uses Samantha; Windows English uses a configured system voice. Setup fills platform-specific paths. See [Mac prerequisites](MACOS_SETUP.md) for native Python and voice requirements.

The installed Chatterbox revision is pinned in `requirements-chatterbox.in`. Keep that revision and the matching checkpoint together. KemeTone is retained for legacy configurations; normal setup does not download it. It is not the default Arabic voice.

## Capture and recognition

The client records 16 kHz mono PCM WAV and uploads one recording per turn to `/speech/transcribe`. The test console stops a recording after a pause, using a simple loudness threshold. Uploads (16 MB) and recording duration (120 s) are bounded, and temporary recordings are removed after processing.

`scripts/speech_worker.py` performs transcription without document prompts or previous-segment conditioning. It rejects near-silence, but noise and unclear words can still cause recognition errors. Recognition output is preserved; the answer layer must not replace an unknown word with an expected test phrase.

## Answers and playback

Voice answers use the selected local or online model. Offline remains local even after a provider failure. GPT-OSS 20B voice answers are limited to 512 response tokens; other voice models use up to 256. Context and history are bounded separately from typed chat.

`backend/app/agents/voice_facts.py` checks before/after phrases in spoken restatements. A conflicting draft gets one correction attempt. An unresolved conflict asks for clarification. This check covers relative timing; it is not a general factuality test.

`backend/app/speech/reply.py` prepares the first speech chunk and validates it against the final answer. The client plays the reply in order and requests the next clip while the current one plays; `test-console/index.html` shows the reference implementation. Chunking preserves word boundaries and decimal values. Playback metrics distinguish first audio, full completion and gaps.

The Chatterbox service caches equivalent spoken text in memory and combines identical concurrent requests. Changed words, numbers, voice settings or speed require new synthesis. Cache reuse does not reduce generation time for every new reply. The Mac's measured complete-reply results are in [current test results](../CURRENT_STATE.md).

## Online options

Voice has its own explicit Offline/Online setting. Online can send recognized text to the configured answer provider while Chatterbox still speaks locally. Optional Azure Speech sends response text to Azure and requires its own key and region. Offline mode refuses Azure synthesis. Provider failures do not change modes or start paid requests automatically.

## Change and test speech

- New-install defaults: `config/speech.yaml`.
- Worker paths and selected voice: saved app settings.
- Recognition: `scripts/speech_worker.py`.
- Arabic synthesis: `scripts/chatterbox_worker.py`, `chatterbox_cpu.py`, `chatterbox_graph.py`.
- Service lifecycle and cancellation: `backend/app/speech/service.py`.
- Capture, interruption and playback: `test-console/index.html`.

Run the native test command after code changes. `scripts/smoke_speech.py` checks real synthesis and `scripts/smoke_voice_conversation.py` runs a recorded voice turn; the test console exercises the live loop. These checks can load models and play audio.

Record expected words, recognized words, final meaning, preparation time, first audio, full playback duration and gaps. Keep physical microphone tests separate from injected recordings. A valid WAV file does not establish pronunciation quality. Test a full reply, then mute, interruption and End before accepting a speech change.

Upstream licenses and references are listed in [third-party notices](../THIRD_PARTY.md).
