# Changelog

## Unreleased: AI core extraction

- Removed the React interface, Electron shell, installers, packaging scripts, build workflow and desktop commands (time/date shortcuts, app control, volume, notes, weather). They remain in Git history.
- Kept the answer engine, document search, voice pipeline, model setup and the optional Research, Vision and MCP extensions.
- Moved the bundled document readers to `document-runtime/` and the settings schema to `backend/app/core/options.py`.
- Added `test-console/index.html` and `scripts/test_console.py` for testing chat, voice, setup and documents in a browser.

## Documentation maintenance — 6 November 2026

- Added direct Mac and Windows download links and a documentation index.
- Combined everyday instructions into one user guide; grouped technical details under Reference.
- Removed outdated development diaries and duplicate reports from the current source tree. They remain in Git history.
- Updated the PDF, source overview and another-computer checklist. Application behavior and installers are unchanged.

## 1.4.2 — Spoken details

- Check before/after timing in spoken summaries; repair a conflicting answer once, then ask for clarification if needed.
- Preserve the original transcript and selected offline model and voice.
- Reuse equivalent cached Arabic audio and combine duplicate synthesis requests.
- Remove the transcription-processing caption beneath the voice circle.
- Consolidate native Mac and Windows builds into one workflow.

## 1.4.1 — Complete voice requests

- Collect queued microphone audio and continuations before answering.
- Ask for an unclear morning hour instead of inventing one.

## 1.4.0 — Documents and saved guidance

- Add file instructions, library memory and extracted-text previews.
- Export and import document libraries between computers.
- Keep imported instructions disabled until reviewed.
- Make clipboard encryption asynchronous and preserve unreadable encrypted history.

## 1.3.2 — Voice and usability

- Reduce Standard Arabic CPU decoder work on Apple Silicon.
- Keep Mac speech settings independent of Windows GPU limits.
- Add explicit online voice selection and full playback timing.
- Preserve transcripts and exclude tool planning from spoken answers.

## 1.3.0 — macOS support

- Add native Mac data paths, Keychain storage, system English speech and desktop controls.
- Add Apple Silicon DMG/ZIP packaging and Mac setup, development and test scripts.
- Keep Windows DPAPI, SAPI, CUDA and EXE packaging support.

## 1.2.0 — Setup and workplace interface

- Add managed model and speech setup with progress, cancellation and retry.
- Simplify everyday settings and move technical options to Advanced settings.
- Remove entertainment integrations from the application.

## 1.1.0 — Desktop tools

- Add Arabic and English commands, editable notes, keyboard shortcuts and clipboard history.
- Add research, optional screen analysis and confirmed external-tool actions.
- Add local model selectors and speech residency controls.

## 1.0.3 — Startup and portable app fixes

- Isolate portable extraction between launches.
- Include the build interpreter's required SSL policy file.
- Wait for backend readiness before opening the interface.
- Return diagnostic references for service failures.

## 1.0.2 — Speech runtime

- Pin the Chatterbox revision that fixes premature V3 speech cutoff.
- Add the Egyptian reference voice and cached speaker conditioning.
- Prepare later speech chunks while the current clip plays.

## 1.0.1 — Chatterbox support

- Add Chatterbox Multilingual V3 in a separate speech environment.
- Keep recognition local and preserve numeric values in spoken text.
- Correct packaged child-process working directories.

## 1.0.0 — Initial desktop application

- Add Electron, React and FastAPI application structure.
- Add local and online answers, document search, saved chats and provider settings.
- Add local recognition, Arabic and English speech, OCR and native packaging checks.

See [current test results](CURRENT_STATE.md) for the latest validation and known limits.
