# Test results — EMS 1.4.2

Updated 6 November 2026. EMS 1.4.2 is installed on the development Mac. Native Mac and Windows builds and packaged UI checks passed. Physical playback and pronunciation on this build remain unverified.

## What changed

- Spoken restatements keep supplied before/after timing. A conflicting draft gets one correction attempt with GPT-OSS 20B; an unresolved conflict asks for clarification before audio. Original recognition text stays in history. An explicit correction takes precedence without discarding the other activities.
- Natural equivalents such as “after finishing your work” are accepted. Tests also cover genuine “after lunch” input, before/after reversal, unknown recognized words and cancellation.
- Local Arabic audio is reused when only diacritics or internal spaces differ, matching the existing Chatterbox normalization. Identical simultaneous requests share one synthesis. Changed words, times, voice or speed invalidate reuse. This is an in-memory cache, cleared when speech closes.
- The transcription-processing caption is hidden below the voice circle. Accessible listening status and useful error controls remain.
- One workflow builds and tests both native packages. Setup, usage, development and reference guides are indexed in [Documentation](README.md). Older development reports remain in Git history.

## What passed on this Mac

The final source checks passed: **283 backend tests / 9 skipped, 30 frontend tests, TypeScript, 14 desktop helper tests**. Real GPT-OSS 20B checks covered work/lunch/before/unclear phrases, followed by three successful correction-and-summary checks.

The complete API test used the existing synthetic three-sentence recording, actual local recognition, GPT-OSS 20B and Chatterbox. Recognition still misheard one word. EMS clarified it instead of inventing lunch. After an explicit correction to “after work,” the final answer retained all three activities, 9 in the morning, shopping after work, and 10 at night. All three WAV clips were validated through their final PCM frames in both the source service and the frozen packaged backend. The packaged recording lasts 9.56 seconds. Live settings were unchanged. The installed About screen shows version 1.4.2, a clean application build, and Offline / 20B.

| Measurement | Source service | Packaged backend |
|---|---:|---:|
| Local voice preparation | 13.82 s | 17.82 s |
| Full recording recognition | 3.10 s | 3.09 s |
| First corrected-reply audio available | 4.46 s | 4.76 s |
| All corrected-reply audio generated | 12.33 s | 11.99 s |
| Generated audio duration | 10.08 s | 9.56 s |
| Repeat with equivalent diacritics, all three clips | 0.0057 s | 0.0059 s |
| Reused WAV bytes identical | Yes, all three clips | Yes, all three clips |

The earlier failed reply had first audio at 7.61 s and all audio at 12.22 s. Its wording and correction differed and its answer was inaccurate, so this is an observed example comparison, not a controlled speed improvement claim. Fresh complete replies still take roughly the same generation time. Reuse accelerates matching repeated wording; it does not accelerate every new question.

## Native packages and installation

[Native CI run 42194954448](https://github.com/Inegm03/EMS-AI-Assistant/actions/runs/42194954448) passed on both systems. Mac: 287 backend tests / 5 skipped. Windows: 291 / 1 skipped. Both passed 30 frontend tests, desktop helpers and three packaged UI tests. The Mac also checked the legacy Office helper with Java 17.

The first Mac CI attempt skipped ad-hoc signing because it was a pull request. The workflow now explicitly permits ad-hoc preview signing without any signing certificate or secret; strict signature verification and packaged launch checks passed afterward. Windows remains unsigned and Mac remains unnotarized.

The development Mac runs the clean application build from `fdf32eb`; CI built `eae8faf`, whose only additional change is the build workflow. The copied application and backend hashes match the tested local package; the signature is valid. The previous app is preserved under Desktop/ems/Previous versions. Models, voice selection and live settings were kept.

## Performance experiments

A shorter opening reduced first-clip generation but worsened the calculated finish from about 14.3 to 18.1 seconds and introduced a roughly 4.4-second gap. It was reverted. Testing 2, 4, 6, 10, 12 and 14 CPU threads did not improve on the existing 8-thread setting; those changes were rejected. Recognition beam changes did not improve this recording either. No weaker model, faster playback setting or altered voice checkpoint was substituted.

An initial fact-check prototype also rejected valid “after finishing your work” wording. Possessive/completion handling and regression tests fixed that false rejection. The complete scenario was rerun afterward.

## Known limits

- Physical playback and a new microphone test did not complete. The native audio queue and test browser process were restricted by this session; opening the local in-app browser test page was declined. No alternate browser route was used after that refusal.
- WAV validation does not score pronunciation. Before/after phrase checks do not prove every fact or task association in an answer. Unknown recognition words still need user correction.
- The fixed-text timing comparisons synthesized every clip but calculated playback scheduling; they did not observe the last word playing.
- Clean-device rollout, publisher signing/notarization and a human Arabic listening review remain separate acceptance work. The local update and native packaged smoke tests have passed.

Raw, invented-data evidence is in [voice-faithfulness.json](benchmarks/voice-faithfulness.json). Reproduce model checks with `.venv/bin/python scripts/verify_voice_facts.py --endpoint <verified local Ollama URL> --output build/voice-facts-new`. Use the Windows `.venv\Scripts\python.exe` path on Windows. Reuse installed models; do not guess the local service port.

## Distribution checks still needed

Before a wider rollout, test installation and updates on a clean Windows account and a clean Mac, including model download recovery. Complete a microphone and Arabic listening check on each device type. Publisher signing, Mac notarization and organization-specific data retention policies are not covered by the automated package tests.

The earlier library tests used invented CSV, JSONL and text data, including instructions, export/import and restart. Their records are retained in [library checks](benchmarks/mac-knowledge-acceptance.json) and [installed library checks](benchmarks/mac-knowledge-delivery.json); they describe an earlier build and do not replace current device acceptance.
