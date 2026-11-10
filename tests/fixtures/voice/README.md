# Local voice benchmark inputs

These are the existing public synthetic questions used in the EMS local voice tests. They contain requests for help organizing the day's work, not private documents or a user's microphone recording. Arabic uses synthetic local speech; English uses local Windows speech. They are input fixtures, not reference voices for cloning.

`scripts/benchmark_streaming_voice.py` reads these files directly. The reports record their SHA-256 hashes and the recognition model's exact transcript for each turn. Fresh answer audio is generated on every run. Different recognition or generation runtimes can produce different wording; inspect the reported text before comparing numbers.

Real room noise, microphone hardware, spontaneous speech and interruption require separate acceptance tests.
