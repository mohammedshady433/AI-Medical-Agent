# Model and provider configuration

**Version 1.4.2.** The desktop Settings screens were removed with the GUI. Change settings with `PATCH /settings` (deep-merged JSON) and store keys with `POST /credentials`. The test console covers the common switches. Both GPT-OSS 20B and Qwen 7B are installed by offline model setup; Qwen 7B is a separate model, not a smaller GPT-OSS. To answer voice turns with Qwen, send `{"assistant": {"voice_model": "qwen2.5:7b-instruct"}}`; for typed chat, set `models.offline.id`. See the [test results](../CURRENT_STATE.md).

The shipped defaults live in `config/models.yaml` and `config/providers.yaml`. Saved user overrides live in `%LOCALAPPDATA%\EMS-AI-Assistant\settings.json` on Windows or `~/Library/Application Support/EMS-AI-Assistant/settings.json` on Mac. They are loaded outside the executable. Editing a bundled YAML file changes defaults for a subsequent build; it does not override settings already saved by a user. Use `PATCH /settings` for runtime changes. Stop the core before manually editing its persistent settings file.

| Setting | Location | Effect | Rebuild? |
|---|---|---|---|
| Online model ID, provider and endpoint | `PATCH /settings` → `models.online`, `providers` | Next online turn | No |
| Offline model ID and endpoint | `PATCH /settings` → `models.offline`, `providers.ollama_local` | Next local turn, after residency checks | No |
| Temperature, response limit, context, timeout, reasoning | `PATCH /settings` → `models.<mode>` | Next request | No |
| API key | `POST /credentials {provider, key}` | Next provider request | No |
| Local speech paths/provider/voice | `PATCH /settings` → `speech` | Speech worker restarts on next use | No |
| Document-sharing permission | `PATCH /settings` → `privacy` | Controls online document tools/history | No |
| Embedding model | Persistent settings or defaults | Requires re-indexing for the new embedding space | No for saved settings |
| New adapter implementation | `backend/app/providers/client.py` | Adds a wire protocol | Yes |
| Personality/default model catalogue | `config/personality.txt`, YAML | Development restart; production rebuild | Yes |

## Verified identifiers and hardware assumptions

Ollama's direct Cloud API uses `gpt-oss:120b` at `https://ollama.com/api/chat`. A local proxy tag such as `gpt-oss:120b-cloud` is a different deployment path and is not used for offline inference. The default offline model is `gpt-oss:20b`.

The original Windows evaluation used an RTX 3080 Laptop GPU with 16 GB VRAM and approximately 32 GB RAM. This is historical Windows evidence, not the current Mac hardware. Standard local 120B is far beyond fast GPU-resident inference on that hardware. The experimental setup requires hardware inspection, explicit acceptance and approximately 80 GB free disk allowance. It is never a default download. The 20B setup allows approximately 24 GB free disk for model/runtime installation. Context size and other resident models affect memory and latency; lower context before increasing concurrency.

## Changing a model

1. Stop the active response.
2. Send `PATCH /settings` with the new `models.online` or `models.offline` values.
3. Choose a provider. Edit its endpoint only if you know that provider's documented API.
4. Load models, or type the exact model ID. Listing is discovery, not proof of inference permission.
5. Adjust limits. `reasoning_effort: low` is the GPT-OSS latency default.
6. Save, then Test connection. The test performs a real minimal inference and may consume paid usage Online.
7. Select the desired operating mode in the header and send a normal question.

Voice has its own **Where should voice answers run?** selector. It defaults Offline. Online voice uses the configured online model and may incur provider charges; recognition remains local. The existing local Chatterbox voice can speak online answers, or Azure Speech can be selected explicitly with its own key and region. Changing typed-chat mode does not change voice mode.

The settings editing tabs do not change the active typed conversation mode. Saving a settings draft also does not restore an older mode selection. Mode changes occur through the visible header or explicit Offline recommendation button.

## Adding a provider

Add a provider entry containing `name`, `adapter`, `endpoint`, `mode` and `secret_env` to the defaults, or to the persisted provider map while EMS is closed. Available adapters are `ollama` and `openai_compatible`. Compatible endpoints are the base API URL, usually ending in `/v1`; the adapter appends `/chat/completions` and `/models`. It uses ordinary chat-completion message/tool schemas. Provider-specific unsupported parameters can cause request errors; test the exact endpoint/model combination.

Map `models.online.provider` to the provider entry and set its model ID. Enter its key using the secure form. Do not put keys in YAML. Offline adapters currently require Ollama, allowing the app to check local residency before sending prompts. A new wire protocol belongs behind the provider abstraction and needs transport, tool-streaming and error-mapping tests.

## Re-indexing and storage

Embeddings must use the same model/dimensions for stored passages and queries. Existing vectors are tagged with their model identifier. A model change must be followed by Retry/re-index for existing documents; old vectors are not silently interpreted in a different embedding space. Models are stored separately from the installer and are preserved by ordinary builds and app upgrades.

Sources: [Ollama Cloud](https://docs.ollama.com/cloud), [Ollama chat API](https://docs.ollama.com/api/chat), [OpenAI GPT-OSS on Ollama](https://developers.openai.com/cookbook/articles/gpt-oss/run-locally-ollama).
