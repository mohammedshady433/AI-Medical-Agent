# API key setup

## Ollama Cloud

1. Sign in to your account at [Ollama](https://ollama.com).
2. Create an API key in the account's key settings. Consult [the current cloud guide](https://docs.ollama.com/cloud) for account access and usage details.
3. Start the core with `scripts/test_console.py`. Ollama Cloud at `https://ollama.com` is the default online provider.
4. In the test console open **Setup & keys**, paste the key into **Online provider API key** and click **Save key**. The field clears after saving; the API afterwards reports only whether a key exists. (Without the console: `POST /credentials {"provider": "ollama_cloud", "key": "..."}`, or run `scripts/key_setup.py`.)
5. Click **Test online**. This sends a small real inference request, not just a model-list lookup.
6. Select **Online** for typed chat. Document sharing with the cloud model is a separate setting: `PATCH /settings {"privacy": {"cloud_document_context": true}}`.

Each Windows account can use its own key. No developer key is shipped. Replacing the key requires no rebuild. Clearing the saved key removes that provider's DPAPI entry. The encrypted key file cannot simply be copied to another Windows account and decrypted there; enter a new key on the new computer.

## Development secrets

Copy `.env.example` to `.env` only if you prefer environment variables during development. Fill `OLLAMA_API_KEY`, `COMPATIBLE_API_KEY`, `AZURE_SPEECH_KEY`, or `BRAVE_SEARCH_API_KEY` as needed. `.env` is ignored by Git. Saved DPAPI/Keychain credentials take precedence. Never print key values from notebook cells, logs, support reports or exception messages.

## Other services

An OpenAI-compatible provider requires that provider's own API key and model access; it does not make GPT-OSS available through the OpenAI API. Azure voice requires an Azure Speech resource, matching region and key. Brave web research requires a Brave Search API key and its explicit privacy checkbox. These are separate accounts and charges.

## Diagnose failures

| Symptom | Meaning and next action |
|---|---|
| No key saved | Enter the key in the provider selected by the active model |
| HTTP 401/403 | Key expired, invalid, restricted, or missing model/account access; check the provider account |
| HTTP 404 | Wrong base endpoint or unavailable model ID; inspect the provider's real model list |
| HTTP 429 | Rate/usage limit; wait or adjust the account/plan, context and response limits |
| HTTP 5xx | Provider-side failure; retry later or manually select Offline |
| Cannot reach cloud provider | Could be connectivity, DNS, proxy or the provider; the app does not claim that all internet is down |
| Empty model list | No visible models were returned; listing permission and inference permission can differ |

All errors preserve the active mode and model. Choosing Offline is always your action.
