from fastapi.testclient import TestClient
from app.main import create_app


def test_voice_preview_is_ephemeral_and_typed_chat_does_not_synthesize(tmp_path):
    import json
    from unittest.mock import AsyncMock
    class Provider:
        adapter = "ollama"
        def __init__(self, *args): pass
        async def stream(self, messages, tools=()):
            yield {"type": "token", "text": "Check the operating instructions first. Then review the report."}
    app = create_app(tmp_path, "token", provider_factory=Provider)
    app.state.settings.update({"language": "en", "mode": "offline"})
    app.state.speech.synthesize = AsyncMock(return_value=(b"prepared-wav", []))
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer token"}
        chat = client.post("/chats", headers=headers).json()["id"]
        def send(voice):
            response = client.post("/chat", headers=headers, json={"chat_id": chat, "text": "What should I check?", "voice": voice})
            assert response.status_code == 200
            return [json.loads(line) for line in response.text.splitlines()]
        assert "voice_audio" not in send(False)[-1]
        app.state.speech.synthesize.assert_not_called()
        assert send(True)[-1]["voice_audio"]["text"] == "Check the operating instructions first."
        assert "voice_audio" not in client.get("/chats/" + chat, headers=headers).text
        app.state.speech.synthesize.assert_awaited_once()


def test_unauthenticated_cannot_read_or_change_settings(tmp_path):
    with TestClient(create_app(tmp_path, "token")) as client:
        assert client.get("/settings").status_code == 401
        assert client.get("/speech/status?language=ar").status_code == 401
        assert client.patch("/settings", json={"mode": "offline"}).status_code == 401


def test_settings_and_documents_persist_across_modes(tmp_path):
    headers = {"Authorization": "Bearer token"}
    with TestClient(create_app(tmp_path, "token")) as client:
        assert client.get("/health", headers=headers).json()["data_dir"] == str(tmp_path)
        readiness = client.get("/speech/status?language=ar", headers=headers)
        assert readiness.status_code == 200
        assert readiness.json() == {
            "ready": False,
            "recognition": False,
            "synthesis": False,
            "provider": "chatterbox",
        }
        chat = client.post("/chats", headers=headers).json()["id"]
        assert client.patch("/settings", headers=headers, json={"mode": "offline"}).status_code == 200
        assert client.get("/chats", headers=headers).json()[0]["id"] == chat
    with TestClient(create_app(tmp_path, "token")) as client:
        assert client.get("/settings", headers=headers).json()["config"]["mode"] == "offline"


def test_upload_rejects_executable(tmp_path):
    with TestClient(create_app(tmp_path, "token")) as client:
        assert (
            client.post(
                "/documents", headers={"Authorization": "Bearer token"}, files={"file": ("bad.exe", b"mz")}
            ).status_code
            == 400
        )


def test_unexpected_failure_is_json_and_logged(tmp_path, monkeypatch, caplog):
    import app.main as main

    def fail():
        raise RuntimeError("broken test resource")

    monkeypatch.setattr(main, "build_info", fail)
    with TestClient(create_app(tmp_path, "token"), raise_server_exceptions=False) as client:
        response = client.get("/health", headers={"Authorization": "Bearer token"})
    assert response.status_code == 500
    assert "error reference" in response.json()["message"]
    assert "broken test resource" not in response.json()["message"]
    assert "broken test resource" in caplog.text


def test_production_secret_storage_is_encrypted(tmp_path, monkeypatch):
    from app.core.secrets import SecretStore

    import sys
    if sys.platform == "darwin":
        rows = {}
        class Keychain:
            def get_password(self, service, name): return rows.get((service, name))
            def set_password(self, service, name, value): rows[service, name] = value
            def delete_password(self, service, name): rows.pop((service, name), None)
        monkeypatch.setattr(SecretStore, '_keychain', staticmethod(Keychain))
    store = SecretStore(tmp_path)
    store.save("test", "not-a-real-api-key")
    assert store.get("test") == "not-a-real-api-key"
    assert not store.path.exists() if sys.platform == "darwin" else "not-a-real-api-key" not in store.path.read_text()
    store.save("test", "")
    assert store.get("test") == ""



def test_setup_and_removed_game_routes(tmp_path):
    headers = {'Authorization': 'Bearer token'}
    with TestClient(create_app(tmp_path, 'token')) as client:
        assert client.get('/setup/status').status_code == 401
        assert client.post('/setup/install', headers=headers, json={'component': 'voice', 'accepted_download': False}).status_code == 400
        assert client.post('/setup/install', headers=headers, json={'component': 'arbitrary', 'accepted_download': True}).status_code == 422
        assert client.get('/desktop/games', headers=headers).status_code == 404
        assert client.post('/desktop/games/launch', headers=headers, json={'id': '620'}).status_code == 404


def test_settings_survive_real_restart_with_voice_runtime(tmp_path):
    import json
    headers = {'Authorization': 'Bearer token'}
    settings = {'speech': {'runtime': {'python': str(tmp_path / 'speech/python.exe'), 'whisper_path': str(tmp_path / 'whisper')}}, 'assistant': {'voice_model': 'qwen2.5:7b-instruct'}}
    with TestClient(create_app(tmp_path, 'token')) as client:
        assert client.patch('/settings', headers=headers, json=settings).status_code == 200
    assert json.loads((tmp_path / 'settings.json').read_text())['speech']['runtime']['python'] == settings['speech']['runtime']['python']
    with TestClient(create_app(tmp_path, 'token')) as client:
        cfg = client.get('/settings', headers=headers).json()['config']
        assert cfg['speech']['runtime']['python'] == settings['speech']['runtime']['python']
        assert cfg['assistant']['voice_model'] == 'qwen2.5:7b-instruct'
