import base64
import json
import pytest

from app.speech.service import Speech


class NoSpeechNetwork:
    def get(self, *args):
        raise AssertionError("Local speech must not read a cloud credential")


@pytest.fixture
def fake_workers(monkeypatch, tmp_path, settings):
    workers = []

    class Process:
        returncode = None

        def __init__(self, command, options):
            self.command, self.options, self.requests = command, options, []
            self.stdin = self.stdout = self

        def write(self, data):
            self.requests.append(json.loads(data))

        async def drain(self):
            pass

        async def readline(self):
            request = self.requests[-1]
            if request["action"] == "transcribe":
                response = {"text": "الكلام اللي اتقال", "language": "ar"}
            elif request["action"] == "synthesize":
                response = {"audio": base64.b64encode(b"test-audio").decode(), "warnings": []}
            else:
                response = {"ready": True}
            return json.dumps(response).encode()

        def terminate(self):
            self.returncode = 0

        async def wait(self):
            return 0

    async def spawn(*command, **options):
        process = Process(command, options)
        workers.append(process)
        return process

    monkeypatch.setattr("app.speech.service.asyncio.create_subprocess_exec", spawn)
    for name in ("recognition.exe", "voice.exe"):
        (tmp_path / name).touch()
    settings.update(
        {
            "mode": "offline",
            "speech": {
                "tts": {"provider": "chatterbox"},
                "runtime": {
                    "python": str(tmp_path / "recognition.exe"),
                    "tts_python": str(tmp_path / "voice.exe"),
                },
            },
        }
    )
    return workers


async def test_replacement_voice_is_local_and_isolated_from_transcription(settings, fake_workers, tmp_path):
    speech = Speech(settings, NoSpeechNetwork())
    result = await speech.transcribe(tmp_path / "recording.webm", "ar")
    assert result["text"] == "الكلام اللي اتقال"
    audio, _ = await speech.synthesize("ميزانية التدريب 25,000 جنيه.", "ar")
    assert audio == b"test-audio"
    await speech.synthesize("Hello.", "en")
    assert len(fake_workers) == 2
    asr, tts = fake_workers
    assert asr.command[0].endswith("recognition.exe")
    assert asr.command[2].endswith("speech_worker.py")
    assert tts.command[0].endswith("voice.exe")
    assert tts.command[2].endswith("chatterbox_worker.py")
    assert [r["action"] for r in asr.requests] == ["transcribe", "synthesize"]
    assert tts.requests[0]["text"] == "ميزانية التدريب 25,000 جنيه."
    assert tts.requests[0]["mode"] == "offline"
    for worker in fake_workers:
        assert worker.options["env"]["HF_HUB_OFFLINE"] == "1"
        assert worker.options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    await speech.close()
    assert all(w.returncode == 0 for w in fake_workers)


async def test_replacement_warmup_does_not_load_rejected_voice(settings, fake_workers):
    speech = Speech(settings, NoSpeechNetwork())
    await speech.local("warmup")
    assert [w.requests[0]["action"] for w in fake_workers] == ["warmup_asr", "warmup"]
    assert "chatterbox_worker.py" in fake_workers[1].command[2]
    await speech.close()


async def test_missing_replacement_runtime_fails_without_fallback(settings):
    settings.update({"speech": {"tts": {"provider": "chatterbox"}}})
    speech = Speech(settings, NoSpeechNetwork())
    with pytest.raises(ValueError, match="Settings.*Set up"):
        await speech.synthesize("مرحبا", "ar")
    assert settings.snapshot()["speech"]["tts"]["provider"] == "chatterbox"


def test_voice_readiness_requires_complete_selected_checkpoint(settings, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    settings.update(
        {
            "speech": {
                "tts": {"provider": "chatterbox", "chatterbox_version": "v3"},
                "runtime": {
                    "python": str(runtime / "asr.exe"),
                    "tts_python": str(runtime / "tts.exe"),
                    "whisper_path": str(runtime),
                    "chatterbox_path": str(runtime),
                },
            }
        }
    )
    speech = Speech(settings, NoSpeechNetwork())
    assert not speech.status("ar")["ready"]
    for name in ("asr.exe", "model.bin", "config.json", "tokenizer.json"):
        (runtime / name).write_bytes(b"fixture")
    assert speech.status("ar")["recognition"]
    assert not speech.status("ar")["synthesis"]
    assert speech.status("en")["ready"]
    for name in (
        "tts.exe",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "ve.pt",
        "conds.pt",
        "grapheme_mtl_merged_expanded_v1.json",
    ):
        (runtime / name).write_bytes(b"fixture")
    assert speech.status("ar")["ready"]
    settings.update({"speech": {"tts": {"chatterbox_version": "v2"}}})
    assert not speech.status("ar")["synthesis"]
    assert speech.status("en")["ready"]


def test_disabled_speech_is_not_reported_ready(settings, fake_workers):
    settings.update({"speech": {"tts": {"provider": "none"}}})
    assert not Speech(settings, NoSpeechNetwork()).status("en")["synthesis"]


async def test_uploaded_recording_is_sent_without_a_temporary_path(settings, fake_workers):
    speech = Speech(settings, NoSpeechNetwork())
    await speech.transcribe(b"recording bytes", "ar")
    request = fake_workers[0].requests[0]
    assert request["audio"] == "cmVjb3JkaW5nIGJ5dGVz"
    assert "path" not in request


async def test_transcription_restarts_worker_after_broken_pipe(settings, fake_workers, monkeypatch):
    speech = Speech(settings, NoSpeechNetwork())
    await speech.transcribe(b"first", "ar")
    worker = fake_workers[0]

    async def broken_pipe():
        worker.returncode = 1
        raise BrokenPipeError()

    monkeypatch.setattr(worker, "drain", broken_pipe)
    assert (await speech.transcribe(b"retry", "ar"))["text"]
    assert len(fake_workers) == 2
    assert fake_workers[1].requests[0]["audio"] == "cmV0cnk="
