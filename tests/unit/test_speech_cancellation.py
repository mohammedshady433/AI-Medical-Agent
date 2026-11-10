import asyncio
import io
import json
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from app.core.settings import Settings
from app.speech.service import Speech

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from speech_cancellation import serve, check_cancelled, cancellable_inference, SpeechCancelled


def test_worker_cancels_one_request_and_keeps_serving(tmp_path):
    flag = tmp_path / "cancel.flag"; flag.touch()
    def lines():
        yield json.dumps({"action": "synthesize", "request_id": "first", "cancel_file": str(flag)})
        yield json.dumps({"action": "warmup", "request_id": "second"})
    def handle(request, event):
        assert request["action"] == "warmup"
        return {"ready": True}
    output = io.StringIO()
    serve(handle, lines(), output)
    assert [json.loads(x) for x in output.getvalue().splitlines()] == [{"cancelled": True}, {"ready": True}]


def test_worker_requests_replacement_after_failed_graph_capture():
    from speech_cancellation import SpeechWorkerRestart
    def handle(request, event):
        raise SpeechWorkerRestart('Invalid capture')
    output = io.StringIO()
    serve(handle, [json.dumps({'action': 'synthesize'})], output)
    assert json.loads(output.getvalue()) == {'retry_without_decoder_graph': True}


def test_settings_upgrade_preserves_new_user_preference_and_custom_pause(tmp_path):
    settings = Settings(tmp_path)
    stored = settings.snapshot()
    stored["assistant"].pop("voice_detection_version")
    stored["assistant"].update(barge_in=False, voice_pause_ms=700)
    settings.path.write_text(json.dumps(stored))
    settings.reload()
    assert settings.snapshot()["assistant"]["barge_in"]
    assert settings.snapshot()["assistant"]["voice_pause_ms"] == 500
    settings.update({"assistant": {"barge_in": False, "voice_pause_ms": 900}})
    assert not Settings(tmp_path).snapshot()["assistant"]["barge_in"]
    assert Settings(tmp_path).snapshot()["assistant"]["voice_pause_ms"] == 900


async def test_cancel_drains_old_response_without_restarting_worker(settings, secrets):
    import types
    cfg = settings.snapshot()["speech"]
    cfg["runtime"]["tts_python"] = sys.executable
    reader = asyncio.StreamReader()
    started = asyncio.Event()
    requests = []
    def write(data):
        request = json.loads(data); requests.append(request)
        if request["action"] == "warmup": reader.feed_data(b'{"ready": true}\n')
        else: started.set()
    process = types.SimpleNamespace(returncode=None, stdin=types.SimpleNamespace(write=write, drain=AsyncMock()), stdout=reader)
    speech = Speech(settings, secrets, "chatterbox")
    speech.process = process; speech.signature = json.dumps(cfg, sort_keys=True)
    task = asyncio.create_task(speech._request(cfg, "synthesize", text="مرحبا", language="ar"))
    await started.wait()
    async def worker_cancel():
        while not Path(requests[0]["cancel_file"]).is_file(): await asyncio.sleep(0.001)
        reader.feed_data(b'{"cancelled": true}\n')
    ack = asyncio.create_task(worker_cancel()); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert speech.process is process
    assert await speech._request(cfg, "warmup") == {"ready": True}
    await ack
    assert not Path(requests[0]["cancel_file"]).exists()
