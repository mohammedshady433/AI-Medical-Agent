import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest


def worker(name, monkeypatch):
    folder = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("warmup_test_" + name, folder / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recognition_primes_public_audio_once_and_discards_transcript(monkeypatch):
    module = worker("speech_worker", monkeypatch)
    monkeypatch.setattr(module, "load_asr", Mock())
    transcribe = Mock(return_value={"text": "discarded"})
    monkeypatch.setattr(module, "transcribe", transcribe)
    request = {"action": "warmup_asr", "config": {"asr": {"device": "cuda"}, "keep_resident": True}}
    for _ in range(2):
        result = module.handle(request)
        assert "text" not in result and "audio" not in result
    transcribe.assert_called_once()
    assert Path(transcribe.call_args.args[0]["path"]).name == "asr-warmup.wav"


@pytest.mark.parametrize("device,resident", [("cuda", True), ("cpu", False)])
def test_approved_voice_primes_once_without_returning_or_playing_audio(monkeypatch, device, resident):
    module = worker("chatterbox_worker", monkeypatch)
    monkeypatch.setattr(module, "load_model", Mock())
    synthesize = Mock(return_value=b"discarded-wave")
    monkeypatch.setattr(module, "synthesize", synthesize)
    cfg = {"tts": {"device": device, "chatterbox_version": "v3"}, "keep_resident": resident}
    for _ in range(2):
        result = module.handle({"action": "warmup", "config": cfg})
        assert "audio" not in result
    synthesize.assert_called_once_with("أهلاً.", "ar", cfg, None)


def test_gpu_offload_leaves_cpu_recognition_loaded(monkeypatch):
    module = worker("speech_worker", monkeypatch)
    model = Mock()
    monkeypatch.setattr(module, "ASR", model)

    module.handle({"action": "offload", "config": {"asr": {"device": "cpu"}}})
    model.model.unload_model.assert_not_called()

    module.handle({"action": "offload", "config": {"asr": {"device": "cuda"}}})
    model.model.unload_model.assert_called_once_with(to_cpu=True)
