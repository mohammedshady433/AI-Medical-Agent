import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def load(monkeypatch):
    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("cpu_test", scripts / "chatterbox_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_cache_and_original_forward_restored_on_cancel(monkeypatch):
    module = load(monkeypatch)
    embeds = Mock()
    embeds.shape = (2, 8, 64)
    embeds.__getitem__ = Mock(return_value="one-row")
    last = Mock()
    last.expand.return_value = "shared-two-rows"
    result = SimpleNamespace(last_hidden_state=last, hidden_states=None, past_key_values="single-cache")
    original = Mock(return_value=result)
    model = SimpleNamespace(forward=original)
    with pytest.raises(InterruptedError):
        with module.single_batch(model, True):
            actual = model.forward(inputs_embeds=embeds, past_key_values="single-cache")
            assert actual.hidden_states == ("shared-two-rows",)
            assert actual.past_key_values == "single-cache"
            assert original.call_args.kwargs["inputs_embeds"] == "one-row"
            assert original.call_args.kwargs["output_hidden_states"] is False
            raise InterruptedError()
    assert model.forward is original


def test_unexpected_vendor_shape_requests_fresh_fallback(monkeypatch):
    module = load(monkeypatch)
    from speech_cancellation import SpeechWorkerRestart
    original = Mock()
    model = SimpleNamespace(forward=original)
    with pytest.raises(SpeechWorkerRestart):
        with module.single_batch(model, True):
            model.forward(inputs_embeds=SimpleNamespace(shape=(3, 1, 8)))
    original.assert_not_called()
    assert model.forward is original


@pytest.mark.parametrize("device,voice,version", [("cuda", "builtin", "v3"), ("cpu", "egyptian", "v3"), ("cpu", "builtin", "v2")])
def test_other_voices_and_devices_keep_original_decoder(monkeypatch, device, voice, version):
    module = load(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    assert not module.supported(None, {"tts": {"device": device, "voice": voice, "chatterbox_version": version}})
