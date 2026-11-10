import sys
from pathlib import Path

from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from speech_text import verbalize_numbers
from chatterbox_worker import text_for_voice
from chatterbox_worker import reference_for_voice
import hashlib
import pytest
import wave


def test_arabic_voice_preserves_english_names_and_document_numbers(monkeypatch):
    monkeypatch.setitem(sys.modules, "num2words", SimpleNamespace(num2words=lambda n, lang: f"<{n}>"))
    original = "افتح Microsoft Word 2026 وَرَاجِع ميزانية ٢٥٬٠٠٠ جنيه في GPT-OSS."
    assert text_for_voice(original, "ar") == "افتح Microsoft Word <2026> وراجع ميزانية <25000> جنيه في GPT-OSS."
    assert text_for_voice(original, "en") == original


def test_grouped_arabic_and_english_numbers_keep_their_value():
    seen = []

    def convert(number):
        seen.append(number)
        return "NUMBER"

    verbalize_numbers("25,000 جنيه و ٢٥٬٠٠٠٫٥ و 12.5 ثم 7, 8", convert)
    assert seen == ["25000", "25000.5", "12.5", "7", "8"]


def test_egyptian_profile_has_the_accepted_bundled_reference():
    reference = reference_for_voice({"tts": {"voice": "egyptian"}})
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == "55c8895ccad321913b85bfe4c4eafa613c1bf17c63b5943a10e2d6bfeb9c5f79"
    with wave.open(str(reference)) as audio:
        assert audio.getnchannels() == 1
        assert 3 < audio.getnframes() / audio.getframerate() < 10
    assert reference_for_voice({"tts": {"voice": "builtin"}}) is None


def test_arbitrary_reference_paths_are_not_voice_profiles(settings):
    with pytest.raises(ValueError, match="voice"):
        reference_for_voice({"tts": {"voice": "https://example.com/voice.wav"}})
    with pytest.raises(ValueError, match="voice"):
        settings.update({"speech": {"tts": {"voice": "C:/unselected-recording.wav"}}})
