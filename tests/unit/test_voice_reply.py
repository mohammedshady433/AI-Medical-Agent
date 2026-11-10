import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from app.services.gpu import GpuCoordinator
from app.speech.reply import prepared_voice_reply, speech_chunks, stable_first_chunk


@pytest.mark.parametrize("language,text", [
    ("ar", "ما توقفش المضخة قبل مراجعة التعليمات. الضغط 2.5 بار."),
    ("en", "Check the operating instructions first. The pressure is 2.5 bar."),
])
async def test_preview_overlaps_but_is_released_only_after_final_answer(settings, language, text, monkeypatch):
    monkeypatch.setattr("app.speech.reply.IS_MACOS", False)
    settings.update({"language": language, "assistant": {"voice_low_latency": True}})
    gpu = GpuCoordinator(settings)
    started, finish = asyncio.Event(), asyncio.Event()
    async def synthesize(*args, **kwargs):
        async with gpu.speech(True):
            started.set()
            return b"wav", []
    speech = type("Speech", (), {"synthesize": staticmethod(synthesize)})()
    async def events():
        async with gpu.assistant({"mode": "offline", "_voice": True}):
            yield {"type": "token", "text": text}
            await started.wait()
            await finish.wait()
            yield {"type": "done", "content": text}
    received = []
    async def run():
        async for event in prepared_voice_reply(events(), speech, gpu, settings.snapshot(), asyncio.Event()):
            received.append(event)
    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 1)
    assert not any("voice_audio" in e for e in received)
    assert gpu.lock.locked()
    finish.set()
    await asyncio.wait_for(task, 1)
    assert received[-1]["voice_audio"]["text"] == speech_chunks(text)[0].strip()
    assert not gpu.lock.locked()


@pytest.mark.parametrize("case", ["rewrite", "tool", "approval", "error", "failure", "cancel", "disconnect"])
async def test_preview_never_leaks_on_changes_failures_or_cancellation(settings, case, monkeypatch):
    monkeypatch.setattr("app.speech.reply.IS_MACOS", False)
    settings.update({"assistant": {"voice_low_latency": True}})
    gpu = GpuCoordinator(settings)
    started, stopped, model_stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancel = asyncio.Event()
    async def synthesize(*args, **kwargs):
        started.set()
        try:
            if case == "failure": raise ValueError("voice error")
            await asyncio.Event().wait()
        finally: stopped.set()
    speech = type("Speech", (), {"synthesize": staticmethod(synthesize)})()
    text = "Don't shut down the pump. Read the instructions."
    async def events():
        try:
            yield {"type": "token", "text": text}
            await started.wait()
            if case in ("cancel", "disconnect"):
                await asyncio.Event().wait()  # Model produces no more tokens.
            if case in ("tool", "approval", "error"):
                yield {"type": "status" if case == "tool" else case}
            yield {"type": "done", "content": "Corrected answer." if case == "rewrite" else text}
        finally: model_stopped.set()
    async def run():
        return [e async for e in prepared_voice_reply(events(), speech, gpu, settings.snapshot(), cancel)]
    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 1)
    if case == "cancel": cancel.set()
    if case == "disconnect": task.cancel()
    if case == "disconnect":
        with pytest.raises(asyncio.CancelledError): await task
    else:
        result = await asyncio.wait_for(task, 1)
        assert not any("voice_audio" in e for e in result)
    assert stopped.is_set() and model_stopped.is_set() and not gpu.lock.locked()


async def test_low_memory_arabic_keeps_sequential_path(settings, monkeypatch):
    monkeypatch.setattr("app.speech.reply.IS_MACOS", False)
    settings.update({"language": "ar", "assistant": {"voice_low_latency": False}})
    speech = type("Speech", (), {"synthesize": AsyncMock()})()
    async def events():
        yield {"type": "token", "text": "أهلاً بيك. نبدأ بمراجعة التقرير."}
        yield {"type": "done", "content": "أهلاً بيك. نبدأ بمراجعة التقرير."}
    result = [e async for e in prepared_voice_reply(events(), speech, GpuCoordinator(settings), settings.snapshot(), asyncio.Event())]
    speech.synthesize.assert_not_called()
    assert "voice_audio" not in result[-1]


async def test_fresh_preview_after_retrieval_still_requires_final_text(settings, monkeypatch):
    monkeypatch.setattr("app.speech.reply.IS_MACOS", False)
    settings.update({"assistant": {"voice_low_latency": True}})
    speech = type("Speech", (), {"synthesize": AsyncMock(return_value=(b"wav", []))})()
    text = "الضغط 2.5 بار [D12]. راجع تعليمات التشغيل قبل أي تعديل."
    async def events():
        yield {"type": "status", "tool": "search_documents"}
        yield {"type": "token", "text": text}
        yield {"type": "done", "content": text, "meta": {"sources": [{"reference": "[D12]"}]}}
    result = [e async for e in prepared_voice_reply(events(), speech, GpuCoordinator(settings), settings.snapshot(), asyncio.Event())]
    assert result[-1]["voice_audio"]["text"] == "الضغط 2.5 بار ."
    assert result[-1]["meta"]["sources"][0]["reference"] == "[D12]"


def test_chunk_contract_and_boundary_lookahead():
    cases = json.loads((Path(__file__).parents[1] / "fixtures/voice/chunks.json").read_text(encoding="utf8"))
    for case in cases:
        assert speech_chunks(case["text"]) == case["chunks"]
    assert stable_first_chunk("Pressure is 2.") is None
    assert stable_first_chunk("Pressure is 2.5 bar.") is None
    assert stable_first_chunk("Pressure is 2.5 bar. Next") == "Pressure is 2.5 bar."


async def test_mac_egyptian_voice_skips_the_unused_preview(settings, monkeypatch):
    monkeypatch.setattr("app.speech.reply.IS_MACOS", True)
    settings.update({
        "language": "ar",
        "speech": {"tts": {"provider": "chatterbox", "voice": "egyptian", "device": "cpu"}},
    })
    speech = type("Speech", (), {"synthesize": AsyncMock()})()
    text = "راجع فاتورة الكهرباء الأول. قارن الاستهلاك بالشهر اللي فات."

    async def events():
        yield {"type": "token", "text": text}
        yield {"type": "done", "content": text}

    result = [e async for e in prepared_voice_reply(events(), speech, GpuCoordinator(settings), settings.snapshot(), asyncio.Event())]
    speech.synthesize.assert_not_called()
    assert result[-1]["content"] == text
    assert "voice_audio" not in result[-1]


def test_egyptian_short_opening_preserves_the_entire_reply():
    text = "قارن المرتب الصافي بعد الخصومات الأول، وبعدها شوف التأمين والإجازات ووقت المواصلات."
    chunks = speech_chunks(text, short_opening=True)
    assert chunks[0] == "قارن المرتب الصافي بعد الخصومات الأول،"
    assert "".join(chunks) == text
    assert stable_first_chunk(text, short_opening=True) == chunks[0]


def test_spoken_chunks_skip_list_markers_but_keep_decimals():
    assert speech_chunks("1. المرتب ٢.٥ ألف.\n٢) شوف المزايا.") == [
        "المرتب ٢.٥ ألف.", "\nشوف المزايا."
    ]
