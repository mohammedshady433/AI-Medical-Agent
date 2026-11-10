"""Lab queue correctness. These tests do not assert unmeasured hardware latency."""
import asyncio
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("voice_stream_probe", Path(__file__).parents[1] / "scripts/voice_stream_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("answer", [
    "ما توقفش المضخة. الضغط 2.5 بار؟ راجع القراءة الأول.",
    "Do not stop the pump. Pressure is 2.5 bar. Check it first!",
    "القراءة " + "1234567890" * 15 + " بار.",
])
def test_token_boundaries_preserve_text(answer):
    buffer, chunks = probe.SentenceBuffer(), []
    for char in answer:
        chunks.extend(buffer.feed(char))
    chunks.extend(buffer.feed("", final=True))
    assert " ".join(" ".join(chunks).split()) == " ".join(answer.split())
    assert not any(chunk.endswith("2.") for chunk in chunks)


async def test_stream_starts_synthesis_before_completion():
    speech_started = asyncio.Event()

    async def events():
        yield {"type": "token", "text": "ابدأ بأهم مهمة. "}
        await asyncio.wait_for(speech_started.wait(), 1)
        yield {"type": "token", "text": "خد استراحة بعدها."}
        yield {"type": "done", "content": "ابدأ بأهم مهمة. خد استراحة بعدها."}

    async def synthesize(text, index):
        speech_started.set()
        return b"audio", {}

    result = await probe.run_pipeline(events(), synthesize, streaming=True)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["ready_seconds"] < result["answer_seconds"]


async def test_baseline_releases_provider_before_bounded_queue():
    released = asyncio.Event()
    answer = "One. Two. Three. Four. Five."

    async def events():
        try:
            yield {"type": "token", "text": answer}
            yield {"type": "done", "content": answer}
        finally:
            released.set()

    async def synthesize(text, index):
        await released.wait()
        return b"audio", {}

    result = await asyncio.wait_for(probe.run_pipeline(events(), synthesize, streaming=False), 1)
    assert len(result["chunks"]) == 5


async def test_cancel_stops_active_synthesis_and_producer():
    running, stopped = asyncio.Event(), asyncio.Event()

    async def events():
        for _ in range(20):
            yield {"type": "token", "text": "A complete sentence. "}

    async def synthesize(text, index):
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(probe.run_pipeline(events(), synthesize, streaming=True))
    await asyncio.wait_for(running.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.parametrize("event", [{"type": "tool_call"}, {"type": "token", "text": "Claim [D1]"},
    {"type": "done", "content": "Rewritten answer"}])
async def test_unsupported_or_rewritten_drafts_fail_closed(event):
    async def events():
        yield event

    async def synthesize(text, index):
        return b"audio", {}

    with pytest.raises(ExceptionGroup):
        await probe.run_pipeline(events(), synthesize, streaming=True)


async def test_synthesis_failure_does_not_deadlock_full_queue():
    async def events():
        for _ in range(20):
            yield {"type": "token", "text": "Another sentence. "}

    async def synthesize(text, index):
        raise RuntimeError("GPU unavailable")

    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(probe.run_pipeline(events(), synthesize, streaming=True), 1)


def test_experimental_precision_is_restored_on_failure(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import benchmark_voice_worker as worker

    matmul = SimpleNamespace(allow_tf32=False)
    cuda = SimpleNamespace(is_bf16_supported=lambda: True, manual_seed_all=lambda n: None, synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, manual_seed=lambda n: None, set_num_threads=lambda n: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)), bfloat16="bf16",
        autocast=lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "torch", torch)
    original = lambda *args, **kwargs: None
    model = SimpleNamespace(t3=SimpleNamespace(inference=original), s3gen=SimpleNamespace(inference=original))
    monkeypatch.setattr(worker.approved, "load_model", lambda cfg: model)

    def fail(*args):
        assert matmul.allow_tf32
        raise RuntimeError("Synthesis failed")

    monkeypatch.setattr(worker.approved, "synthesize", fail)
    with pytest.raises(RuntimeError):
        worker.handle({"action": "synthesize", "config": {"tts": {"device": "cuda"}},
            "precision": "tf32", "seed": 1, "text": "Test", "language": "ar"})
    assert not matmul.allow_tf32
    assert model.t3.inference is original and model.s3gen.inference is original
