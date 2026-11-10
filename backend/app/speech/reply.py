"""Prepare one local clip early; release audio only after final answer validation."""
import asyncio
import base64
from contextlib import aclosing
import re
import sys
import time

from app.core.options import resident_voice

IS_MACOS = sys.platform == "darwin"


def speech_text(text):
    # Numbered Markdown items are layout, not speech or decimal values.
    text = re.sub(r"^[ \t]*[0-9٠-٩]+[.)][ \t]+", "", text, flags=re.MULTILINE)
    return re.sub(r"\[D\d+\]|[`#*_]", "", text).strip()


def speech_chunks(text, short_opening=False):
    """Keep in sync with speechChunks() in test-console/index.html; fixtures cover Arabic and English."""
    clean = speech_text(text)
    sentences = re.findall(r"[\s\S]+?(?:[.!?؟](?=\s|$)|\n+|$)", clean)
    result = []
    opening_limit = 56 if short_opening else 80
    pause_start = 24 if short_opening else 32
    for piece in sentences:
        while piece:
            limit = opening_limit if not result else 120
            # A complete opening clause is faster to render without changing words.
            pause = re.search(r"[,،;؛](?=\s)", piece[pause_start:limit]) if not result else None
            cut = pause_start + pause.end() if pause else 0
            if not cut and len(piece) > limit:
                space = re.search(r"\s+\S*$", piece[:limit])
                cut = space.start() + 1 if space else limit
            if not cut:
                cut = len(piece)
            result.append(piece[:cut])
            piece = piece[cut:]
    return [piece for piece in result if piece.strip()]


def buffered_egyptian_voice(config):
    voice = config["speech"]["tts"]
    return IS_MACOS and config["language"] == "ar" and voice["provider"] == "chatterbox" and voice["voice"] == "egyptian"


def stable_first_chunk(draft, short_opening=False):
    """Do not guess whether the last punctuation/token is a complete boundary."""
    pieces = speech_chunks(draft, short_opening)
    if not pieces:
        return None
    first = pieces[0]
    # The next non-whitespace character proves that this boundary is stable.
    clean = speech_text(draft)
    if len(clean) > len(first.rstrip()) and clean[len(first.rstrip()):].strip():
        return first.strip()
    return None


async def prepared_voice_reply(events, speech, gpu, config, cancel):
    """No draft audio leaves the server. At most one local synthesis is in flight."""
    resident = resident_voice(config)
    language = config["language"]
    voice = config["speech"]["tts"]
    # macOS Chatterbox runs on CPU, independently of Ollama's Metal layers.
    mac_cpu_voice = IS_MACOS and voice["provider"] == "chatterbox" and voice["device"] == "cpu"
    enabled = (resident or language == "en" or mac_cpu_voice) and voice["provider"] not in ("azure", "none")
    task, candidate, draft = None, None, ""
    # The client plays a single continuous clip for Egyptian Arabic replies.
    # Preparing a discarded opening would make that full clip take longer.
    short_opening = buffered_egyptian_voice(config)
    enabled = enabled and not short_opening
    allowed = enabled
    started = time.perf_counter()
    preview_started = None
    cancelled = asyncio.create_task(cancel.wait())

    async def discard():
        nonlocal task
        if task is not None:
            if not task.done() and not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            task = None

    try:
        async with gpu.voice_turn(enabled and resident):
            try:
                async with aclosing(cancellable_events(events, cancelled)) as replies:
                    async for event in replies:
                        kind = event["type"]
                        if kind == "token" and allowed and task is None:
                            draft += event["text"]
                            candidate = stable_first_chunk(draft, short_opening)
                            if candidate:
                                preview_started = time.perf_counter() - started
                                task = asyncio.create_task(speech.synthesize(
                                    candidate, language, local_only=True))
                        elif kind in ("status", "approval", "error", "cancelled"):
                            allowed = enabled and kind in ("status", "approval")
                            await discard()
                            draft, candidate = "", None
                        elif kind == "done" and task is not None:
                            pieces = speech_chunks(event.get("content", ""), short_opening)
                            if not pieces or pieces[0].strip() != candidate or cancel.is_set():
                                await discard()
                            else:
                                await asyncio.wait({task, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                                if cancel.is_set():
                                    await discard()
                                    yield {"type": "cancelled"}
                                    return
                                try:
                                    audio, warnings = task.result()
                                    # Ephemeral event data: never persist audio in conversation history.
                                    event = {**event, "voice_audio": {
                                        "text": candidate, "language": language,
                                        "audio": base64.b64encode(audio).decode("ascii"), "warnings": warnings,
                                    }, "voice_timing": {"preview_started_seconds": preview_started,
                                                        "ready_seconds": time.perf_counter() - started}}
                                except (Exception, asyncio.CancelledError):
                                    pass  # Normal synthesis remains available if preview failed.
                        if cancel.is_set() and kind == "done":
                            yield {"type": "cancelled"}
                            return
                        yield event
            finally:
                await discard()  # Before releasing the GPU lease, including stream disconnect.
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)


async def cancellable_events(events, cancelled):
    """An interruption also stops a model that is silently thinking or waiting."""
    pending = None
    async with aclosing(events):
        try:
            while True:
                pending = asyncio.create_task(anext(events))
                await asyncio.wait({pending, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                if cancelled.done():
                    if not pending.done() and not pending.cancelling():
                        pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    yield {"type": "cancelled"}
                    return
                try:
                    yield pending.result()
                except StopAsyncIteration:
                    return
        finally:
            if pending is not None:
                if not pending.done() and not pending.cancelling():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
