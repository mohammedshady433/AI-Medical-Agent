"""Experimental sentence queue for benchmarks only; not connected to EMS voice mode."""
import asyncio
import re
import time


class SentenceBuffer:
    """Wait for an unambiguous boundary; preserve decimals, numbers and negations."""

    def __init__(self, limit=120):
        self.pending = ""
        self.limit = limit

    def feed(self, text, final=False):
        self.pending += text
        pieces = []
        while self.pending:
            boundary = re.search(r"[.!?؟](?=\s)|\n", self.pending)
            cut = boundary.end() if boundary else 0
            if len(self.pending) > self.limit and (not cut or cut > self.limit):
                spaces = list(re.finditer(r"\s+", self.pending[:self.limit]))
                # Do not split a number or identifier that exceeds the limit.
                cut = spaces[-1].end() if spaces else 0
            if not cut:
                if not final:
                    break
                cut = len(self.pending)
            piece, self.pending = self.pending[:cut], self.pending[cut:]
            if piece.strip():
                pieces.append(piece.strip())
        return pieces


async def run_pipeline(events, synthesize, *, streaming, clock=time.perf_counter):
    """Bounded producer/consumer with fail-fast cleanup and no background audio after cancellation.

    This lab path rejects tools and references: incremental draft text cannot yet be
    trusted for document answers or external actions. It never plays audio itself.
    """
    started = clock()
    queue = asyncio.Queue(maxsize=2)
    buffer = SentenceBuffer()
    report = {"chunks": [], "answer": "", "first_token_seconds": None}
    draft = ""

    async def produce():
        nonlocal draft
        completed = False
        async for event in events:
            kind = event["type"]
            if kind == "error":
                raise RuntimeError(event.get("message", "Answer failed"))
            if kind == "token":
                if report["first_token_seconds"] is None:
                    report["first_token_seconds"] = clock() - started
                draft += event["text"]
                if "[D" in draft:
                    raise ValueError("Document citations require final validation; excluded from this probe.")
                if streaming:
                    for piece in buffer.feed(event["text"]):
                        await queue.put((piece, clock() - started))
            elif kind == "done":
                report["answer_seconds"] = clock() - started
                report["answer"] = event["content"]
                if event["content"].strip() != draft.strip():
                    raise ValueError("Final answer differs from streamed draft; do not speak it incrementally.")
                if streaming:
                    for piece in buffer.feed("", final=True):
                        await queue.put((piece, clock() - started))
                completed = True
            else:
                raise ValueError(f"This conversation-only probe does not support event {kind!r}.")
        if not completed:
            raise ValueError("Answer ended without a completion event.")
        # Release the assistant's GPU context before baseline synthesis takes its lock.
        if not streaming:
            for piece in buffer.feed(report["answer"], final=True):
                await queue.put((piece, clock() - started))
        await queue.put(None)

    async def consume():
        playback_end = 0
        next_prefetch = 0
        while True:
            item = await queue.get()
            if item is None:
                return
            text, queued = item
            # Match one-ahead playback: do not synthesize the entire answer into an
            # unbounded audio backlog while the first sentence is still speaking.
            await asyncio.sleep(max(0, next_prefetch - clock()))
            begin = clock()
            audio, details = await synthesize(text, len(report["chunks"]))
            ready = clock()
            play_start = max(playback_end, ready)
            playback_end = play_start + details.get("duration_seconds", 0)
            next_prefetch = play_start
            report["chunks"].append({
                "text": text, "queued_seconds": queued,
                "synthesis_seconds": clock() - begin,
                "ready_seconds": ready - started,
                "audio": audio, **details,
            })

    # TaskGroup cancels the other side on failure, even if it is blocked on a full queue.
    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        group.create_task(consume())
    return report
