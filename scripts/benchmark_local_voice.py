"""Opt-in local voice timings using public recordings, never cloud keys or private files."""
import argparse
import asyncio
import json
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf8")
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.storage.database import Database
from app.rag.library import Library
from app.agents.assistant import Assistant, voice_config
from app.speech.service import Speech
from app.services.gpu import GpuCoordinator
from app.services.local_runtime import LocalRuntime
from app.providers.client import Provider


class NoCloudKeys:
    def get(self, *args): raise AssertionError("This benchmark must never request a cloud key.")


def first_chunk(text):
    clean = re.sub(r"\[D\d+\]|[`#*_]", "", text).strip()
    sentence = re.split(r"(?<=[.!?؟])\s|\n", clean, maxsplit=1)[0]
    if len(sentence) > 120:
        cut = sentence.rfind(" ", 0, 121)
        sentence = sentence[:cut if cut > 0 else 120]
    return sentence


async def benchmark(args):
    source = Settings(Path(args.data_dir)).snapshot()
    report = {"physical_microphone_test": False, "cloud_requests": 0, "includes_end_of_turn_detection": False, "measure": "Recorded audio upload equivalent to first playable reply chunk; microphone pause and playback startup are additional.", "runs": []}
    target = ROOT / "logs" / args.output
    with tempfile.TemporaryDirectory(prefix="voice-benchmark-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder)); source["mode"] = "offline"
        source["privacy"].update(cloud_document_context=False, web_search=False)
        source["assistant"]["mcp_servers"] = []
        if args.asr_cpu: source["speech"]["asr"].update(device="cpu", compute_type="int8")
        settings.update(source)
        runtime = LocalRuntime(settings); await runtime.start()
        gpu = GpuCoordinator(settings); secrets = NoCloudKeys()
        speech = Speech(settings, secrets, gpu=gpu); gpu.offload_speech = speech.offload
        db = Database(settings.data)
        assistant = Assistant(settings, secrets, db, Library(db, settings), gpu=gpu)
        try:
            for model in args.models:
                await speech.close(); await gpu.release()
                settings.update({"assistant": {"voice_model": model, "voice_low_latency": not args.conservative}})
                prepare = time.perf_counter()
                await speech.local("warmup")
                cfg = voice_config(settings.snapshot())
                async with gpu.assistant(cfg):
                    await Provider(cfg, secrets).preload()
                preparation = round(time.perf_counter() - prepare, 3)
                print(json.dumps({"model": model, "preparation_seconds": preparation}), flush=True)
                for language in args.languages:
                    settings.update({"language": language})
                    fixture = ROOT / "logs" / ("voice-question-" + language + ".wav")
                    if not fixture.is_file(): raise ValueError("Supply public fixtures logs/voice-question-ar.wav and voice-question-en.wav before benchmarking.")
                    for turn in range(args.runs):
                        row = {"model": model, "language": language, "turn": turn + 1, "preparation_seconds": preparation}
                        started = time.perf_counter()
                        transcript = await speech.transcribe(fixture, language)
                        row["asr_seconds"] = round(time.perf_counter() - started, 3)
                        row["heard"] = transcript["text"]
                        answer_started = time.perf_counter(); done = None
                        async for event in assistant.reply(db.new_chat(), transcript["text"], voice=True):
                            if event["type"] == "done": done = event
                            if event["type"] == "error": raise RuntimeError(event["message"])
                        if not done: raise RuntimeError("No answer returned.")
                        row["answer_seconds"] = round(time.perf_counter() - answer_started, 3)
                        row["first_token_seconds"] = done["meta"].get("first_token_seconds")
                        row["answer"] = done["content"]
                        chunk = first_chunk(done["content"])
                        row["cache_entries_before_speech"] = len(speech.audio_cache)
                        speech_started = time.perf_counter()
                        # Fresh synthesis is the useful latency measure, not a repeated WAV cache hit.
                        speech.audio_cache.clear(); speech.cache_bytes = 0
                        audio, warnings = await speech.synthesize(chunk, language, local_only=True)
                        row["first_audio_seconds"] = round(time.perf_counter() - started, 3)
                        row["tts_first_chunk_seconds"] = round(time.perf_counter() - speech_started, 3)
                        row["spoken_chunk"], row["warnings"], row["audio_bytes"] = chunk, warnings, len(audio)
                        label = model.replace(":", "-") + "-" + language + "-" + str(turn + 1)
                        (ROOT / "logs" / ("benchmark-" + label + ".wav")).write_bytes(audio)
                        report["runs"].append(row)
                        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
                        print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            await speech.close(); await gpu.release(); await runtime.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=["gpt-oss:20b", "qwen2.5:7b-instruct"], default=["gpt-oss:20b", "qwen2.5:7b-instruct"])
    parser.add_argument("--languages", nargs="+", choices=["ar", "en"], default=["ar", "en"])
    parser.add_argument("--runs", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--asr-cpu", action="store_true")
    parser.add_argument("--conservative", action="store_true", help="Disable resident speech/partial-GPU voice optimization for a baseline")
    parser.add_argument("--output", default="local-voice-benchmark.json", choices=["local-voice-benchmark.json", "voice-20b-benchmark.json", "voice-7b-benchmark.json", "voice-20b-cpu-asr.json"])
    asyncio.run(benchmark(parser.parse_args()))
