"""Local-only A/B voice experiment. Does not change the installed app or its settings."""
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import time
import wave
import psutil

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/voice"
sys.path.insert(0, str(ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf8")
from app.core.settings import Settings, local_url
from app.storage.database import Database
from app.rag.library import Library
from app.agents.assistant import Assistant, voice_config
from app.speech.service import Speech
from app.speech.reply import prepared_voice_reply, speech_chunks
from app.services.gpu import GpuCoordinator
from app.services.local_runtime import LocalRuntime
from app.providers.client import Provider
from voice_stream_probe import run_pipeline


async def run_prepared_pipeline(events, synthesize, gpu, config):
    """Exercise the production final-validation gate with the same seeded voice."""
    begin = time.perf_counter()
    report = {"chunks": [], "first_token_seconds": None}
    preview = {}

    class Adapter:
        async def synthesize(self, text, language, local_only):
            audio, details = await synthesize(text, 0)
            preview.update(text=text, audio=audio, details=details)
            return audio, []

    async for event in prepared_voice_reply(events, Adapter(), gpu, config, asyncio.Event()):
        if event["type"] == "token" and report["first_token_seconds"] is None:
            report["first_token_seconds"] = time.perf_counter() - begin
        if event["type"] == "done":
            report.update(answer=event["content"], answer_seconds=event["meta"]["total_seconds"])
            report["preview_used"] = "voice_audio" in event
            prepared = event.get("voice_audio")
        elif event["type"] in ("error", "cancelled"):
            raise RuntimeError(str(event))
    playback_end, next_prefetch = 0, 0
    for index, text in enumerate(speech_chunks(report["answer"])):
        await asyncio.sleep(max(0, next_prefetch - time.perf_counter()))
        started = time.perf_counter()
        if index == 0 and prepared:
            audio, details = preview["audio"], preview["details"]
        else:
            audio, details = await synthesize(text, index)
        ready = time.perf_counter()
        playback_start = max(playback_end, ready)
        playback_end = playback_start + details["duration_seconds"]
        next_prefetch = playback_start
        report["chunks"].append({"text": text, "queued_seconds": started - begin,
            "synthesis_seconds": ready - started, "ready_seconds": ready - begin,
            "audio": audio, **details})
    return report


class NoCloudKeys:
    def get(self, *args):
        raise AssertionError("Cloud credentials are not available to this experiment")


class SeededProvider(Provider):
    gpu_layers = 16

    def payload(self, messages, tools):
        data = super().payload(messages, tools)
        data["options"].update(temperature=0, seed=20261102)
        if data["model"] == "gpt-oss:20b":
            data["options"]["num_gpu"] = self.gpu_layers
        return data


class LabGpu(GpuCoordinator):
    """Overlap is confined to this isolated experiment; production guards are unchanged."""
    overlap = False

    @asynccontextmanager
    async def speech(self, uses_cuda):
        if self.overlap:
            yield
        else:
            async with super().speech(uses_cuda):
                yield


class VoiceWorker:
    def __init__(self, settings, cpu_threads=4):
        self.settings, self.process = settings, None
        self.cpu_threads = cpu_threads
        self.profile = None

    async def request(self, action, **payload):
        cfg = self.settings.snapshot()["speech"]
        cfg["keep_resident"] = True
        if self.process is None:
            # Do not pass service credentials to the experimental subprocess.
            env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET"))}
            env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
            with (self.settings.data / "voice-probe.log").open("ab") as log:
                self.process = await asyncio.create_subprocess_exec(
                    cfg["runtime"]["tts_python"], "-u", str(ROOT / "scripts/benchmark_voice_worker.py"),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                    env=env, limit=64 * 1024**2, creationflags=0x08000000 if os.name == "nt" else 0,
                )
        try:
            self.process.stdin.write((json.dumps({"action": action, "config": cfg, "cpu_threads": self.cpu_threads,
                                                  "production_profile": self.profile, **payload}) + "\n").encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), 180)
            if not line:
                raise RuntimeError("Experimental voice worker stopped; inspect voice-probe.log")
            result = json.loads(line)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        except BaseException:
            await self.close()
            raise

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.communicate()
        self.process = None


def wave_details(audio):
    with wave.open(io.BytesIO(audio)) as wav:
        channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
        frames, data = wav.getnframes(), wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Benchmark expects PCM16 output")
    samples = struct.unpack(f"<{len(data) // 2}h", data)
    window = max(1, rate // 100) * channels
    onset = None
    for offset in range(0, len(samples), window):
        part = samples[offset:offset + window]
        if part and (sum(x * x for x in part) / len(part)) ** 0.5 > 196:
            onset = offset / channels / rate
            break
    if onset is None:
        raise ValueError("Synthesis returned silence")
    return {"duration_seconds": frames / rate, "leading_silence_seconds": onset}


async def monitor_gpu(samples, stop):
    while not stop.is_set():
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            output, _ = await asyncio.wait_for(process.communicate(), 5)
            values = [int(x.strip()) for x in output.decode().splitlines()[0].split(",")]
            samples.append(dict(zip(("memory_mib", "utilization_percent", "temperature_c"), values)))
        except (OSError, ValueError, IndexError, TimeoutError):
            return
        try:
            await asyncio.wait_for(stop.wait(), 1)
        except TimeoutError:
            pass


def summarize(report):
    groups = {}
    for row in report["runs"]:
        key = " / ".join((row["model"], row["language"], row["variant"]))
        groups.setdefault(key, []).append(row["first_playable_seconds"])
    report["summary"] = {key: {"n": len(values), "median_seconds": round(statistics.median(values), 3),
        "min_seconds": round(min(values), 3), "max_seconds": round(max(values), 3)} for key, values in groups.items()}


async def benchmark(args):
    compact_comparison = args.compact_decoder or args.compact_voice
    SeededProvider.gpu_layers = args.gpu_layers
    source = Settings(Path(args.data_dir)).snapshot()
    private_runtime = Path(source["local_runtime"]["executable"]).resolve().parent
    for process in psutil.process_iter(["name"]):
        if process.info["name"].lower() == "ems-backend.exe":
            raise RuntimeError("Close EMS before a clean benchmark; its resident speech workers compete for VRAM.")
        if process.info["name"].lower() != "llama-server.exe":
            continue
        try:
            if Path(process.exe()).resolve().is_relative_to(private_runtime) and not psutil.pid_exists(process.ppid()):
                raise RuntimeError(f"Orphaned private model worker PID {process.pid} holds GPU resources. Stop and inspect it before benchmarking.")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not local_url(source["providers"]["ollama_local"]["endpoint"]):
        raise ValueError("Only a local Ollama endpoint is permitted")
    if source["speech"]["tts"]["provider"] != "chatterbox" or source["speech"]["tts"]["voice"] != "egyptian":
        raise ValueError("This comparison requires the approved Egyptian Chatterbox voice")
    source["mode"] = "offline"
    source["privacy"].update(cloud_document_context=False, web_search=False)
    source["assistant"].update(mcp_servers=[], voice_low_latency=True)
    precisions = ["fp32"] + (["bf16"] if args.bf16 else []) + (["fp16"] if args.fp16 else []) + (["tf32"] if args.tf32 else []) + (["lean"] if args.lean else []) + (["graph"] if args.decoder_graph else [])
    if compact_comparison:
        precisions = ['graph', 'full' if args.compact_voice else 'compact']
    report = {"schema": 1, "completed": False, "production_changed": False, "cloud_requests": 0,
        "physical_microphone_test": False, "includes_endpointing_or_playback": False,
        "measure": "Recorded utterance to first playable WAV. Leading silence and simulated playback gaps are reported separately; these are not microphone-to-speaker measurements.",
        "scope": "Conversation-only pilot: no documents, external tools, noisy room, or user barge-in acceptance.",
        "seed": 20261102, "20b_gpu_layers": args.gpu_layers, "tts_cpu_threads": args.tts_threads,
        "runs": [], "preparation": [], "tts_probes": []}
    output = ROOT / "logs" / args.output
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "results.json"

    def save():
        summarize(report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")

    with tempfile.TemporaryDirectory(prefix="stream-benchmark-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder))
        settings.update(source)
        runtime = LocalRuntime(settings)
        gpu = LabGpu(settings)
        secrets = NoCloudKeys()
        speech = Speech(settings, secrets, gpu=gpu)
        voice = VoiceWorker(settings, args.tts_threads)
        if compact_comparison:
            voice.profile = 'graph'

        async def prepare_profile(precision):
            if not compact_comparison or voice.profile == precision:
                return
            # Never compare with a second copy of the voice consuming VRAM.
            # A fresh worker reloads original weights; float() cannot undo rounding.
            await voice.close()
            voice.profile = precision
            begin = time.perf_counter()
            await voice.request('warmup')
            await voice.request('synthesize', text='أهلاً.', language='ar', precision=precision, seed=20261103)
            report['preparation'].append({'speech_profile': precision, 'seconds': time.perf_counter() - begin})
        db = Database(settings.data)
        assistant = Assistant(settings, secrets, db, Library(db, settings), gpu=gpu, provider_factory=SeededProvider)
        samples, stop = [], asyncio.Event()
        monitor = asyncio.create_task(monitor_gpu(samples, stop))
        try:
            await runtime.start()
            await gpu.release()
            report["initial_gpu_sample"] = samples[-1] if samples else None
            start = time.perf_counter()
            await speech.local("warmup_asr")
            report["voice_runtime"] = await voice.request("warmup")
            report["speech_load_seconds"] = time.perf_counter() - start
            if args.decoder_graph:
                start = time.perf_counter()
                for precision in ('fp32', 'graph'):
                    await voice.request('synthesize', text='أهلاً.', language='ar', precision=precision, seed=20261103)
                report['voice_prime_seconds'] = time.perf_counter() - start
            if compact_comparison:
                await voice.request('synthesize', text='أهلاً.', language='ar', precision='graph', seed=20261103)

            # Fixed-text probes isolate voice generation speed from changing LLM answers.
            phrases = ["ابدأ بأهم مهمة عندك، وبعدها خد استراحة قصيرة.",
                       "ما توقفش المضخة قبل ما تراجع تعليمات التشغيل.",
                       "الضغط اتنين ونص بار، ودرجة الحرارة خمسة وثلاثين درجة."]
            if args.tts_probe:
                for repeat in range(args.runs):
                    for number, phrase in enumerate(phrases):
                        for precision in (precisions if repeat % 2 == 0 else list(reversed(precisions))):
                            await prepare_profile(precision)
                            start = time.perf_counter()
                            result = await voice.request("synthesize", text=phrase, language="ar", precision=precision, seed=20261102 + number)
                            elapsed = time.perf_counter() - start
                            audio = base64.b64decode(result["audio"])
                            name = f"tts-{number}-{repeat}-{precision}.wav"
                            (output / name).write_bytes(audio)
                            heard = await speech.transcribe(output / name, "ar")
                            row = {"text": phrase, "precision": precision, "repeat": repeat + 1,
                                "seconds": elapsed, "audio": name, "round_trip_transcript": heard["text"],
                                **wave_details(audio), **result["timings"]}
                            report["tts_probes"].append(row)
                            save()
                            print(json.dumps({"tts_probe": row}, ensure_ascii=False), flush=True)

            for model in ([] if args.tts_only else args.models):
                await gpu.release()
                settings.update({"assistant": {"voice_model": model}})
                start = time.perf_counter()
                async with gpu.assistant(voice_config(settings.snapshot())):
                    await SeededProvider(voice_config(settings.snapshot()), secrets).preload()
                report["preparation"].append({"model": model, "model_load_seconds": time.perf_counter() - start})
                # First ASR invocation includes lazy CUDA work; keep it outside warm comparisons.
                for language in args.languages:
                    await speech.transcribe(FIXTURES / f"voice-question-{language}.wav", language)
                for language in args.languages:
                    # Hold language/prompt constant within each alternating comparison.
                    # Prime both variants equally: changing AR/EN between repeats otherwise
                    # rewards whichever variant follows the other in Ollama's prompt cache.
                    settings.update({"language": language})
                    warm_question = await speech.transcribe(FIXTURES / f"voice-question-{language}.wav", language)
                    async for _ in assistant.reply(db.new_chat(), warm_question["text"], voice=True):
                        pass
                    for repeat in range(args.runs):
                        settings.update({"language": language})
                        fixture = FIXTURES / f"voice-question-{language}.wav"
                        variants = (["sequential-fp32-4threads", "sequential-fp32-1threads"]
                                    if args.compare_threads else ["sequential-fp32", "streaming-fp32"])
                        if args.reply_speed:
                            variants = ["sequential-fp32", "prepared-fp32"]
                        if args.decoder_graph:
                            variants = ['prepared-fp32', 'prepared-graph']
                        if compact_comparison:
                            variants = ['prepared-' + precision for precision in precisions]
                        if language == "ar" and args.bf16:
                            variants.append("streaming-bf16")
                        if language == "ar" and args.fp16:
                            variants.append("streaming-fp16")
                        if language == "ar" and args.tf32:
                            variants.append("streaming-tf32")
                        if repeat % 2:
                            variants.reverse()
                        for variant in variants:
                            await prepare_profile(variant.split('-')[1])
                            voice.cpu_threads = (1 if variant.endswith("1threads") else 4) if args.compare_threads else args.tts_threads
                            gpu.overlap = variant.startswith("streaming")
                            sample_start = len(samples)
                            begin = time.perf_counter()
                            transcript = await speech.transcribe(fixture, language)
                            asr = time.perf_counter() - begin

                            async def synthesize(text, index):
                                speech.audio_cache.clear()
                                speech.cache_bytes = 0
                                if language == "ar":
                                    async with gpu.speech(True):
                                        result = await voice.request("synthesize", text=text, language=language,
                                            precision=variant.split("-")[1], seed=20261102 + index)
                                    audio = base64.b64decode(result["audio"])
                                    details = result["timings"]
                                else:
                                    audio, warnings = await speech.synthesize(text, language, local_only=True)
                                    details = {"warnings": warnings}
                                return audio, {**details, **wave_details(audio)}

                            events = assistant.reply(db.new_chat(), transcript["text"], voice=True)
                            if variant.startswith('prepared-'):
                                result = await run_prepared_pipeline(events, synthesize, gpu, settings.snapshot())
                            else:
                                result = await run_pipeline(events, synthesize, streaming=gpu.overlap)
                            if not result["chunks"]:
                                raise ValueError("No speech chunks generated")
                            name = f"{model.replace(':', '-')}-{language}-{repeat + 1}-{variant}"
                            playback_end, gaps = 0, []
                            for index, chunk in enumerate(result["chunks"]):
                                wav = chunk.pop("audio")
                                chunk["audio_file"] = f"{name}-{index}.wav"
                                (output / chunk["audio_file"]).write_bytes(wav)
                                ready = asr + chunk["ready_seconds"]
                                if index:
                                    gaps.append(max(0, ready - playback_end))
                                playback_end = max(playback_end, ready) + chunk["duration_seconds"]
                            row = {"model": model, "language": language, "repeat": repeat + 1, "variant": variant,
                                "asr_seconds": asr, "heard": transcript["text"],
                                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                "first_playable_seconds": asr + result["chunks"][0]["ready_seconds"],
                                "first_signal_estimate_seconds": asr + result["chunks"][0]["ready_seconds"] + result["chunks"][0]["leading_silence_seconds"],
                                "peak_gpu_memory_mib": max((x["memory_mib"] for x in samples[sample_start:]), default=None),
                                "simulated_max_playback_gap_seconds": max(gaps, default=0), **result}
                            report["runs"].append(row)
                            save()
                            print(json.dumps({"model": model, "language": language, "variant": variant, "repeat": repeat + 1,
                                "first_playable_seconds": round(row["first_playable_seconds"], 3),
                                "gap_seconds": round(row["simulated_max_playback_gap_seconds"], 3),
                                "answer": result["answer"]}, ensure_ascii=False), flush=True)
            report["completed"] = True
        finally:
            stop.set()
            await monitor
            report["gpu"] = {"samples": len(samples),
                "peak_memory_mib": max((x["memory_mib"] for x in samples), default=None),
                "peak_temperature_c": max((x["temperature_c"] for x in samples), default=None)}
            await voice.close()
            await speech.close()
            gpu.overlap = False
            await gpu.release()
            await runtime.close()
            log = settings.data / "voice-probe.log"
            if log.exists():
                (output / log.name).write_bytes(log.read_bytes())
            save()
    print(json.dumps({"report": str(report_path), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=["gpt-oss:20b", "qwen2.5:7b-instruct"], default=["qwen2.5:7b-instruct", "gpt-oss:20b"])
    parser.add_argument("--languages", nargs="+", choices=["ar", "en"], default=["ar", "en"])
    parser.add_argument("--runs", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--tts-probe", action="store_true")
    parser.add_argument("--compare-threads", action="store_true", help="Interleave 4/1 CPU thread sequential pipelines; same FP32 voice and seed")
    parser.add_argument("--reply-speed", action="store_true", help="Compare 1.2.2 sequential voice with finalized first-clip preparation")
    parser.add_argument('--decoder-graph', action='store_true', help='Compare the 1.2.3 prepared reply with FP32 CUDA graph decoding, same voice and final-answer gate')
    parser.add_argument('--compact-decoder', action='store_true', help='Compare the 1.2.4 FP32 graph with a compact transformer and FP32 speech projection/sampler, in separate warmed workers')
    parser.add_argument('--compact-voice', action='store_true', help='Compare 1.2.4 with compact decoder and waveform estimator; same weights, reference, sampling and playback speed')
    parser.add_argument("--tts-only", action="store_true", help="Skip LLM runs; use with --tts-probe for fixed-phrase timing")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lean", action="store_true", help="Compare FP32 decoder without unused intermediate hidden states")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gpu-layers", type=int, choices=[8, 12, 16], default=16,
        help="20B-only experimental GPU allocation. Never saved to app settings.")
    parser.add_argument("--tts-threads", type=int, choices=[1, 2, 4], default=4,
        help="Experimental TTS CPU thread count; weights and app settings are unchanged")
    parser.add_argument("--output", default="streaming-voice-pilot", type=lambda x: x if x and all(c.isalnum() or c in "-_" for c in x) else parser.error("Output must be a folder name"))
    args = parser.parse_args()
    if args.tts_only and not args.tts_probe:
        parser.error("--tts-only requires --tts-probe")
    if args.decoder_graph and args.languages != ['ar']:
        parser.error('--decoder-graph requires --languages ar; English uses the Windows voice')
    if (args.compact_decoder or args.compact_voice) and (args.languages != ['ar'] or any((args.decoder_graph, args.reply_speed, args.compare_threads, args.bf16, args.fp16, args.tf32, args.lean)) or (args.compact_decoder and args.compact_voice)):
        parser.error('Compact comparisons require --languages ar and cannot be mixed with other comparison modes')
    asyncio.run(benchmark(args))
