"""Opt-in real Arabic worker cancellation check. Isolated config; no credentials/documents."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.speech.service import Speech


class NoCloud:
    def get(self, *args): raise AssertionError("No cloud credentials available")


async def main():
    source = Settings(Path(os.environ["LOCALAPPDATA"]) / "EMS-AI-Assistant").snapshot()
    source["mode"] = "offline"
    source["assistant"].update(voice_low_latency=True, voice_model="gpt-oss:20b", mcp_servers=[])
    report = {"ok": False, "cloud_requests": 0, "physical_microphone_test": False, "interruptions": []}
    with tempfile.TemporaryDirectory(prefix="speech-cancel-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder)); settings.update(source)
        speech = Speech(settings, NoCloud())
        try:
            prepared = await speech.local("warmup")
            report['decoder_graph'] = prepared.get('decoder_graph', False)
            report['compact_decoder'] = prepared.get('compact_decoder', False)
            report['compact_waveform'] = prepared.get('compact_waveform', False)
            print("Local voice prepared", flush=True)
            worker = speech.voice_worker
            original = worker.process.pid
            for index in range(3):
                task = asyncio.create_task(speech.synthesize("صباح الخير. ممكن أساعدك في مراجعة المستندات وترتيب المهام والاجتماعات النهارده.", "ar", local_only=True))
                await asyncio.sleep(0.7)
                started = time.perf_counter(); task.cancel()
                try: await task
                except asyncio.CancelledError: pass
                cancelled = time.perf_counter() - started
                assert worker.process is not None and worker.process.pid == original, "Cancellation reloaded the voice worker"
                report["interruptions"].append({"cancel_seconds": cancelled, "same_worker": True})
            started = time.perf_counter()
            audio, _ = await speech.synthesize("تمام، أنا معاك.", "ar", local_only=True)
            report["next_reply_seconds"] = time.perf_counter() - started
            assert audio.startswith(b"RIFF") and worker.process.pid == original
            (ROOT / "logs/voice-after-interruption.wav").write_bytes(audio)
            report["transcripts"] = {}
            for language in ("ar", "en"):
                recorded = ROOT / f"logs/voice-interruption-{language}.wav"
                heard = await speech.transcribe(recorded, language)
                assert heard["text"].strip(), "Interrupted utterance became empty"
                report["transcripts"][language] = heard["text"]
            await speech.offload()
            prepared = await speech.local('warmup')
            assert worker.process.pid == original, 'Offload unnecessarily replaced the speech worker'
            report['offload_rewarm_graph'] = prepared.get('decoder_graph', False)
            report['offload_rewarm_compact'] = prepared.get('compact_decoder', False)
            report['offload_rewarm_waveform'] = prepared.get('compact_waveform', False)
            report["ok"] = True
        finally:
            await speech.close()
            (ROOT / "logs/local-speech-cancellation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__": asyncio.run(main())
