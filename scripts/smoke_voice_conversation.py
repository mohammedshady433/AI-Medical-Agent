"""Explicit audio -> local ASR -> configured model -> local speech integration check.

This uses a supplied recording, not the physical microphone. Online mode consumes quota.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.core.secrets import SecretStore
from app.storage.database import Database
from app.rag.library import Library
from app.agents.assistant import Assistant
from app.speech.service import Speech


async def main(args):
    real = Settings()
    secrets = SecretStore(real.data)
    report = {"physical_microphone_test": False, "mode": args.mode, "language": args.language}
    with tempfile.TemporaryDirectory(prefix="voice-pipeline-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder))
        config = real.snapshot()
        config["mode"] = args.mode
        config["language"] = args.language
        config["privacy"]["cloud_document_context"] = False
        config["privacy"]["web_search"] = False
        if args.cuda:
            config["speech"]["asr"].update(device="cuda", compute_type="float16")
        settings.update(config)
        database = Database(settings.data)
        assistant = Assistant(settings, secrets, database, Library(database, settings))
        speech = Speech(settings, secrets)
        try:
            await speech.synthesize("أَهْلًا" if args.language == "ar" else "Hello.", args.language)
            started = time.perf_counter()
            transcription = await speech.transcribe(Path(args.audio).resolve(), args.language)
            report["transcription"] = transcription
            if not transcription["text"].strip():
                raise RuntimeError("No transcript returned.")
            events = [
                event
                async for event in assistant.reply(database.new_chat(), transcription["text"], voice=True)
            ]
            done = next((event for event in events if event["type"] == "done"), None)
            if not done:
                report["events"] = events
                raise RuntimeError("The voice question did not receive an answer.")
            report["answer"] = done
            tts_start = time.perf_counter()
            audio, warnings = await speech.synthesize(done["content"], args.language)
            target = ROOT / "logs" / ("voice-reply-" + args.language + ".wav")
            target.write_bytes(audio)
            report.update(
                ok=True,
                audio_file=str(target),
                audio_bytes=len(audio),
                tts_seconds=round(time.perf_counter() - tts_start, 3),
                total_seconds=round(time.perf_counter() - started, 3),
                warnings=warnings,
            )
        except Exception as exc:
            report.update(ok=False, error=str(exc)[:500])
        finally:
            await speech.close()
            (ROOT / "logs" / ("voice-conversation-" + args.language + ".json")).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", choices=["ar", "en"], default="ar")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--cuda", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
