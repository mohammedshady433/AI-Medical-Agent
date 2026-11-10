"""Generate listening samples from actual assistant replies. Uses the saved Online key only with --online.

Speech synthesis and recognition remain local. Synthetic questions do not access the user's files.
Recognition checks intelligibility, not naturalness; listening acceptance is required separately.
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
    with tempfile.TemporaryDirectory(prefix="voice-evaluation-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder))
        cfg = real.snapshot()
        cfg.update(mode="online", language="ar")
        cfg["privacy"].update(cloud_document_context=False, web_search=False)
        cfg["speech"]["tts"].update(provider=args.provider, auto_speak=True)
        if args.voice_python:
            cfg["speech"]["runtime"]["tts_python"] = str(Path(args.voice_python).resolve())
        if args.model_dir:
            cfg["speech"]["runtime"]["chatterbox_path"] = str(Path(args.model_dir).resolve())
        settings.update(cfg)
        secrets = SecretStore(real.data)
        db = Database(settings.data)
        assistant = Assistant(settings, secrets, db, Library(db, settings))
        speech = Speech(settings, secrets)
        results = []
        try:
            await speech.local("warmup")
            questions = [
                "صباح الخير. قولّي في جملتين قصيرين إنت ممكن تساعدني في إيه في الشغل؟",
                "رد في جملتين قصيرين: إزاي أنظم اجتماع مع فريق الشغل وأتابع المطلوب بعده؟",
                "اكتب جملة قصيرة تقول إن ميزانية التدريب خمسة وعشرين ألف جنيه، وإن التقرير في Microsoft Word.",
            ]
            reused = json.loads(Path(args.reuse).read_text(encoding="utf8"))["samples"] if args.reuse else []
            if reused:
                questions = [sample["question"] for sample in reused]
            for i, question in enumerate(questions, 1):
                started = time.perf_counter()
                if reused:
                    answer = reused[i - 1]["answer"]
                else:
                    events = [event async for event in assistant.reply(db.new_chat(), question, voice=True)]
                    done = next((e for e in events if e["type"] == "done"), None)
                    if not done:
                        raise RuntimeError(str(events[-1]))
                    answer = done["content"]
                tts_start = time.perf_counter()
                audio, warnings = await speech.synthesize(answer, "ar")
                tts_seconds = time.perf_counter() - tts_start
                target = ROOT / "logs" / f"{args.provider}-listening-{i}.wav"
                target.write_bytes(audio)
                transcript = await speech.transcribe(target, "ar")
                result = {
                    "question": question,
                    "answer": answer,
                    "transcript": transcript,
                    "file": str(target),
                    "tts_seconds": round(tts_seconds, 3),
                    "total_seconds": round(time.perf_counter() - started, 3),
                    "warnings": warnings,
                }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            await speech.close()
            (ROOT / "logs" / f"{args.provider}-listening.json").write_text(
                json.dumps(
                    {
                        "native_listener_acceptance": "pending",
                        "physical_microphone_test": False,
                        "samples": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf8",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["chatterbox", "kemetone"], default="chatterbox")
    parser.add_argument("--voice-python")
    parser.add_argument("--model-dir")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--online",
        action="store_true",
        help="Authorize synthetic questions to the configured Online model (may consume quota).",
    )
    source.add_argument(
        "--reuse", help="Re-synthesize saved evaluation answers locally without model/API requests."
    )
    asyncio.run(main(parser.parse_args()))
