"""Real local-model regression: preserve spoken morning/evening appointment times."""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.agents.assistant import Assistant
from app.core.settings import Settings, local_url
from app.storage.database import Database
from app.rag.library import Library

class NoSecrets:
    def get(self, *args):
        raise AssertionError("This check must stay offline.")

async def run(args):
    if not local_url(args.endpoint):
        raise ValueError("Use the verified local Ollama endpoint.")
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    settings = Settings(folder / "data")
    settings.update({"mode": "offline", "language": "ar", "providers": {"ollama_local": {"endpoint": args.endpoint}}, "speech": {"tts": {"voice": "builtin"}}})
    db = Database(settings.data)
    assistant = Assistant(settings, NoSecrets(), db, Library(db, settings))
    results = []
    # Invented appointment examples; never reads the user's conversation database.
    cases = [
        ("عندي موعد الساعة عشرة بالليل. أكد الموعد في جملة قصيرة.", "عاشرة|عشرة|10|١٠", "مساء|ليل", "صباح"),
        ("عندي موعد الساعة سبعة بالليل. أكد الموعد في جملة قصيرة.", "سابعة|سبعة|7|٧", "مساء|ليل", "صباح"),
        ("عندي اجتماع الساعة تسعة الصبح. أكد الموعد في جملة قصيرة.", "تاسعة|تسعة|9|٩", "صباح|صبح", "مساء"),
    ]
    for question, hour, period, wrong_period in cases:
        started = time.perf_counter()
        events = [e async for e in assistant.reply(db.new_chat(), question, voice=True)]
        answer = events[-1].get("content", "")
        plain = re.sub(r"[\u064b-\u065f\u0670]", "", answer)
        passed = bool(re.search(hour, plain) and re.search(period, plain) and not re.search(wrong_period, plain))
        results.append({"question": question, "answer": answer, "seconds": round(time.perf_counter() - started, 3), "passed": passed, "model": events[-1].get("meta", {}).get("model")})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    for malformed in ("الساعة ساعة الصبح", "الساعة الساعة الصبح"):
        question = f"عندي اجتماع {malformed} وبعد الشغل أشتري حاجات للبيت. وعندي موعد الساعة عشرة بالليل. لخص المواعيد في جملة قصيرة."
        started = time.perf_counter()
        events = [e async for e in assistant.reply(db.new_chat(), question, voice=True)]
        answer = events[-1].get("content", "")
        plain = re.sub(r"[\u064b-\u065f\u0670]", "", answer)
        passed = bool(re.search("؟|\\?", plain) and re.search("اجتماع|موعد", plain) and not re.search("واحدة|(?:^|\\D)[1١](?:\\D|$)|مبكر", plain))
        results.append({"question": question, "answer": answer, "seconds": round(time.perf_counter() - started, 3), "passed": passed, "model": events[-1].get("meta", {}).get("model")})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    if not all(r["passed"] for r in results):
        raise SystemExit("Review spoken-time preservation before release.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))
