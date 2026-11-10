r"""Real local-model acceptance on invented fixtures; never touches the live library.

Example: .venv/bin/python scripts/verify_knowledge.py --endpoint http://127.0.0.1:11435 --output build/knowledge-live
Windows: .venv\Scripts\python.exe scripts/verify_knowledge.py --output build/knowledge-live
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings, local_url
from app.storage.database import Database
from app.rag.library import Library
from app.agents.assistant import Assistant


class NoCloudSecrets:
    def get(self, *args):
        raise AssertionError("This acceptance test must stay offline.")


async def run(args):
    if not local_url(args.endpoint):
        raise ValueError("Use the existing local Ollama endpoint.")
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    settings = Settings(folder / "data")
    settings.update({"mode": "offline", "language": "en", "providers": {"ollama_local": {"endpoint": args.endpoint}},
        "documents": {"embedding": {"endpoint": args.endpoint}}, "models": {"offline": {"id": "gpt-oss:20b", "temperature": 0.0}}})
    db = Database(settings.data)
    library = Library(db, settings)
    for name in ("products.csv", "staff-policy.txt", "questions.jsonl"):
        content = (ROOT / "examples/knowledge" / name).read_bytes()
        path = settings.data / name
        path.write_bytes(content)
        doc_id, _ = library.register(path, name, hashlib.sha256(content).hexdigest())
        await library.ingest(doc_id)
        if db.documents()[0]["status"] != "ready":
            raise AssertionError("Live embedding setup is not ready: " + name)
        library.update(doc_id, "Use this as evidence for company policy or product questions. Keep units and cite the source.", True, True)
    db.save_knowledge("Use the library for company policies and products. Say when information is not provided. Internal demonstration code: CEDAR-42.")
    # Reopen the database before answering to exercise persistent memory.
    db = Database(settings.data)
    library = Library(db, settings)
    assistant = Assistant(settings, NoCloudSecrets(), db, library)
    cases = [
        ("product", "en", "According to our product library, what is the Cedar Desk price and warranty?", ["7250", "24"], True),
        ("arabic_policy", "ar", "وفقاً لسياسة الشركة في الملفات، ما بدل التدريب السنوي ومن يوافق على الدورة؟", ["8000"], True),
        ("missing", "en", "According to our files, how many days of parental leave do we get? If it isn't stated, say it isn't provided.", [], False),
        ("memory_after_restart", "en", "What internal demonstration code did I ask you to remember?", ["CEDAR-42"], False),
    ]
    results = []
    for name, language, question, expected, needs_sources in cases:
        settings.update({"language": language})
        start = time.perf_counter()
        events = [event async for event in assistant.reply(db.new_chat(), question)]
        done = next((e for e in reversed(events) if e["type"] == "done"), None)
        content = done["content"] if done else ""
        normalized = content.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).replace(",", "").replace("٬", "")
        passed = bool(done) and all(item in normalized for item in expected) and (not needs_sources or bool(done["meta"]["sources"]))
        if name == "missing":
            passed = passed and any(w in content.lower() for w in ("not provided", "not state", "not specified", "isn't", "doesn't", "do not", "don't"))
        if name == "memory_after_restart":
            passed = passed and "saved_owner_instructions" not in content
        record = {"case": name, "question": question, "answer": content, "seconds": round(time.perf_counter() - start, 3),
            "automated_check_passed": passed, "meta": done["meta"] if done else {}, "errors": [e for e in events if e["type"] == "error"]}
        results.append(record)
        (folder / "results.json").write_text(json.dumps({"model": "gpt-oss:20b", "scope": "Four public-fixture cases; not a general accuracy score. No speech synthesis included.", "results": results}, ensure_ascii=False, indent=2), encoding="utf8")
        print(name, "PASS" if passed else "REVIEW", record["seconds"], flush=True)
    if not all(r["automated_check_passed"] for r in results):
        raise SystemExit("Review failed acceptance cases before release.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))
