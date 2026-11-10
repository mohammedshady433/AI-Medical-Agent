"""Explicit real inference checks. Uses saved credentials; prints no secrets.

--documents sends only synthetic public test text, never the user's library.
Running this script may consume provider quota. It is not part of default tests.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.core.secrets import SecretStore
from app.providers.client import Provider, ProviderError
from app.storage.database import Database
from app.rag.library import Library
from app.agents.assistant import Assistant
from app.services.local_runtime import LocalRuntime


async def main(args):
    real = Settings()
    secrets = SecretStore(real.data)
    report = {"mode": args.mode, "checks": [], "synthetic_documents_only": True}
    with tempfile.TemporaryDirectory(prefix="ems-live-", dir=ROOT / "build") as folder:
        settings = Settings(Path(folder))
        config = real.snapshot()
        config["mode"] = args.mode
        config["privacy"]["cloud_document_context"] = bool(args.documents)
        config["privacy"]["web_search"] = False
        settings.update(config)
        runtime = LocalRuntime(settings)
        report["model"] = config["models"][args.mode]["id"]
        started = time.perf_counter()
        try:
            if args.mode == "offline":
                await runtime.start()
            report["connection"] = await Provider(config, secrets).test()
            report["connection_seconds"] = round(time.perf_counter() - started, 3)
            database = Database(settings.data)
            library = Library(database, settings)
            assistant = Assistant(settings, secrets, database, library)
            questions = [("ar", "أهلاً، ممكن تساعدني في ترتيب أولويات يومي؟ جاوب في جملتين.")]
            if args.documents:
                for name, text in [
                    (
                        "cedar.txt",
                        "Synthetic EMS test data. Project Cedar has an approved budget of EGP 725,000. Its delivery date is 14 November 2026. The project manager is Nora.",
                    ),
                    (
                        "nile.txt",
                        "بيانات اختبار مصطنعة. مشروع النيل عدد موظفيه 18 موظف. مدير المشروع اسمه كريم. موعد التسليم يوم 20 ديسمبر 2026.",
                    ),
                ]:
                    path = Path(folder) / name
                    path.write_text(text, encoding="utf8")
                    doc_id, _ = library.register(path, name, hashlib.sha256(path.read_bytes()).hexdigest())
                    await library.ingest(doc_id)
                report["index_status"] = [
                    {k: d[k] for k in ("name", "status", "chunks")} for d in database.documents()
                ]
                questions.extend(
                    [
                        ("ar", "من المستندات، ميزانية مشروع Cedar كام بالجنيه؟ اذكر المصدر."),
                        (
                            "en",
                            "According to the documents, how many staff work on the Nile project and who manages it? Cite your source.",
                        ),
                    ]
                )
            for language, question in questions:
                settings.update({"language": language})
                chat_id = database.new_chat()
                events = [event async for event in assistant.reply(chat_id, question)]
                done = next((event for event in events if event["type"] == "done"), None)
                report["checks"].append(
                    {
                        "language": language,
                        "question": question,
                        "result": done,
                        "errors": [e for e in events if e["type"] == "error"],
                        "tools": [e["tool"] for e in events if e["type"] == "status"],
                    }
                )
                if not done:
                    raise RuntimeError("A real assistant request did not complete; see report.")
                if done["meta"]["mode"] != args.mode or done["meta"]["model"] != report["model"]:
                    raise RuntimeError("The response did not use the requested mode and model.")
                latin = len(re.findall(r"[A-Za-z]", done["content"]))
                arabic = len(re.findall(r"[\u0621-\u064a]", done["content"]))
                expected_language = latin > arabic if language == "en" else arabic > latin
                report["checks"][-1]["expected_response_language"] = expected_language
                if not expected_language:
                    raise RuntimeError("The fixture answer did not use the selected response language; see report.")
                if args.documents and len(report["checks"]) > 1:
                    content = done["content"].translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).lower()
                    source_names = {source["name"] for source in done["meta"].get("sources", [])}
                    if language == "ar":
                        correct = "725000" in re.sub(r"[,٬\s]", "", content) and "cedar.txt" in source_names
                    else:
                        correct = (
                            bool(re.search(r"\b18\b", content))
                            and any(name in content for name in ("karim", "kareem", "كريم"))
                            and "nile.txt" in source_names
                        )
                    report["checks"][-1]["expected_facts_and_source"] = correct
                    if not correct:
                        raise RuntimeError("A document answer missed the expected facts or citation; see report.")
            report["ok"] = True
        except ProviderError as exc:
            report["error"] = exc.event()
            report["ok"] = False
        except Exception as exc:
            report["error"] = type(exc).__name__ + ": " + str(exc)[:300]
            report["ok"] = False
        finally:
            await runtime.close()
            report["total_seconds"] = round(time.perf_counter() - started, 3)
            target = ROOT / "logs" / ("live-" + args.mode + ".json")
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--documents", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
