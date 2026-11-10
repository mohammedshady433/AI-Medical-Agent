"""Check that real GPT-OSS asks for missing tasks instead of inventing a schedule."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.agents.assistant import Assistant
from app.core.settings import Settings, local_url
from app.storage.database import Database
from app.rag.library import Library

class NoSecrets:
    def get(self, *args):
        raise AssertionError('This test must stay offline.')

async def run(args):
    if not local_url(args.endpoint):
        raise ValueError('Use the verified local Ollama endpoint.')
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    settings = Settings(folder / 'data')
    settings.update({'mode': 'offline', 'language': 'ar', 'providers': {'ollama_local': {'endpoint': args.endpoint}}, 'speech': {'tts': {'voice': 'builtin'}}})
    db = Database(settings.data)
    assistant = Assistant(settings, NoSecrets(), db, Library(db, settings))
    results = []
    for question in ['ممكن تساعدني أرتب الشغل النهاردة؟ رد بجملة خصيرة', 'ممكن تساعدني أرتب شغلي النهاردة؟ رد بجملة قصيرة']:
        started = time.perf_counter()
        events = [e async for e in assistant.reply(db.new_chat(), question, voice=True)]
        answer = events[-1].get('content', '')
        passed = 'مهام' in answer and not any(word in answer for word in ('تقرير', 'اجتماع', 'بريد')) and ('؟' in answer or '?' in answer)
        results.append({'question': question, 'answer': answer, 'seconds': round(time.perf_counter()-started, 3), 'passed': passed, 'meta': events[-1].get('meta')})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (folder / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    if not all(r['passed'] for r in results):
        raise SystemExit('Review voice grounding before release.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args()))
