"""Verify voice timing facts with the selected local 20B model; no user data or cloud access."""
import asyncio,json,re,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'backend'))
from app.agents.assistant import Assistant
from app.core.settings import Settings
from app.storage.database import Database
from app.rag.library import Library
from app.services.local_runtime import LocalRuntime
class NoSecrets:
 def get(self,*args):raise AssertionError('No cloud credentials in offline test.')
async def main():
 settings=Settings(args.output/'data')
 settings.update({'mode':'offline','language':'ar','providers':{'ollama_local':{'endpoint':args.endpoint}},'speech':{'tts':{'voice':'builtin'}},'assistant':{'voice_mode':'offline','voice_model':'gpt-oss:20b'}})
 db=Database(settings.data);assistant=Assistant(settings,NoSecrets(),db,Library(db,settings));runtime=LocalRuntime(settings)
 await runtime.start(); rows=[]
 try:
  cases=[
   ('work','عندي اجتماع الساعة تسعة صباحًا، وبعد الشغل أشتري حاجات للبيت، وعندي موعد مع طبيب الأسنان الساعة عشرة بالليل. لخص الأنشطة الثلاثة ومواعيدها في ثلاث جمل قصيرة.',['بعد (?:انتهاء )?(?:الشغل|العمل)','(?:9|٩|تسعة|تاسعة).*?(?:صباح|صبح)','(?:10|١٠|عشرة|عاشرة).*?(?:مساء|ليل)','أسنان|اسنان'],['الغداء']),
   ('lunch','لخص: أشتري حاجات للبيت بعد الغداء.',['بعد الغداء'],['بعد العمل|بعد الشغل']),
   ('before','لخص: أتمرن قبل الشغل ثم أشتري حاجات للبيت بعد الغداء.',['قبل (?:العمل|الشغل)','بعد الغداء'],['بعد العمل|قبل الغداء']),
   ('unclear','لخص كلامي: أشتري حاجات للبيت بعد الشهن.',['الشهن|وضح|توضيح|تقصد|غير واضح'],['الغداء|بعد العمل|بعد الشغل']),
  ]
  for repeat in range(2):
   for name,q,wanted,forbidden in cases:
    start=time.perf_counter();ev=[e async for e in assistant.reply(db.new_chat(),q,voice=True)];done=ev[-1];answer=done.get('content','');plain=re.sub(r'[\u064b-\u065f\u0670]','',answer)
    checks=[bool(re.search(p,plain)) for p in wanted]+[not bool(re.search(p,plain)) for p in forbidden]
    row={'case':name,'repeat':repeat+1,'question':q,'answer':answer,'seconds':round(time.perf_counter()-start,3),'meta':done.get('meta'),'checks':checks,'ok':all(checks)}
    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 finally: await runtime.close()
 assert all(row['ok'] for row in rows)
if __name__ == '__main__':
 import argparse
 from app.core.settings import local_url
 parser=argparse.ArgumentParser(description='Real GPT-OSS 20B before/after timing checks, using invented examples.')
 parser.add_argument('--endpoint',required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 if not local_url(args.endpoint):raise ValueError('Use the verified local Ollama endpoint.')
 args.output.mkdir(parents=True,exist_ok=False)
 asyncio.run(main())
