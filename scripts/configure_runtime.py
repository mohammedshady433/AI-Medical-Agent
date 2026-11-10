import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.platforms import venv_python, ollama_name
from app.core.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--speech-root")
parser.add_argument("--ollama-root")
parser.add_argument("--tts-root")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--assistant-model", choices=["qwen2.5:7b-instruct", "gpt-oss:20b"])
parser.add_argument("--vision-model", action="store_true")
args = parser.parse_args()
settings = Settings()
if args.assistant_model:
    settings.update({"models": {"offline": {"id": args.assistant_model, "reasoning_effort": "low" if args.assistant_model == "gpt-oss:20b" else None}}, "assistant": {"voice_model": args.assistant_model}})
if args.vision_model:
    settings.update({"assistant": {"vision": {"model": "qwen2.5vl:3b"}}})
if args.tts_root:
    root = Path(args.tts_root).resolve()
    patch = {
        "runtime": {
            "tts_python": str(venv_python(root / "voice-env")),
            "chatterbox_path": str(root / "chatterbox-v3"),
        },
        "tts": {"chatterbox_version": "v3"},
    }
    if args.cuda:
        patch["tts"]["device"] = "cuda"
    settings.update({"speech": patch})
if args.speech_root:
    root = Path(args.speech_root).resolve()
    settings.update(
        {
            "speech": {
                "runtime": {
                    "python": str(venv_python(root / "speech-env")),
                    "whisper_path": str(root / "whisper"),
                    "kemetone_path": str(root / "kemetone"),
                }
            }
        }
    )
    if args.cuda:
        settings.update(
            {"speech": {"asr": {"device": "cuda", "compute_type": "float16"}, "tts": {"device": "cuda"}}}
        )
if args.ollama_root:
    root = Path(args.ollama_root).resolve()
    settings.update(
        {
            "local_runtime": {
                "executable": str(root / "ollama" / ollama_name()),
                "models_dir": str(root / "ollama-models"),
            },
            "providers": {"ollama_local": {"endpoint": "http://127.0.0.1:11435"}},
            "documents": {"embedding": {"endpoint": "http://127.0.0.1:11435"}},
        }
    )
print("Updated runtime paths in", settings.path, "without displaying credentials.")
