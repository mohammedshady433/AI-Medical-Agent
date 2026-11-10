"""Explicit live speech check. Does not access the microphone or call paid APIs."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.settings import Settings
from app.core.secrets import SecretStore
from app.speech.service import Speech


async def main(args):
    settings = Settings()
    service = Speech(settings, SecretStore(settings.data))
    result = {}
    if settings.snapshot()["speech"]["tts"]["provider"] == "azure":
        raise ValueError("This local-only smoke check will not call Azure. Select KemeTone first.")
    try:
        for language, text in [
            ("ar", "أَهْلًا، أَنَا مَعَاك. مُمْكِن أَسَاعْدَك فِي مُرَاجْعَة الْمُسْتَنَدَات."),
            ("en", "Hello. I can help you review your documents."),
        ]:
            started = time.perf_counter()
            audio, warnings = await service.synthesize(text, language)
            target = ROOT / "logs" / f"speech-{language}.wav"
            target.write_bytes(audio)
            import wave

            with wave.open(str(target)) as wav:
                duration = wav.getnframes() / wav.getframerate()
                assert duration > 0.2 and wav.getnchannels() >= 1
            result["tts_" + language] = {
                "path": str(target),
                "duration_seconds": duration,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "warnings": warnings,
            }
        if args.audio:
            started = time.perf_counter()
            result["asr"] = await service.transcribe(Path(args.audio), args.language)
            result["asr"]["latency_seconds"] = round(time.perf_counter() - started, 3)
        (ROOT / "logs/speech-smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        (ROOT / "logs/speech-smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )
        await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio")
    parser.add_argument("--language", default="ar", choices=["ar", "en"])
    args = parser.parse_args()
    asyncio.run(main(args))
