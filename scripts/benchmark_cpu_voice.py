"""Compare the full Arabic reply on the pinned Mac voice runtime, without keys.

Run with runtime/voice-env/bin/python. Does not edit settings or model files.
Synthesis is real. Playback timings are a one-ahead scheduling simulation;
use verify-full-reply.cjs for real Electron playback, including the final clip.
"""
import argparse
import hashlib
import io
import json
import platform
import random
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.speech.reply import speech_chunks
import chatterbox_worker as worker

TEXT = "قارن الراتب بعد خصم المصاريف، ثم راجع التأمين والإجازات. احسب وقت المواصلات وتكلفتها، واسأل عن فرص التعلم والترقية. اختر العرض الذي يناسب أولوياتك، وليس الراتب وحده."


def run(args):
    import numpy as np
    import torch
    cfg = json.loads((args.data_dir / "settings.json").read_text(encoding="utf8"))["speech"]
    if cfg["tts"]["voice"] != "builtin" or cfg["tts"]["device"] != "cpu":
        raise ValueError("This comparison requires the existing standard Arabic CPU voice.")
    cfg["keep_resident"] = False
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"machine": platform.machine(), "torch": torch.__version__, "text": TEXT,
              "cloud_requests": 0, "physical_microphone": False, "playback": "simulated one-ahead",
              "excludes": ["recognition", "answer generation", "microphone endpointing", "speaker startup"],
              "runs": []}
    start = time.perf_counter()
    worker.handle({"action": "warmup", "config": cfg})
    report["cold_prepare_seconds"] = round(time.perf_counter() - start, 3)
    chunks = speech_chunks(TEXT)
    for repeat in range(args.runs):
        for variant in (["original", "optimized"] if repeat % 2 == 0 else ["optimized", "original"]):
            cfg["_cpu_single_batch"] = variant == "optimized"
            start = time.perf_counter()
            row = {"variant": variant, "repeat": repeat + 1, "chunks": []}
            end_play, previous_play_start = 0.0, 0.0
            for index, chunk in enumerate(chunks):
                # Fixed seeds are benchmark-only. Production randomness is unchanged.
                seed = 500 + repeat * 10 + index
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                generated = time.perf_counter()
                audio = worker.synthesize(chunk, "ar", cfg)
                elapsed = time.perf_counter() - generated
                with wave.open(io.BytesIO(audio)) as wav:
                    duration = wav.getnframes() / wav.getframerate()
                # The UI requests i+1 when i starts playing; only one clip ahead.
                ready = previous_play_start + elapsed
                play = max(ready, end_play)
                gap = max(0, ready - end_play) if index else 0
                end_play, previous_play_start = play + duration, play
                filename = f"{repeat + 1}-{variant}-{index + 1}.wav"
                (args.output / filename).write_bytes(audio)
                row["chunks"].append({"text": chunk.strip(), "synthesis_seconds": round(elapsed, 3),
                    "audio_seconds": duration, "play_start_seconds": round(play, 3),
                    "gap_seconds": round(gap, 3), "sha256": hashlib.sha256(audio).hexdigest(), "audio": filename})
            row.update(first_audio_seconds=row["chunks"][0]["play_start_seconds"],
                finish_seconds=round(end_play, 3), total_synthesis_seconds=round(time.perf_counter() - start, 3))
            report["runs"].append(row)
            (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps({k: v for k, v in row.items() if k != "chunks"}), flush=True)
    report["all_paired_audio_identical"] = all(
        [c["sha256"] for c in report["runs"][r]["chunks"]] == [c["sha256"] for c in report["runs"][r+1]["chunks"]]
        for r in range(0, len(report["runs"]), 2))
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print("Paired audio identical:", report["all_paired_audio_identical"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, choices=[2, 4], default=2)
    run(parser.parse_args())
