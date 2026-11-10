"""JSON-line speech subprocess. Stdout contains protocol only; models load once on demand."""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ASR = None
ASR_WARMED = False
TTS = None
DLL_HANDLES = []


def load_asr(cfg):
    global ASR
    if os.name == "nt" and not DLL_HANDLES:
        import site

        for directory in site.getsitepackages():
            dlls = Path(directory) / "torch/lib"
            if dlls.is_dir():
                os.environ["PATH"] = str(dlls) + os.pathsep + os.environ.get("PATH", "")
                DLL_HANDLES.append(os.add_dll_directory(str(dlls)))
    from faster_whisper import WhisperModel

    model_path = Path(cfg["runtime"]["whisper_path"])
    if not (model_path / "model.bin").is_file():
        raise ValueError("Whisper model files are missing. Complete the speech setup first.")
    if ASR is None:
        ASR = WhisperModel(
            str(model_path),
            device=cfg["asr"]["device"],
            compute_type=cfg["asr"]["compute_type"],
            cpu_threads=8,
            local_files_only=True,
        )
    else:
        ASR.model.load_model()
    return ASR


def transcribe(request):
    import numpy as np
    from faster_whisper.audio import decode_audio

    cfg = request["config"]
    source = io.BytesIO(base64.b64decode(request["audio"], validate=True)) if "audio" in request else request["path"]
    audio = decode_audio(source, sampling_rate=16000)
    if len(audio) > 16000 * cfg["asr"]["max_seconds"]:
        raise ValueError("Recording is too long. Keep each recording under two minutes.")
    if len(audio) < 4000 or float(np.sqrt(np.mean(audio**2))) < 0.002:
        return {"text": "", "language": request["language"], "warning": "No clear speech detected."}
    asr = load_asr(cfg)
    # No document vocabulary or invented context is ever supplied to ASR.
    segments, info = asr.transcribe(
        audio,
        language=request["language"] if request["language"] in ("ar", "en") else None,
        task="transcribe",
        beam_size=cfg["asr"]["beam_size"],
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=None,
        hallucination_silence_threshold=1.2,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.7)
    if cfg["asr"]["device"] == "cuda" and not cfg.get("keep_resident"):
        asr.model.unload_model(to_cpu=True)
    return {"text": text, "language": info.language}


def english_voice(text, speed):
    if sys.platform == "darwin":
        import subprocess
        with tempfile.TemporaryDirectory(prefix="ems-say-") as tmp:
            path = Path(tmp) / "speech.wav"
            result = subprocess.run(
                ["/usr/bin/say", "-v", "Samantha", "-r", str(round(180 * speed)),
                 "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(path)],
                input=text, text=True, capture_output=True, timeout=120,
            )
            if result.returncode or not path.is_file():
                raise ValueError("Install the Samantha English voice in macOS Spoken Content settings.")
            import wave
            with wave.open(str(path), "rb") as wav:
                if not wav.getnframes():
                    raise ValueError("macOS returned no speech audio. Check the Samantha voice installation and speech-service access.")
            return path.read_bytes()
    import win32com.client

    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voices = voice.GetVoices()
    selected = next((v for v in voices if "zira" in v.GetDescription().lower()), None)
    if selected is None:
        selected = next((v for v in voices if "english" in v.GetDescription().lower()), None)
    if selected is None:
        raise ValueError("Install a Windows English speech voice to read English locally.")
    voice.Voice = selected
    voice.Rate = max(-5, min(5, round((speed - 1) * 5)))
    with tempfile.TemporaryDirectory(prefix="ems-sapi-") as tmp:
        path = Path(tmp) / "speech.wav"
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(path), 3)
        voice.AudioOutputStream = stream
        voice.Speak(text)
        stream.Close()
        return path.read_bytes()


def arabic_voice(text, cfg):
    global TTS
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(4)
    from num2words import num2words

    folder = Path(cfg["runtime"]["kemetone_path"])
    if not (folder / "kemetone.pth").is_file():
        raise ValueError("KemeTone files are missing. Complete the speech setup first.")
    if cfg["runtime"]["espeak_library"]:
        os.environ["KEMETONE_ESPEAK_LIB"] = cfg["runtime"]["espeak_library"]
    if cfg["runtime"]["espeak_data"]:
        os.environ["KEMETONE_ESPEAK_DATA"] = cfg["runtime"]["espeak_data"]
    sys.path.insert(0, str(folder)) if str(folder) not in sys.path else None
    if TTS is None:
        from kokoro import KModel
        from kemetone import EgyptianG2P
        from kemetone_loader import load_verified_weights

        model = (
            KModel(
                repo_id="Rabe3/kemetone",
                config=str(folder / "config.json"),
                model=str(folder / "kemetone.pth"),
            )
            .to(cfg["tts"]["device"])
            .eval()
        )
        load_verified_weights(model, folder / "kemetone.pth")
        voice = torch.load(
            folder / "voices/kemetone.pt", map_location=cfg["tts"]["device"], weights_only=True
        )
        TTS = model, voice, EgyptianG2P()
    model, voice, g2p = TTS
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: num2words(m.group(), lang="ar"), text)
    letters = dict(
        zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                "إيه",
                "بي",
                "سي",
                "دي",
                "إي",
                "إف",
                "جي",
                "إتش",
                "آي",
                "جاي",
                "كاي",
                "إل",
                "إم",
                "إن",
                "أو",
                "بي",
                "كيو",
                "آر",
                "إس",
                "تي",
                "يو",
                "في",
                "دبليو",
                "إكس",
                "واي",
                "زد",
            ],
        )
    )
    warnings = []
    if re.search("[a-zA-Z]", text):
        warnings.append("English terms in Arabic speech are spelled out; the written answer is unchanged.")
        text = re.sub("[a-zA-Z]+", lambda m: " ".join(letters[c] for c in m.group().upper()), text)
    # Small pronunciation lexicon, not an extra LLM rewriting the answer.
    pronunciation = {
        "دلوقتي": "دِلْوَقْتِي",
        "عايزة": "عَايْزَة",
        "ممكن": "مُمْكِن",
        "معاك": "مَعَاك",
        "النهارده": "النَّهَارْدَه",
        "كده": "كِدَه",
        "دي": "دِي",
        "ده": "دَه",
        "إزاي": "إِزَّاي",
        "إيه": "إِيه",
        "أوي": "أَوِي",
        "حلو": "حِلْو",
    }
    text = " ".join(pronunciation.get(word, word) for word in text.split())
    phonemes, current = [], ""
    # Bound by phoneme count, not Unicode text length (Kokoro maximum 510).
    for word in re.split(r"\s+", text):
        piece = g2p(word)
        if len(piece) > 480:
            raise ValueError("A speech token is too long. Shorten this part of the answer.")
        if len(current) + len(piece) > 480:
            phonemes.append(current.strip())
            current = ""
        current += piece + " "
    if current.strip():
        phonemes.append(current.strip())
    audio = []
    with torch.inference_mode():
        for piece in phonemes:
            result = model(piece, voice[len(piece) - 1], speed=cfg["tts"]["speed"])
            tensor = result.audio if hasattr(result, "audio") else result
            audio.append(tensor.detach().cpu().numpy().reshape(-1))
    if not audio:
        raise ValueError("No speakable Arabic text found.")
    out = io.BytesIO()
    sf.write(out, np.concatenate(audio), 24000, format="WAV", subtype="PCM_16")
    return out.getvalue(), warnings


def handle(request):
    global ASR_WARMED
    started = time.perf_counter()
    if request["action"] == "offload":
        cfg = request["config"]
        if ASR is not None and cfg["asr"]["device"] == "cuda":
            ASR.model.unload_model(to_cpu=True)
        result = {"released": True}
    elif request["action"] == "transcribe":
        result = transcribe(request)
    elif request["action"] in ("warmup", "warmup_asr"):
        cfg = request["config"]
        if request["action"] == "warmup":
            arabic_voice("أَهْلًا", cfg)
        model = load_asr(cfg)
        if cfg["asr"]["device"] == "cuda" and cfg.get("keep_resident") and not ASR_WARMED:
            fixture = Path(__file__).resolve().parents[1] / "assets/voices/asr-warmup.wav"
            # Public synthetic fixture, never user audio or document vocabulary.
            if fixture.is_file():
                transcribe({"path": str(fixture), "language": "en", "config": cfg})
                ASR_WARMED = True
        if cfg["asr"]["device"] == "cuda" and not cfg.get("keep_resident"):
            model.model.unload_model(to_cpu=True)
        result = {"ready": True}
    elif request["action"] == "synthesize":
        if request["language"] == "en":
            audio, warnings = english_voice(request["text"], request["config"]["tts"]["speed"]), []
        else:
            audio, warnings = arabic_voice(request["text"], request["config"])
        result = {"audio": base64.b64encode(audio).decode("ascii"), "warnings": warnings}
    else:
        raise ValueError("Unknown speech action.")
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(json.loads(line))
        except Exception as exc:
            import traceback

            traceback.print_exc(file=sys.stderr)
            result = {"error": str(exc)[:500]}
        print(json.dumps(result, ensure_ascii=False), flush=True)
