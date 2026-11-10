"""Persistent, local-only Arabic Chatterbox worker. Uses no cloud endpoints or automatic downloads."""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

from speech_text import LATIN, verbalize_numbers
from speech_cancellation import cancellable_inference, check_cancelled, serve
from chatterbox_cpu import single_batch, supported as cpu_supported

MODEL = None
WARMED = False
DECODER_GRAPH = None
WAVEFORM_COMPACT = False


def release_decoder_graph():
    global DECODER_GRAPH, WARMED
    if DECODER_GRAPH is not None:
        DECODER_GRAPH.uninstall()
        DECODER_GRAPH = None
    WARMED = False


def configure_decoder_graph(cfg):
    global DECODER_GRAPH, WAVEFORM_COMPACT
    import torch
    from chatterbox_graph import CompactDecoderGraph, DecoderGraph, compact_supported, supported
    from speech_cancellation import SpeechWorkerRestart
    compact = compact_supported(MODEL.t3.tfmr, cfg)
    # Casting rounded weights back to float32 would not restore the checkpoint.
    # A changed/unsupported profile must reload the original weights instead.
    if next(MODEL.t3.tfmr.parameters()).dtype == torch.float16 and not compact:
        release_decoder_graph()
        raise SpeechWorkerRestart('Reload the original FP32 voice for this runtime profile')
    if supported(MODEL.t3.tfmr, cfg):
        if DECODER_GRAPH is None:
            cls = CompactDecoderGraph if compact else DecoderGraph
            DECODER_GRAPH = cls(MODEL.t3.tfmr).install()
    elif DECODER_GRAPH is not None:
        release_decoder_graph()
    WAVEFORM_COMPACT = configure_waveform_precision(MODEL.s3gen.flow.decoder.estimator, cfg, compact)


def configure_waveform_precision(estimator, cfg, compact):
    import torch
    from speech_cancellation import SpeechWorkerRestart
    enabled = compact and cfg.get('_compact_waveform', True)
    dtype = next(estimator.parameters()).dtype
    if dtype == torch.float16 and not enabled:
        raise SpeechWorkerRestart('Reload the original FP32 waveform weights')
    if enabled and dtype != torch.float16:
        try:
            estimator.half()
        except Exception as error:
            raise SpeechWorkerRestart('Reload the source waveform weights in FP32') from error
    return enabled


def reference_for_voice(cfg):
    profile = cfg["tts"].get("voice", "builtin")
    if profile == "builtin":
        return None
    if profile != "egyptian":
        raise ValueError("Choose an installed Arabic voice in Voice Settings.")
    reference = Path(__file__).resolve().parents[1] / "assets/voices/egyptian-reference.wav"
    if not reference.is_file():
        raise ValueError("The Egyptian voice reference is missing. Reinstall the EMS application package.")
    return reference


def load_model(cfg):
    global MODEL
    import torch

    device = cfg["tts"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for the Arabic voice. Select CPU in Voice Settings.")
    if MODEL is not None:
        if device != 'cuda' and DECODER_GRAPH is not None:
            release_decoder_graph()
        for component in (MODEL.t3, MODEL.s3gen, MODEL.ve):
            component.to(device)
        MODEL.conds.to(device)
        configure_decoder_graph(cfg)
        return MODEL
    folder = Path(cfg["runtime"]["chatterbox_path"])
    version = cfg["tts"].get("chatterbox_version", "v3")
    if version not in ("v2", "v3"):
        raise ValueError("Unsupported Chatterbox version. Choose v2 or v3 in Voice Settings.")
    filename = f"t3_mtl23ls_{version}.safetensors"
    for name in (filename, "s3gen.pt", "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json"):
        if not (folder / name).is_file():
            raise ValueError(
                "Some Arabic voice files are missing. Open Settings, choose Voice conversation, and click Set up to resume the download."
            )
    threads = min(8, os.cpu_count() or 4) if sys.platform == "darwin" and device == "cpu" else 4
    torch.set_num_threads(threads)
    from chatterbox.mtl_tts import (
        ChatterboxMultilingualTTS,
        Conditionals,
        T3,
        T3Config,
        VoiceEncoder,
        S3Gen,
        MTLTokenizer,
    )
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    # Load the explicitly selected checkpoint with strict tensor checks and a
    # local Arabic/English tokenizer. Never rename a V3 file to V2.
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(str(folder / filename)), strict=True)
    ve, gen = VoiceEncoder(), S3Gen()
    ve.load_state_dict(torch.load(folder / "ve.pt", map_location="cpu", weights_only=True), strict=True)
    gen.load_state_dict(torch.load(folder / "s3gen.pt", map_location="cpu", weights_only=True), strict=True)
    for component in (t3, ve, gen):
        component.to(device).eval()

    class ArabicEnglishTokenizer(MTLTokenizer):
        def __init__(self, path):
            # Arabic/English need no Chinese segmenter or remote Cangjie cache.
            self.tokenizer = Tokenizer.from_file(str(path))
            self.cangjie_converter = None
            self.check_vocabset_sot_eot()

    tokenizer = ArabicEnglishTokenizer(folder / "grapheme_mtl_merged_expanded_v1.json")
    conds = Conditionals.load(folder / "conds.pt", map_location="cpu").to(device)
    model = ChatterboxMultilingualTTS(t3, gen, ve, tokenizer, device, conds)
    reference = reference_for_voice(cfg)
    if reference:
        # Build the speaker conditioning once; every reply reuses it locally.
        model.prepare_conditionals(str(reference), exaggeration=0.4)
    MODEL = model
    configure_decoder_graph(cfg)
    return MODEL


def text_for_voice(text, language):
    if language != "ar":
        return text
    from num2words import num2words

    # Natural input: no G2P, no diacritizer, no Arabic spelling of English names.
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    parts, offset = [], 0
    for match in LATIN.finditer(text):
        parts.append(verbalize_numbers(text[offset : match.start()], lambda n: num2words(n, lang="ar")))
        parts.append(match.group())
        offset = match.end()
    parts.append(verbalize_numbers(text[offset:], lambda n: num2words(n, lang="ar")))
    return "".join(parts)


def synthesize(text, language, cfg, cancelled=None):
    import numpy as np
    import soundfile as sf
    import torch

    model = load_model(cfg)
    text = text_for_voice(text, language)
    if not text.strip() or language not in ("ar", "en"):
        raise ValueError("Enter Arabic or English text for local speech.")
    segments = re.findall(r"[^.!?؟\n]+[.!?؟\n]*", text)
    chunks = []
    for segment in segments:
        words, current = segment.split(), ""
        for word in words:
            if len(current) + len(word) > 220 and current:
                chunks.append(current)
                current = ""
            current = (current + " " + word).strip()
        if current:
            chunks.append(current)
    audio = []
    accelerate_cpu = cpu_supported(model.t3.tfmr, cfg)
    for chunk in chunks:
        try:
            with cancellable_inference(model, cancelled), single_batch(model.t3.tfmr, accelerate_cpu):
                wav = model.generate(
                    chunk,
                    language_id=language,
                    cfg_weight=0.3 if cfg["tts"].get("voice") == "egyptian" else 0.0,
                    exaggeration=0.4,
                    temperature=0.65,
                    repetition_penalty=1.2,
                )
            if not torch.isfinite(wav).all():
                raise ValueError('Local speech produced non-finite audio')
        except Exception as error:
            from speech_cancellation import SpeechCancelled, SpeechWorkerRestart
            if isinstance(error, (SpeechCancelled, SpeechWorkerRestart)):
                raise
            if accelerate_cpu or WAVEFORM_COMPACT or (DECODER_GRAPH is not None and DECODER_GRAPH.dtype is not None):
                raise SpeechWorkerRestart('Retry local speech with the original FP32 checkpoint') from error
            raise
        audio.append(wav.squeeze().detach().cpu().numpy())
    waveform = np.concatenate(audio)
    speed = cfg["tts"]["speed"]
    if speed != 1:
        import librosa

        waveform = librosa.effects.time_stretch(waveform, rate=speed)
    output = io.BytesIO()
    sf.write(output, waveform, model.sr, format="WAV", subtype="PCM_16")
    return output.getvalue()


def handle(request, cancelled=None):
    global WARMED
    started = time.perf_counter()
    cfg = request["config"]
    if request["action"] == "offload":
        result = {"released": True}
    elif request["action"] == "warmup":
        load_model(cfg)
        if (cfg["tts"]["device"] == "cpu" or cfg.get("keep_resident")) and not WARMED:
            synthesize("أهلاً.", "ar", cfg, cancelled)
            WARMED = True  # Silent warmup only; no audio is returned or played.
        result = {"ready": True, "voice_model": "Chatterbox Multilingual " + cfg["tts"]["chatterbox_version"]}
    elif request["action"] == "synthesize":
        audio = synthesize(request["text"], request["language"], cfg, cancelled)
        check_cancelled(cancelled)
        result = {"audio": base64.b64encode(audio).decode("ascii"), "warnings": []}
    else:
        raise ValueError("The voice worker only supports synthesis and warmup.")
    if MODEL is not None and cfg["tts"]["device"] == "cuda" and (request["action"] == "offload" or not cfg.get("keep_resident")):
        import torch

        release_decoder_graph()  # Captured GPU addresses must not survive a move to CPU.
        for component in (MODEL.t3, MODEL.s3gen, MODEL.ve):
            component.to("cpu")
        MODEL.conds.to("cpu")
        torch.cuda.empty_cache()
    result["seconds"] = round(time.perf_counter() - started, 3)
    result["decoder_graph"] = bool(DECODER_GRAPH is not None and DECODER_GRAPH.graph is not None and not DECODER_GRAPH.disabled)
    result["compact_decoder"] = bool(DECODER_GRAPH is not None and DECODER_GRAPH.dtype is not None)
    result["compact_waveform"] = WAVEFORM_COMPACT and request['action'] != 'offload'
    return result


if __name__ == "__main__":
    serve(handle)
