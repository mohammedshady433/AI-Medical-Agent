"""Opt-in lab worker: approved voice, fixed seed, optional BF16 or TF32 execution.

No installed runtime, weights, reference, or production worker is modified.
"""
import base64
import contextlib
import json
import sys
import time

import chatterbox_worker as approved
from benchmark_chatterbox_decoder import lean_decoder
from chatterbox_graph import DecoderGraph


def handle(request):
    import torch

    cfg = request["config"]
    cfg['_decoder_graph'] = False  # Explicit baseline; graph is controlled by this probe.
    production_profile = request.get('production_profile')
    if production_profile:
        if production_profile not in ('graph', 'compact', 'full'):
            raise ValueError('Unsupported production comparison profile')
        cfg.update(_decoder_graph=True, _compact_decoder=production_profile != 'graph',
                   _compact_waveform=production_profile == 'full')
    model = approved.load_model(cfg)
    if request["action"] == "warmup":
        return {"ready": True, "attention": model.t3.tfmr.config._attn_implementation,
                "dtype": str(next(model.t3.parameters()).dtype)}
    if request["action"] != "synthesize":
        raise ValueError("Unsupported benchmark operation")
    precision = request["precision"]
    if precision not in ("fp32", "bf16", "fp16", "tf32", "lean", "graph", "compact", "full"):
        raise ValueError("Unknown experimental precision")
    if cfg["tts"]["device"] != "cuda":
        raise ValueError("This experiment requires CUDA")
    threads = request.get("cpu_threads", 4)
    if threads not in (1, 2, 4):
        raise ValueError("Experimental CPU thread count must be 1, 2 or 4")
    torch.set_num_threads(threads)
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16")
    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    timings = {"t3_seconds": 0.0, "vocoder_seconds": 0.0}
    inference, vocoder = model.t3.inference, model.s3gen.inference

    def timed_t3(*args, **kwargs):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                                enabled=precision in ("bf16", "fp16")):
                return inference(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            timings["t3_seconds"] += time.perf_counter() - begin

    def timed_vocoder(*args, **kwargs):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            return vocoder(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            timings["vocoder_seconds"] += time.perf_counter() - begin

    model.t3.inference, model.s3gen.inference = timed_t3, timed_vocoder
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    if precision == 'graph' and not production_profile and not hasattr(model, '_probe_graph'):
        model._probe_graph = DecoderGraph(model.t3.tfmr)
    graph = getattr(model, '_probe_graph', None)
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        if graph:
            model.t3.tfmr.forward = graph.forward if precision == 'graph' else graph.original
        with lean_decoder(precision == "lean"):
            audio = approved.synthesize(request["text"], request["language"], cfg)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        model.t3.inference, model.s3gen.inference = inference, vocoder
        if graph:
            model.t3.tfmr.forward = graph.original
    active_graph = approved.DECODER_GRAPH if production_profile else graph
    if precision in ('graph', 'compact', 'full') and (active_graph.disabled or not active_graph.replays):
        raise RuntimeError('Decoder graph did not run; do not report this as accelerated')
    return {"audio": base64.b64encode(audio).decode("ascii"), "timings": timings}


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(json.loads(line))
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(error)[:500]}
        print(json.dumps(result, ensure_ascii=False), flush=True)
