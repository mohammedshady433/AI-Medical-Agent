"""Avoid duplicate CFG decoding for the pinned Mac standard Arabic voice.

With cfg_weight=0, Chatterbox builds two identical batches and uses only the
first. Keep one KV cache, then expose the same final hidden state twice to the
unmodified upstream sampler. Weights, precision and sampling stay unchanged.
"""
from contextlib import contextmanager
import platform
import sys


def supported(model, cfg):
    if (sys.platform != "darwin" or platform.machine() != "arm64"
            or cfg["tts"]["device"] != "cpu"
            or cfg["tts"].get("voice") != "builtin"
            or cfg["tts"].get("chatterbox_version") != "v3"
            or not cfg.get("_cpu_single_batch", True)):
        return False
    import torch
    import transformers
    return (torch.__version__.split("+")[0] == "2.6.0"
            and transformers.__version__ == "5.2.0"
            and model.config.model_type == "llama"
            and model.config._attn_implementation == "sdpa"
            and not model.training
            and next(model.parameters()).dtype == torch.float32)


@contextmanager
def single_batch(model, enabled):
    """Scope the adapter to one synthesis and restore it even on interruption."""
    if not enabled:
        yield
        return
    from speech_cancellation import SpeechWorkerRestart
    original = model.forward

    def forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if args or embeds is None or embeds.shape[0] != 2:
            raise SpeechWorkerRestart("Unexpected speech decoder shape; retry the original CPU path")
        kwargs["inputs_embeds"] = embeds[:1]
        kwargs["output_hidden_states"] = False
        result = original(**kwargs)
        # The pinned T3 backend reads only hidden_states[-1]. The sampler
        # still gets two rows, but the unused unconditional row costs no compute.
        result.hidden_states = (result.last_hidden_state.expand(2, -1, -1),)
        return result

    model.forward = forward
    try:
        yield
    finally:
        model.forward = original
