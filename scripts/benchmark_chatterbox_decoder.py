"""Rejected lab experiment: remove intermediate states during speech decoding.

The pinned Chatterbox T3 decoder consumes logits and the KV cache only. Keep
FP32, projection shape, sampling, weights and vocoder unchanged. This is not
enabled in the production worker: the measured speed gain was inconsistent.
"""
from contextlib import contextmanager


@contextmanager
def lean_decoder(enabled=True):
    if not enabled:
        yield
        return
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    import torch

    original = T3HuggingfaceBackend.forward

    @torch.inference_mode()
    def forward(self, inputs_embeds, past_key_values=None, use_cache=True,
                output_attentions=False, output_hidden_states=True, return_dict=True):
        if not return_dict or output_attentions:
            return original(self, inputs_embeds, past_key_values, use_cache,
                            output_attentions, output_hidden_states, return_dict)
        result = self.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                            use_cache=use_cache, output_attentions=False,
                            output_hidden_states=False, return_dict=True)
        return CausalLMOutputWithCrossAttentions(
            logits=self.speech_head(result.last_hidden_state),
            past_key_values=result.past_key_values,
        )
    T3HuggingfaceBackend.forward = forward
    try:
        yield
    finally:
        T3HuggingfaceBackend.forward = original
