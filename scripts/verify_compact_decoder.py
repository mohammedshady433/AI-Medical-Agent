"""Real CUDA boundary/cache tests for the compact decoder. No model downloads."""
import copy
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaModel
from chatterbox_graph import CompactDecoderGraph
from speech_cancellation import SpeechCancelled, SpeechWorkerRestart


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    config = LlamaConfig(vocab_size=8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, attn_implementation='sdpa')
    model = LlamaModel(config).cuda().eval()
    reference = copy.deepcopy(model).half()
    graph = CompactDecoderGraph(model, capacity=16).install()
    def forward(fn, value, past=None):
        return fn(inputs_embeds=value, past_key_values=past, use_cache=True,
                  output_hidden_states=True, output_attentions=False, return_dict=True)
    for length, count in ((7, 20), (3, 5)):
        prefix = torch.randn(2, length, 64, device='cuda')
        base, accelerated = forward(reference, prefix.half()), forward(model, prefix)
        for _ in range(count):
            token = torch.randn(2, 1, 64, device='cuda')
            base = forward(reference, token.half(), base.past_key_values)
            accelerated = forward(model, token, accelerated.past_key_values)
            assert accelerated.last_hidden_state.dtype == torch.float32
            assert all(state.dtype == torch.float32 for state in accelerated.hidden_states)
            assert all(layer.keys.dtype == torch.float16 for layer in accelerated.past_key_values.layers)
            torch.testing.assert_close(base.last_hidden_state.float(), accelerated.last_hidden_state, rtol=1e-3, atol=1e-3)
        assert int(accelerated.past_key_values.get_seq_length()) == length + count
    assert graph.replays and not graph.disabled
    def stop(*_):
        raise SpeechCancelled()
    hook = model.register_forward_pre_hook(stop)
    try:
        try:
            forward(model, token, accelerated.past_key_values)
            raise AssertionError('Cancellation hook was bypassed')
        except SpeechCancelled:
            pass
    finally:
        hook.remove()
    graph.uninstall()
    model.cpu()
    assert graph.cache is None and graph.graph is None
    model.cuda()
    rebuilt = CompactDecoderGraph(model, capacity=16).install()
    prefix = forward(model, torch.randn(2, 3, 64, device='cuda'))
    forward(model, token, prefix.past_key_values)
    assert rebuilt.replays and not rebuilt.disabled
    rebuilt.uninstall()
    report = {'ok': True, 'real_cuda': True, 'random_tiny_weights_only': True,
              'fp32_output_boundary': True, 'fp16_kv_cache': True, 'cache_overflow_continues': True,
              'new_prefix_reset': True, 'cancellation_hook_preserved': True, 'offload_rewarm': True}
    root = Path(__file__).resolve().parents[1]
    (root / 'logs/compact-decoder-correctness.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
