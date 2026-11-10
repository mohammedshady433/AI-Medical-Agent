"""Opt-in real CUDA correctness/recovery checks; random tiny weights, no downloads."""
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaModel
from chatterbox_graph import DecoderGraph
from speech_cancellation import SpeechCancelled, SpeechWorkerRestart


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    cfg = LlamaConfig(vocab_size=8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, attn_implementation='sdpa')
    model = LlamaModel(cfg).cuda().eval()
    ordinary = model.forward
    graph = DecoderGraph(model, capacity=16).install()
    def forward(fn, inp, past=None):
        return fn(inputs_embeds=inp, past_key_values=past, use_cache=True,
                  output_hidden_states=True, output_attentions=False, return_dict=True)
    for length, count in ((7, 20), (3, 5)):
        prefix = torch.randn(2, length, 64, device='cuda')
        base, accelerated = forward(ordinary, prefix), forward(model, prefix)
        for _ in range(count):
            token = torch.randn(2, 1, 64, device='cuda')
            base = forward(ordinary, token, base.past_key_values)
            accelerated = forward(model, token, accelerated.past_key_values)
            torch.testing.assert_close(base.last_hidden_state, accelerated.last_hidden_state, rtol=1e-5, atol=1e-5)
        assert int(accelerated.past_key_values.get_seq_length()) == length + count
    assert graph.replays > 0 and not graph.disabled
    def stop(*_):
        raise SpeechCancelled()
    hook = model.register_forward_pre_hook(stop)
    try:
        try:
            forward(model, token, accelerated.past_key_values)
            raise AssertionError('Cancellation was bypassed')
        except SpeechCancelled:
            pass
    finally:
        hook.remove()
    graph.uninstall()

    # A failed capture may poison CUDA. Signal a fresh ordinary worker, not reuse.
    broken = DecoderGraph(model, capacity=16).install()
    def fail_in_capture(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            torch.zeros(1, device='cuda').item()
        return ordinary(*args, **kwargs)
    broken.original = fail_in_capture
    prefix = torch.randn(2, 3, 64, device='cuda')
    initial = forward(ordinary, prefix)
    # Use independent caches: a fallback is allowed to update its own prefix.
    initial2 = forward(ordinary, prefix)
    try:
        forward(model, token, initial2.past_key_values)
        raise AssertionError('Invalid CUDA capture must request worker replacement')
    except SpeechWorkerRestart:
        pass
    assert broken.disabled and broken.graph is None
    report = {'ok': True, 'real_cuda': True, 'random_tiny_weights_only': True,
              'cache_overflow_continues': True, 'new_prefix_reset': True,
              'cancellation_hook_preserved': True, 'capture_failure_requests_fresh_worker': True}
    root = Path(__file__).resolve().parents[1]
    (root / 'logs/decoder-graph-correctness.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
