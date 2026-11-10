import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import AsyncMock
import asyncio
import json


SCRIPTS = Path(__file__).parents[2] / 'scripts'


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import chatterbox_graph
    return chatterbox_graph


def fake_runtime(monkeypatch):
    dtype = object()
    torch = SimpleNamespace(__version__='2.6.0+cu124', float32=dtype)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='5.2.0'))
    monkeypatch.setitem(sys.modules, 'transformers.modeling_outputs', SimpleNamespace(BaseModelOutputWithPast=lambda **kw: kw))
    return dtype


def test_only_measured_resident_runtime_uses_acceleration(monkeypatch):
    m = module(monkeypatch)
    dtype = fake_runtime(monkeypatch)
    model = SimpleNamespace(config=SimpleNamespace(model_type='llama', _attn_implementation='sdpa'), training=False,
                            parameters=lambda: iter([SimpleNamespace(dtype=dtype)]))
    cfg = {'keep_resident': True, 'tts': {'device': 'cuda'}}
    assert m.supported(model, cfg)
    assert not m.supported(model, {**cfg, 'keep_resident': False})
    assert not m.supported(model, {**cfg, 'tts': {'device': 'cpu'}})
    assert not m.supported(model, {**cfg, '_decoder_graph': False})
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='6.0.0'))
    assert not m.supported(model, cfg)


def test_compact_decoder_requires_supported_gpu_and_can_be_disabled(monkeypatch):
    m = module(monkeypatch)
    dtype = fake_runtime(monkeypatch)
    torch = sys.modules['torch']
    torch.float16 = object()
    torch.cuda = SimpleNamespace(get_device_capability=lambda _: (8, 6))
    model = SimpleNamespace(config=SimpleNamespace(model_type='llama', _attn_implementation='sdpa'), training=False,
                            parameters=lambda: iter([SimpleNamespace(dtype=dtype, device='cuda')]))
    cfg = {'keep_resident': True, 'tts': {'device': 'cuda'}}
    assert m.compact_supported(model, cfg)
    assert not m.compact_supported(model, {**cfg, '_compact_decoder': False})
    assert not m.compact_supported(model, {**cfg, '_decoder_graph': False})
    assert not m.compact_supported(model, {**cfg, 'tts': {'device': 'cuda', 'chatterbox_version': 'v2'}})
    torch.cuda.get_device_capability = lambda _: (7, 5)
    assert m.supported(model, cfg) and not m.compact_supported(model, cfg)


def test_compact_decoder_keeps_projection_fp32_and_cache_compact(monkeypatch):
    m = module(monkeypatch)
    fake_runtime(monkeypatch)
    half_dtype = sys.modules['torch'].float16 = object()
    model = SimpleNamespace(forward=Mock(), half=Mock())
    graph = m.CompactDecoderGraph(model)
    inputs = SimpleNamespace(to=Mock(return_value='half-inputs'))
    state = SimpleNamespace(float=Mock(return_value='float-output'))
    cache = object()
    result = SimpleNamespace(last_hidden_state=state, hidden_states=(state,), past_key_values=cache)
    called = {}
    def forward(self, *args, **kwargs):
        called.update(kwargs)
        return result
    monkeypatch.setattr(m.DecoderGraph, 'forward', forward)
    assert graph.forward(inputs_embeds=inputs, use_cache=True) is result
    inputs.to.assert_called_once_with(dtype=half_dtype)
    assert called['inputs_embeds'] == 'half-inputs'
    assert result.last_hidden_state == 'float-output' and result.hidden_states == ('float-output',)
    assert result.past_key_values is cache
    model.half.assert_called_once()


def test_compact_failure_reloads_source_weights_but_cancellation_propagates(monkeypatch):
    import pytest
    m = module(monkeypatch)
    fake_runtime(monkeypatch)
    sys.modules['torch'].float16 = object()
    graph = m.CompactDecoderGraph(SimpleNamespace(forward=Mock(), half=Mock()))
    for error in (m.SpeechCancelled(), RuntimeError('GPU failure')):
        def fail(*_, **__):
            raise error
        monkeypatch.setattr(m.DecoderGraph, 'forward', fail)
        expected = m.SpeechCancelled if isinstance(error, m.SpeechCancelled) else m.SpeechWorkerRestart
        with pytest.raises(expected):
            graph.forward()


def test_changed_compact_profile_requests_original_weights_in_fresh_worker(monkeypatch):
    import pytest
    m = module(monkeypatch)
    fake_runtime(monkeypatch)
    half_dtype = sys.modules['torch'].float16 = object()
    spec = importlib.util.spec_from_file_location('compact_worker_test', SCRIPTS / 'chatterbox_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    tfmr = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype=half_dtype)]))
    worker.MODEL = SimpleNamespace(t3=SimpleNamespace(tfmr=tfmr))
    worker.release_decoder_graph = Mock()
    with pytest.raises(m.SpeechWorkerRestart):
        worker.configure_decoder_graph({'keep_resident': True, 'tts': {'device': 'cpu'}})
    worker.release_decoder_graph.assert_called_once()


def test_waveform_precision_is_guarded_and_cannot_restore_rounded_weights(monkeypatch):
    import pytest
    m = module(monkeypatch)
    dtype = fake_runtime(monkeypatch)
    half_dtype = sys.modules['torch'].float16 = object()
    spec = importlib.util.spec_from_file_location('waveform_worker_test', SCRIPTS / 'chatterbox_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    param = SimpleNamespace(dtype=dtype)
    estimator = SimpleNamespace(parameters=lambda: iter([param]), half=Mock())
    assert not worker.configure_waveform_precision(estimator, {}, False)
    assert not worker.configure_waveform_precision(estimator, {'_compact_waveform': False}, True)
    estimator.half.assert_not_called()
    assert worker.configure_waveform_precision(estimator, {}, True)
    estimator.half.assert_called_once()
    param.dtype = half_dtype
    with pytest.raises(m.SpeechWorkerRestart):
        worker.configure_waveform_precision(estimator, {}, False)
    param.dtype = dtype
    estimator.half.side_effect = RuntimeError('allocation failed')
    with pytest.raises(m.SpeechWorkerRestart):
        worker.configure_waveform_precision(estimator, {}, True)


def test_nonfinite_optimized_audio_requests_fresh_fp32_worker(monkeypatch):
    import contextlib
    import pytest
    m = module(monkeypatch)
    spec = importlib.util.spec_from_file_location('audio_fallback_test', SCRIPTS / 'chatterbox_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setitem(sys.modules, 'soundfile', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(isfinite=lambda _: SimpleNamespace(all=lambda: False)))
    model = SimpleNamespace(generate=Mock(return_value='invalid-waveform'), t3=SimpleNamespace(tfmr=Mock()))
    worker.load_model = lambda _: model
    worker.cpu_supported = lambda *_: False
    worker.text_for_voice = lambda text, _: text
    worker.cancellable_inference = lambda *_: contextlib.nullcontext()
    worker.WAVEFORM_COMPACT = True
    cfg = {'tts': {'voice': 'egyptian', 'speed': 1}}
    with pytest.raises(m.SpeechWorkerRestart):
        worker.synthesize('كلام.', 'ar', cfg)
    model.generate.side_effect = m.SpeechCancelled()
    with pytest.raises(m.SpeechCancelled):
        worker.synthesize('كلام.', 'ar', cfg)


def test_capture_failure_falls_back_and_does_not_retry_every_token(monkeypatch):
    m = module(monkeypatch)
    dtype = fake_runtime(monkeypatch)
    model = SimpleNamespace(forward=Mock(return_value='ordinary'))
    graph = m.DecoderGraph(model).install()
    graph.initialize = Mock(side_effect=MemoryError('optional allocation unavailable'))
    embeds = SimpleNamespace(device=SimpleNamespace(type='cuda'), dtype=dtype, shape=(2, 1, 64))
    past = SimpleNamespace(get_seq_length=lambda: 3)
    for _ in range(2):
        assert model.forward(inputs_embeds=embeds, past_key_values=past) == 'ordinary'
    graph.initialize.assert_called_once()
    assert graph.disabled and graph.cache is None and graph.graph is None
    assert graph.original.call_count == 2


def test_full_cache_converts_prefix_without_truncating_speech(monkeypatch):
    m = module(monkeypatch)
    dtype = fake_runtime(monkeypatch)
    model = SimpleNamespace(forward=Mock(return_value='continued'))
    graph = m.DecoderGraph(model, capacity=8).install()
    graph.cursor, graph.cache = 8, object()
    dynamic = object()
    graph.dynamic = Mock(return_value=dynamic)
    embeds = SimpleNamespace(device=SimpleNamespace(type='cuda'), dtype=dtype, shape=(2, 1, 64))
    assert model.forward(inputs_embeds=embeds, past_key_values=graph.cache) == 'continued'
    assert graph.original.call_args.kwargs['past_key_values'] is dynamic
    graph.dynamic.assert_called_once()


def test_offload_releases_captured_addresses_and_rearms_warmup(monkeypatch):
    module(monkeypatch)
    spec = importlib.util.spec_from_file_location('graph_worker_test', SCRIPTS / 'chatterbox_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    calls = []
    worker.DECODER_GRAPH = SimpleNamespace(uninstall=lambda: calls.append('release'))
    component = SimpleNamespace(to=lambda device: calls.append(device))
    worker.MODEL = SimpleNamespace(t3=component, s3gen=component, ve=component, conds=component)
    worker.WARMED = True
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append('empty'))))
    result = worker.handle({'action': 'offload', 'config': {'tts': {'device': 'cuda'}, 'keep_resident': True}})
    assert calls == ['release', 'cpu', 'cpu', 'cpu', 'cpu', 'empty']
    assert worker.DECODER_GRAPH is None and not worker.WARMED and not result['decoder_graph']


async def test_failed_capture_restarts_once_without_changing_saved_voice(settings, secrets, monkeypatch):
    from app.speech.service import Speech
    cfg = settings.snapshot()['speech']
    cfg['runtime']['tts_python'] = sys.executable
    saved = settings.snapshot()['speech']
    requests, processes = [], []

    async def spawn(*args, **kwargs):
        reader = asyncio.StreamReader()
        number = len(processes)
        def write(data):
            requests.append(json.loads(data))
            reply = {'retry_without_decoder_graph': True} if number == 0 else {'audio': 'UklGRg=='}
            reader.feed_data((json.dumps(reply) + '\n').encode())
        process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(write=write, drain=AsyncMock()), stdout=reader,
                                  kill=Mock(), communicate=AsyncMock(return_value=(b'', b'')))
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    speech = Speech(settings, secrets, 'chatterbox')
    assert await speech._request(cfg, 'synthesize', text='نفس الصوت.', language='ar') == {'audio': 'UklGRg=='}
    assert len(processes) == 2 and speech.decoder_graph_disabled
    processes[0].kill.assert_called_once()
    processes[1].kill.assert_not_called()
    assert requests[1]['config']['_decoder_graph'] is False
    assert '_decoder_graph' not in cfg and settings.snapshot()['speech'] == saved
