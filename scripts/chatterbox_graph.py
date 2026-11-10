"""Bounded decoder graphs for the pinned local Chatterbox runtime."""
import logging
from speech_cancellation import SpeechCancelled, SpeechWorkerRestart


logger = logging.getLogger(__name__)


def supported(model, cfg):
    """Keep unmeasured runtimes, CPU and nonresident profiles on ordinary FP32."""
    if not cfg.get('keep_resident') or cfg['tts']['device'] != 'cuda' or not cfg.get('_decoder_graph', True):
        return False
    import torch
    import transformers
    return (torch.__version__.split('+')[0] == '2.6.0' and transformers.__version__ == '5.2.0'
            and model.config.model_type == 'llama' and model.config._attn_implementation == 'sdpa'
            and not model.training and next(model.parameters()).dtype in
            (torch.float32, getattr(torch, 'float16', None)))


def compact_supported(model, cfg):
    """Eligibility for static conversion of the measured V3 voice components."""
    if (not supported(model, cfg) or not cfg.get('_compact_decoder', True)
            or cfg['tts'].get('chatterbox_version', 'v3') != 'v3'):
        return False
    import torch
    return torch.cuda.get_device_capability(next(model.parameters()).device)[0] >= 8


class DecoderGraph:
    def __init__(self, model, capacity=512, dtype=None):
        self.model, self.original = model, model.forward
        self.capacity = capacity
        self.dtype = dtype
        self.graph = self.cache = self.input = self.position = self.mask = self.output = self.slots = None
        self.cursor = 0
        self.disabled = False
        self.replays = 0
        self.capture_attempted = False

    def install(self):
        self.model.forward = self.forward
        return self

    def close(self):
        self.graph = self.cache = self.input = self.position = self.mask = self.output = self.slots = None
        self.cursor = 0

    def uninstall(self):
        self.model.forward = self.original
        self.close()

    def dynamic(self):
        from transformers.cache_utils import DynamicCache
        result = DynamicCache(config=self.model.config)
        for index, layer in enumerate(self.cache.layers):
            result.update(layer.keys[:, :, :self.cursor].clone(), layer.values[:, :, :self.cursor].clone(), index)
        return result

    def initialize(self, embeds, past):
        import torch
        from transformers.cache_utils import StaticCache

        self.cursor = int(past.get_seq_length())
        if self.cache is None:
            if torch.cuda.mem_get_info(embeds.device)[0] < 768 * 1024**2:
                raise MemoryError('Insufficient free GPU memory for the optional decoder graph')
            self.cache = StaticCache(config=self.model.config, max_cache_len=self.capacity)
        for target, source in zip(self.cache.layers, past.layers):
            if not target.is_initialized:
                target.lazy_initialization(source.keys, source.values)
            target.keys.zero_()
            target.values.zero_()
            target.keys[:, :, :self.cursor].copy_(source.keys)
            target.values[:, :, :self.cursor].copy_(source.values)
        if self.graph is not None:
            return
        self.input = torch.empty_like(embeds)
        self.position = torch.zeros(1, device=embeds.device, dtype=torch.long)
        self.mask = torch.empty((2, 1, 1, self.capacity), device=embeds.device, dtype=embeds.dtype)
        self.slots = torch.arange(self.capacity, device=embeds.device)
        self.set_input(embeds)

        def step():
            return self.original(inputs_embeds=self.input, past_key_values=self.cache,
                cache_position=self.position, position_ids=self.position.unsqueeze(0),
                attention_mask=self.mask, use_cache=True, output_hidden_states=False,
                output_attentions=False, return_dict=True).last_hidden_state

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_attempted = True
        # An outer stream context also restores the caller after capture failure.
        with torch.cuda.stream(stream):
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = step()

    def set_input(self, embeds):
        import torch
        self.input.copy_(embeds)
        self.position.fill_(self.cursor)
        self.mask.zero_()
        self.mask.masked_fill_(self.slots[None, None, None, :] > self.position, torch.finfo(embeds.dtype).min)

    def forward(self, *args, **kwargs):
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPast
        embeds, past = kwargs.get('inputs_embeds'), kwargs.get('past_key_values')
        if (self.disabled or args or embeds is None or embeds.device.type != 'cuda'
                or embeds.dtype != (self.dtype or torch.float32) or embeds.shape[:2] != (2, 1)
                or past is None or kwargs.get('output_attentions')):
            return self.original(*args, **kwargs)
        size = self.cursor if past is self.cache else int(past.get_seq_length())
        if size >= self.capacity:
            if past is self.cache:
                kwargs['past_key_values'] = self.dynamic()
            return self.original(*args, **kwargs)
        try:
            if past is not self.cache:
                self.initialize(embeds, past)
            self.set_input(embeds)
            self.graph.replay()
            self.cursor += 1
            self.replays += 1
            return BaseModelOutputWithPast(last_hidden_state=self.output, past_key_values=self.cache,
                hidden_states=(self.output,) if kwargs.get('output_hidden_states') else None)
        except Exception as error:
            if self.capture_attempted:
                # CUDA can retain an invalid capture state; do not reuse that process.
                self.disabled = True
                self.close()
                raise SpeechWorkerRestart('Restart local speech without CUDA graph acceleration') from error
            if past is self.cache:
                kwargs['past_key_values'] = self.dynamic()
            logger.exception('Local decoder graph unavailable; using ordinary FP32 inference')
            self.disabled = True
            self.close()
            return self.original(*args, **kwargs)


class CompactDecoderGraph(DecoderGraph):
    """Static FP16 transformer with the existing FP32 boundary on both sides.

    Avoid per-token autocast and keep the speech projection/sampling outside
    reduced precision. Uninstall releases the graph, not the precision change;
    switching back to FP32 requires a fresh worker loading the source weights.
    """
    def __init__(self, model, capacity=512):
        import torch
        try:
            model.half()
        except Exception as error:
            raise SpeechWorkerRestart('Reload the source voice weights in FP32') from error
        super().__init__(model, capacity, dtype=torch.float16)

    def forward(self, *args, **kwargs):
        try:
            if kwargs.get('inputs_embeds') is not None:
                kwargs = {**kwargs, 'inputs_embeds': kwargs['inputs_embeds'].to(dtype=self.dtype)}
            result = super().forward(*args, **kwargs)
            result.last_hidden_state = result.last_hidden_state.float()
            if result.hidden_states is not None:
                result.hidden_states = tuple(state.float() for state in result.hidden_states)
            return result
        except (SpeechCancelled, SpeechWorkerRestart):
            raise
        except Exception as error:
            raise SpeechWorkerRestart('Restart local speech with the original FP32 decoder') from error
