"""Rejected waveform-graph experiment: graph capture overhead outweighed the gain."""
import logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)

@contextmanager
def graph_masks():
    """The pinned inference mask, with all-false rows handled on the GPU."""
    from chatterbox.models.s3gen import decoder
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask
    original = decoder.add_optional_chunk_mask

    def mask(xs, masks, dynamic, dynamic_left, decoding_size, static_size, left_chunks, full=True):
        if dynamic:
            return original(xs, masks, dynamic, dynamic_left, decoding_size, static_size, left_chunks, full)
        result = masks
        if static_size > 0:
            result = masks & subsequent_chunk_mask(xs.size(1), static_size, left_chunks, xs.device).unsqueeze(0)
        return result | ~result.any(dim=-1, keepdim=True)

    decoder.add_optional_chunk_mask = mask
    try:
        yield
    finally:
        decoder.add_optional_chunk_mask = original


class EstimatorGraph:
    """Lab probe: one exact-shape graph for the ten waveform diffusion steps."""
    def __init__(self, model):
        self.model, self.original = model, model.forward
        self.graph = self.inputs = self.output = self.signature = None
        self.disabled = False
        self.replays = 0

    def close(self):
        self.graph = self.inputs = self.output = self.signature = None

    def forward(self, *args, **kwargs):
        import torch
        if self.disabled or args or not kwargs or any(v is not None and (not torch.is_tensor(v) or v.device.type != 'cuda') for v in kwargs.values()):
            return self.original(*args, **kwargs)
        signature = tuple((k, (tuple(v.shape), v.dtype, v.device) if v is not None else None) for k, v in kwargs.items())
        try:
            if signature != self.signature:
                self.close()
                self.inputs = {k: v.clone() if v is not None else None for k, v in kwargs.items()}
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with graph_masks():
                    with torch.cuda.stream(stream):
                        self.original(**self.inputs)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream):
                        with torch.cuda.graph(self.graph, stream=stream):
                            self.output = self.original(**self.inputs)
                self.signature = signature
            for key, value in kwargs.items():
                if value is not None:
                    self.inputs[key].copy_(value)
            self.graph.replay()
            self.replays += 1
            return self.output
        except Exception:
            logger.exception('Waveform graph unavailable; using ordinary inference')
            self.disabled = True
            self.close()
            return self.original(*args, **kwargs)
