"""Rejected FP32 waveform-bucket experiment: padding increased runtime."""
import torch
from benchmark_chatterbox_vocoder import graph_masks


class WaveformBuckets:
    def __init__(self, model):
        self.model, self.original = model, model.forward
        self.graphs = {}
        self.replays = 0

    def forward(self, **kwargs):
        length = kwargs['x'].shape[-1]
        bucket = next((n for n in (256, 384, 512, 768, 1024) if length <= n), None)
        if bucket is None:
            return self.original(**kwargs)
        if bucket not in self.graphs:
            inputs = {}
            for key, value in kwargs.items():
                if value is None:
                    inputs[key] = None
                elif key in ('x', 'mask', 'mu', 'cond'):
                    inputs[key] = torch.nn.functional.pad(value, (0, bucket - length))
                else:
                    inputs[key] = value.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with graph_masks():
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.original(**inputs)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph, stream=stream):
                        output = self.original(**inputs)
            self.graphs[bucket] = (graph, inputs, output)
        graph, inputs, output = self.graphs[bucket]
        for key, value in kwargs.items():
            if value is not None:
                if key in ('x', 'mask', 'mu', 'cond'):
                    inputs[key].zero_()
                    inputs[key][..., :length].copy_(value)
                else:
                    inputs[key].copy_(value)
        graph.replay()
        self.replays += 1
        return output[..., :length]
