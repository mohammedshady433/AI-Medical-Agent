"""Rejected experiment: a full speech-step graph saved only about 0.1 seconds.

Not enabled by the application; the small gain did not justify replacing the
vendor sampling loop. The inference order follows Resemble AI's
MIT-licensed Chatterbox T3 implementation (copyright 2025 Resemble AI).
"""
import torch
from transformers.cache_utils import StaticCache, DynamicCache
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor, TopPLogitsWarper, MinPLogitsWarper,
)


class SpeechStepGraph:
    def __init__(self, t3, capacity=512):
        self.t3, self.capacity = t3, capacity
        self.original = t3.inference
        self.forward = t3.tfmr.forward
        self.graph = self.cache = None
        self.signature = None
        self.replays = 0

    def initialize(self, logits, past, cfg, temperature, repetition, min_p, top_p):
        signature = (cfg, temperature, repetition, min_p, top_p)
        if self.signature is not None and self.signature != signature:
            raise ValueError('Experiment expects one fixed sampling profile')
        self.signature = signature
        self.cursor = int(past.get_seq_length())
        if self.cache is None:
            self.cache = StaticCache(config=self.t3.tfmr.config, max_cache_len=self.capacity)
            self.logits = logits.clone()
            self.position = torch.zeros(1, dtype=torch.long, device=logits.device)
            self.speech_position = self.position.clone()
            self.history = torch.full((1, 1001), self.t3.hp.start_speech_token, dtype=torch.long, device=logits.device)
            self.token = self.position.reshape(1, 1).clone()
            self.mask = torch.empty((2, 1, 1, self.capacity), device=logits.device, dtype=logits.dtype)
            self.slots = torch.arange(self.capacity, device=logits.device)
            self.cfg, self.temperature = cfg, temperature
            self.penalty = RepetitionPenaltyLogitsProcessor(float(repetition))
            self.min_p, self.top_p = MinPLogitsWarper(min_p), TopPLogitsWarper(top_p)

        def reset():
            for target, source in zip(self.cache.layers, past.layers):
                if not target.is_initialized:
                    target.lazy_initialization(source.keys, source.values)
                target.keys.zero_()
                target.values.zero_()
                target.keys[:, :, :self.cursor].copy_(source.keys)
                target.values[:, :, :self.cursor].copy_(source.values)
            self.logits.copy_(logits)
            self.position.fill_(self.cursor)
            self.speech_position.fill_(1)
            self.history.fill_(self.t3.hp.start_speech_token)

        reset()
        if self.graph is None:
            rng = torch.cuda.get_rng_state(logits.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(self.graph, stream=stream):
                    self.step()
            reset()
            torch.cuda.set_rng_state(rng, logits.device)

    def sample(self):
        scores = self.logits[0:1] + self.cfg * (self.logits[0:1] - self.logits[1:2])
        scores = self.penalty(self.history, scores)
        if self.temperature != 1:
            scores = scores / self.temperature
        scores = self.min_p(self.history, scores)
        scores = self.top_p(self.history, scores)
        self.token.copy_(torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1))
        self.history.scatter_(1, self.speech_position.reshape(1, 1), self.token)

    def step(self):
        self.sample()
        embeds = self.t3.speech_emb(self.token) + self.t3.speech_pos_emb.get_fixed_embedding(self.speech_position)
        embeds = torch.cat([embeds, embeds])
        self.mask.zero_()
        self.mask.masked_fill_(self.slots[None, None, None, :] > self.position, torch.finfo(embeds.dtype).min)
        out = self.forward(inputs_embeds=embeds, past_key_values=self.cache,
            cache_position=self.position, position_ids=self.position.unsqueeze(0),
            attention_mask=self.mask, use_cache=True, output_hidden_states=False,
            output_attentions=False, return_dict=True)
        self.logits.copy_(self.t3.speech_head(out.last_hidden_state)[:, -1, :])
        self.position.add_(1)
        self.speech_position.add_(1)

    @torch.inference_mode()
    def inference(self, *, t3_cond, text_tokens, max_new_tokens=1000,
                  temperature=0.8, cfg_weight=0.5, repetition_penalty=1.2,
                  min_p=0.05, top_p=0.95, **kwargs):
        assert not kwargs, kwargs
        assert max_new_tokens <= 1000
        bos = torch.full_like(text_tokens[:, :1], self.t3.hp.start_speech_token)
        embeds, _ = self.t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                               speech_tokens=bos, cfg_weight=cfg_weight)
        bos_one = bos[:1]
        bos_emb = self.t3.speech_emb(bos_one) + self.t3.speech_pos_emb.get_fixed_embedding(0)
        inputs = torch.cat([embeds, torch.cat([bos_emb, bos_emb])], dim=1)
        # Preserve the original prefill projection shape and arithmetic.
        output = self.t3.tfmr(inputs_embeds=inputs, use_cache=True, output_hidden_states=True, return_dict=True)
        logits = self.t3.speech_head(output.hidden_states[-1])[:, -1, :]
        if inputs.shape[1] >= self.capacity:
            return self.original(t3_cond=t3_cond, text_tokens=text_tokens, max_new_tokens=max_new_tokens,
                temperature=temperature, cfg_weight=cfg_weight, repetition_penalty=repetition_penalty,
                min_p=min_p, top_p=top_p)
        self.initialize(logits, output.past_key_values, cfg_weight, temperature, repetition_penalty, min_p, top_p)
        predicted = []
        dynamic = None
        for i in range(max_new_tokens):
            if self.cursor < self.capacity:
                self.graph.replay()
                self.replays += 1
                self.cursor += 1
            else:
                if dynamic is None:
                    dynamic = DynamicCache(config=self.t3.tfmr.config)
                    for index, layer in enumerate(self.cache.layers):
                        dynamic.update(layer.keys[:, :, :self.cursor].clone(), layer.values[:, :, :self.cursor].clone(), index)
                self.sample()
            token = self.token.clone()
            predicted.append(token)
            if int(token.item()) == self.t3.hp.stop_speech_token:
                break
            if dynamic is not None:
                embeds = self.t3.speech_emb(token) + self.t3.speech_pos_emb.get_fixed_embedding(i + 1)
                out = self.t3.tfmr(inputs_embeds=torch.cat([embeds, embeds]), past_key_values=dynamic,
                    use_cache=True, output_hidden_states=False, return_dict=True)
                self.logits.copy_(self.t3.speech_head(out.last_hidden_state)[:, -1, :])
                self.speech_position.add_(1)
        return torch.cat(predicted, dim=1)
