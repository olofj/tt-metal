# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Chat glue for the Inkling-2-Mini server, no device code: prompt from messages, sampling, and the output parser
that splits the generated tokens into reasoning and content.

The model writes each reply as messages: ``<|message_model|>[name]<|content_thinking|> ... <|end_message|>`` for its
reasoning and ``<|message_model|><|content_text|> ... <|end_message|>`` for the answer, then
``<|content_model_end_sampling|>`` (the eos token).
"""

from __future__ import annotations

import torch

MESSAGE_MODEL = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
END_MESSAGE = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"
# what the UI's effort menu may send (the chat template's own levels); "off" means reasoning effort 0
EFFORT_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]


class SpecialIds:
    def __init__(self, tok):
        get = tok.convert_tokens_to_ids
        self.model, self.thinking, self.text, self.end = get(MESSAGE_MODEL), get(THINKING), get(TEXT), get(END_MESSAGE)
        self.stop = {get(EOS), get(ENDOFTEXT)}
        self.all = set(tok.all_special_ids) | {self.model, self.thinking, self.text, self.end} | self.stop
        # forced end of reasoning (thinking budget, UI "stop reasoning"): close the thinking message, open the answer
        self.end_reasoning = [self.end, self.model, self.text]


def template_kwargs(req):
    """chat template variables from an OpenAI / llama-server request: reasoning_effort, chat_template_kwargs."""
    kw = dict(req.get("chat_template_kwargs") or {})
    enable = kw.pop("enable_thinking", None)
    eff = req.get("reasoning_effort", kw.get("reasoning_effort"))
    if eff in ("off", "none") or enable is False:
        eff = "none"
    elif eff in (None, "default", "auto"):
        eff = None
    if eff is not None:
        kw["reasoning_effort"] = eff
    else:
        kw.pop("reasoning_effort", None)
    return kw


def prompt_ids(tok, messages, kwargs):
    """Token ids of the conversation in the model's chat template. The template ends with the user's (or the effort
    system) message: the model opens its own reply."""
    ids = tok.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


class Sampler:
    """Greedy when temperature <= 0 or top_k == 1 (the device argmax is enough), else temperature / top-k / top-p /
    min-p on the host logits."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None):
        self.t, self.top_p, self.top_k, self.min_p = float(temperature), float(top_p), int(top_k or 0), float(min_p)
        self.greedy = self.t <= 0 or self.top_k == 1
        self.gen = torch.Generator()
        if seed is not None and int(seed) >= 0:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

    # top-p candidates: the nucleus is a prefix of the sorted probabilities, so a partial top-k that already holds
    # top_p of the mass gives the exact nucleus without sorting the whole vocab (a full sort only when it does not)
    NUCLEUS_K = 1024

    def __call__(self, logits):
        """llama.cpp's order: top-k, top-p, min-p on the temperature-scaled softmax; exact, no full-vocab sort."""
        x = logits.float() / self.t
        idx = None
        if 0 < self.top_k < x.numel():
            x, idx = torch.topk(x, self.top_k)  # sorted, descending
        p = torch.softmax(x, dim=-1)
        if self.top_p < 1.0:
            if idx is None:
                p, idx = torch.topk(p, min(self.NUCLEUS_K, p.numel()))
                if float(p.sum()) < self.top_p:  # rare: a flat distribution
                    p, idx = torch.sort(torch.softmax(x, dim=-1), descending=True)
            keep = int((torch.cumsum(p, 0) - p < self.top_p).sum())  # >= 1: always keeps the top token
            p, idx = p[:keep], idx[:keep]
        if self.min_p > 0:
            keep = p >= self.min_p * p.max()
            p = p[keep]
            idx = keep.nonzero().reshape(-1) if idx is None else idx[keep]
        c = torch.cumsum(p, 0)  # inverse CDF draw (torch.multinomial is slow over 200K entries)
        i = min(int(torch.searchsorted(c, torch.rand(1, generator=self.gen) * c[-1])), p.numel() - 1)
        return int(idx[i]) if idx is not None else i


class _Stops:
    """stop strings on one text channel: returns (text safe to emit now, stopped)."""

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.buf = ""

    def push(self, text, final=False):
        self.buf += text
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0:
                out, self.buf = self.buf[:i], ""
                return out, True
        if final or not self.stops:
            out, self.buf = self.buf, ""
            return out, False
        # hold back a tail that could still grow into a stop string
        hold = max((k for s in self.stops for k in range(1, len(s)) if self.buf.endswith(s[:k])), default=0)
        out, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
        return out, False


class OutputParser:
    """Generated token ids -> ("reasoning" | "content", text) pieces, decoded incrementally (a piece that ends in a
    partial UTF-8 sequence waits for the next token). raw=True (reasoning_format "none"): every token, special ones
    included, goes to content."""

    def __init__(self, tok, special, stops=(), raw=False):
        self.tok, self.sp, self.raw = tok, special, raw
        self.channel = "content"  # text with no message header is the answer
        self.in_header = False
        self.seg, self.sent = [], 0
        self.stops = _Stops(stops)
        self.text = {"reasoning": "", "content": ""}
        self.n_reasoning = 0
        self.stopped = False  # a stop string matched

    def _flush(self, final=False):
        if not self.seg:
            return []
        s = self.tok.decode(self.seg, skip_special_tokens=False)
        if not final and s.endswith("�"):
            return []
        new, self.sent = s[self.sent :], len(s)
        return self._emit(self.channel, new, final)

    def _emit(self, ch, text, final=False):
        if ch == "content":
            text, hit = self.stops.push(text, final)
            self.stopped |= hit
        if not text:
            return []
        self.text[ch] += text
        return [(ch, text)]

    def _new_segment(self, channel):
        out = self._flush(final=True)
        self.seg, self.sent, self.channel = [], 0, channel
        return out

    def push(self, t):
        if self.stopped:
            return []
        if self.raw:
            self.seg.append(t)
            return self._flush()
        sp = self.sp
        if t == sp.model:
            out = self._new_segment(self.channel)
            self.in_header = True  # [name] up to the content token
            return out
        if t in sp.all:
            if t == sp.thinking:
                out = self._new_segment("reasoning")
            elif t == sp.end:
                out = self._new_segment("content")
            elif t in sp.stop:
                return self._new_segment(self.channel)
            else:  # <|content_text|> and the other content kinds (tool calls are passed on as text)
                out = self._new_segment("content")
            self.in_header = False
            return out
        if self.in_header:
            return []
        if self.channel == "reasoning":
            self.n_reasoning += 1
        self.seg.append(t)
        return self._flush()

    def finish(self):
        out = self._flush(final=True)
        if not self.stopped:
            out += self._emit("content", "", final=True)
        self.seg = []
        return out
