# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""A runner without a device, for the HTTP layer: every reply is the same short reasoning + answer, one token per
step, in the model's own message format (server --fake, tests/test_server.py)."""

from __future__ import annotations

import time

import torch

REPLY = (
    "<|message_model|><|content_thinking|>The user says hello. I answer briefly.<|end_message|>"
    "<|message_model|><|content_text|>Hello! I am a stub, not the model.<|end_message|><|content_model_end_sampling|>"
)


class EchoRunner:
    def __init__(self, tok, max_ctx=4096, reply=REPLY, step_s=0.0, spec=False):
        """spec: also offer spec_pass (engine.py), with a made-up drafter: every 3rd draft is wrong and every 5th
        pass declines (the engine's speculative path without a card)."""
        self.max_ctx, self.step_s = max_ctx, step_s
        self.spec, self.n_draft, self.n_pass, self.n_step = spec, 0, 0, 0
        self.script = tok.encode(reply, add_special_tokens=False)
        self.vocab = len(tok)
        self.seq = []
        self.out = {}  # pos -> the step's argmax
        self.resets = 0

    def reset(self):
        self.seq = []
        self.out = {}
        self.resets += 1

    def feed(self, token, pos, read=True):
        assert pos == len(self.seq), (pos, len(self.seq))
        self.n_step += 1
        self.seq.append(token)
        self.out[pos] = self._next()
        if self.step_s:
            time.sleep(self.step_s)

    def feed_next(self, pos):
        self.feed(self.out[pos - 1], pos)

    def token_at(self, pos):
        assert pos == len(self.seq) - 1 or pos == len(self.seq) - 2, (pos, len(self.seq))
        return self.out[pos]

    def spec_begin(self):
        pass

    def spec_ready(self):
        return self.n_step % 7 != 6  # now and then: a plain step

    def spec_pass(self, token, pos, max_rows, stop=()):
        assert pos == len(self.seq), (pos, len(self.seq))
        self.n_pass += 1
        self.n_step += 1
        if max_rows < 2 or self.n_pass % 5 == 0:
            return None
        self.seq.append(token)
        tg = [self._next()]
        while len(tg) < min(max_rows, 3):
            self.n_draft += 1
            if self.n_draft % 3 == 0:  # a wrong draft: the pass ends at the model's token
                break
            self.seq.append(tg[-1])
            tg.append(self._next())
            if tg[-2] in stop:
                break
        self.out = {len(self.seq) - 1: tg[-1]}
        return tg

    def _next(self):
        # the reply so far (longest tail of seq that is a head of the script) -> its next token; else a tail of >= 2
        # tokens found inside the script (a forced end of reasoning) -> the token after it; else the first token
        s = self.script
        for k in range(min(len(s) - 1, len(self.seq)), 0, -1):
            if self.seq[-k:] == s[:k]:
                return s[k]
        for k in range(min(len(s) - 1, len(self.seq), 8), 1, -1):
            tail = self.seq[-k:]
            for j in range(k, len(s)):
                if s[j - k : j] == tail:
                    return s[j]
        return s[0]

    def read_logits(self):
        lg = torch.full((self.vocab,), -10.0)
        lg[self.out[len(self.seq) - 1]] = 10.0
        return lg
