# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""One thread owns the model and runs one chat request at a time: `Engine.run()`, on the main thread for the card
(ttnn opens the device there and its exit handlers close it there: a device opened on another thread aborts at
exit when the driver's chip lock is released by a thread that does not own it).

A runner does the device work, one token per step:
    reset()              forget the sequence (conv histories; the KV caches are position-masked)
    feed(token, pos)     run one decode step (may return before the device is done)
    read_token() -> int  argmax of the last step
    read_logits()        fp32 host logits [vocab] of the last step
    max_ctx
`TTRunner` is the Inkling2 traced decode (tt/generator.py); the tests use a stub.

KV reuse: the engine keeps the token ids it has fed (`fed`). A request whose prompt starts with all of them feeds
only the rest; any other prompt resets and feeds from position 0. The decode state can not be rolled back (the short
conv histories hold the last positions), so a prompt that diverges inside `fed` costs a full prefill.
"""

from __future__ import annotations

import queue
import threading
import time

import torch

from models.demos.inkling2_mini.server.chat import OutputParser


class TTRunner:
    """Inkling2Model + Generator on the card: prefill = traced decode token by token, then traced decode."""

    def __init__(self, cache, max_ctx, trace_mib=512, log=print):
        import ttnn
        from models.demos.inkling2_mini.tt.device import open_device
        from models.demos.inkling2_mini.tt.generator import Generator
        from models.demos.inkling2_mini.tt.model import VOCAB_REAL, Inkling2Model

        self.ttnn, self.vocab, self.max_ctx = ttnn, VOCAB_REAL, max_ctx
        t0 = time.time()
        self.dev = open_device(trace_region_size=trace_mib << 20)
        try:
            self.model = Inkling2Model(self.dev, root=cache, max_ctx=max_ctx, log=lambda s: None)
            log(f"model loaded in {time.time() - t0:.0f} s")
            self.gen = Generator(self.model)
            self.gen.capture()
            log(f"decode traces captured, {time.time() - t0:.0f} s")
        except BaseException:
            self.ttnn.close_device(self.dev)
            raise
        self.out = None

    def reset(self):
        self.model.reset()

    def feed(self, token, pos):
        self.out = self.gen.decode(token, pos, traced=True)

    def read_token(self):
        return self.gen.read_token(self.out)

    def read_logits(self):
        return self.ttnn.to_torch(self.out["logits"]).float().reshape(-1)[: self.vocab]

    def close(self):
        self.gen.release()
        self.ttnn.close_device(self.dev)


class Job:
    """One request: the prompt and its options in, events out (`events` queue): ("progress", dict),
    ("piece", (channel, text)), ("done", dict), ("error", str)."""

    def __init__(self, prompt, max_tokens, sampler, stops=(), raw=False, thinking_budget=-1):
        self.prompt, self.max_tokens, self.sampler = list(prompt), max_tokens, sampler
        self.stops, self.raw, self.thinking_budget = list(stops), raw, thinking_budget
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.end_reasoning = threading.Event()
        self.id = None


class Engine:
    def __init__(self, make_runner, tokenizer, special, log=print):
        self.tok, self.sp, self.log = tokenizer, special, log
        self.runner = None
        self.error = None
        self.jobs = queue.Queue()
        self.busy = threading.Event()
        self.current = None
        self.fed = []
        self.last_logits = None  # host logits after fed[-1]
        self.ready = threading.Event()
        self._make = make_runner
        self.thread = None

    def start(self):
        """run() on a new thread (stub runners only; the card runner runs on the main thread)."""
        self.thread = threading.Thread(target=self.run, name="inkling-engine", daemon=True)
        self.thread.start()
        return self

    @property
    def max_ctx(self):
        return self.runner.max_ctx if self.runner else None

    def submit(self, job):
        self.jobs.put(job)
        return job

    def stop(self, wait=True):
        cur = self.current
        if cur is not None:
            cur.cancel.set()
        self.jobs.put(None)
        if wait and self.thread is not None:
            self.thread.join()

    def run(self):
        """Load the model, then run jobs until stop()."""
        try:
            self.runner = self._make()
            # the per-token host work (sampling over 200K logits) is small: one thread is ~10x faster than the
            # default pool on a busy host
            torch.set_num_threads(1)
        except Exception as ex:  # noqa: BLE001
            self.error = f"model load failed: {ex!r}"
            self.log(self.error)
            self.ready.set()
            while (job := self.jobs.get()) is not None:
                job.events.put(("error", self.error))
            return
        self.ready.set()
        try:
            while (job := self.jobs.get()) is not None:
                self.busy.set()
                self.current = job
                try:
                    self._run(job)
                except Exception as ex:  # noqa: BLE001
                    self.log(f"request failed: {ex!r}")
                    self.fed, self.last_logits = [], None  # the device state is unknown now
                    job.events.put(("error", repr(ex)))
                finally:
                    self.current = None
                    self.busy.clear()
        finally:
            close = getattr(self.runner, "close", None)
            if close:
                close()

    def _feed(self, ids, job, t0, total, cached):
        """Feed ids after self.fed; progress events every ~0.5 s. Returns False if the client went away."""
        r = self.runner
        t_last = time.perf_counter()
        for i, t in enumerate(ids):
            if job.cancel.is_set():
                return False
            r.feed(t, len(self.fed))
            self.fed.append(t)
            self.last_logits = None
            now = time.perf_counter()
            if now - t_last > 0.5:
                t_last = now
                job.events.put(
                    (
                        "progress",
                        {"total": total, "cache": cached, "processed": cached + i + 1, "time_ms": (now - t0) * 1e3},
                    )
                )
        return True

    def _next(self, job):
        if job.sampler.greedy:
            return self.runner.read_token()
        return job.sampler(self.runner.read_logits())

    def _run(self, job):
        r, prompt = self.runner, job.prompt
        if len(prompt) >= r.max_ctx:
            job.events.put(("error", f"prompt is {len(prompt)} tokens, the context is {r.max_ctx}"))
            return
        t0 = time.perf_counter()
        n = len(self.fed)
        reuse = n > 0 and prompt[:n] == self.fed and (n < len(prompt) or self.last_logits is not None)
        if not reuse:
            r.reset()
            self.fed, self.last_logits = [], None
        cached = len(self.fed)
        if not self._feed(prompt[cached:], job, t0, len(prompt), cached):
            return
        n_prompt = len(prompt) - cached

        parser = OutputParser(self.tok, self.sp, job.stops, job.raw)
        budget = r.max_ctx - len(prompt)
        limit = budget if job.max_tokens is None or job.max_tokens < 0 else min(job.max_tokens, budget)
        forced = []
        new = []
        finish = "length"
        t1 = None
        while len(new) < limit:
            if forced:
                t = forced.pop(0)
            elif self.last_logits is not None:  # prompt == fed: sample from the kept logits
                lg = self.last_logits
                t = int(lg.argmax()) if job.sampler.greedy else job.sampler(lg)
            else:
                t = self._next(job)
            if t1 is None:
                t1 = time.perf_counter()  # prompt done: first token known
            if t in self.sp.stop and not job.raw:
                finish = "stop"
                break
            new.append(t)
            for piece in parser.push(t):
                job.events.put(("piece", piece))
            if parser.stopped:
                finish = "stop"
                break
            if job.cancel.is_set():
                finish = "cancelled"
                break
            if parser.channel == "reasoning" and not forced and not parser.in_header:
                if job.end_reasoning.is_set() or 0 <= job.thinking_budget <= parser.n_reasoning:
                    forced = list(self.sp.end_reasoning)
                    job.end_reasoning.clear()
                    job.thinking_budget = -1
            if len(new) == limit:
                break
            r.feed(t, len(self.fed))
            self.fed.append(t)
            self.last_logits = None
        if t1 is None:
            t1 = time.perf_counter()
        for piece in parser.finish():
            job.events.put(("piece", piece))
        if self.last_logits is None and self.fed:
            self.last_logits = r.read_logits()  # lets a request with prompt == fed (e.g. after n_predict 0) go on
        t2 = time.perf_counter()
        n_gen = len(new)
        timings = {
            "cache_n": cached,
            "prompt_n": n_prompt,
            "prompt_ms": (t1 - t0) * 1e3,
            "prompt_per_token_ms": (t1 - t0) * 1e3 / max(n_prompt, 1),
            "prompt_per_second": n_prompt / (t1 - t0) if n_prompt and t1 > t0 else 0.0,
            "predicted_n": n_gen,
            "predicted_ms": (t2 - t1) * 1e3,
            "predicted_per_token_ms": (t2 - t1) * 1e3 / max(n_gen, 1),
            "predicted_per_second": n_gen / (t2 - t1) if n_gen and t2 > t1 else 0.0,
        }
        self.log(
            f"prompt {len(prompt)} (cached {cached}) {timings['prompt_per_second']:.1f} tok/s, "
            f"gen {n_gen} {timings['predicted_per_second']:.1f} tok/s, {finish}"
        )
        job.events.put(
            (
                "done",
                {
                    "finish_reason": "stop" if finish == "cancelled" else finish,
                    "timings": timings,
                    "prompt_tokens": len(prompt),
                    "completion_tokens": n_gen,
                    "reasoning": parser.text["reasoning"],
                    "content": parser.text["content"],
                },
            )
        )
