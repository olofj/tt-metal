# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""One thread owns the model and runs one chat request at a time: `Engine.run()`, on the main thread for the card
(ttnn opens the device there and its exit handlers close it there: a device opened on another thread aborts at
exit when the driver's chip lock is released by a thread that does not own it).

A runner does the device work, one token per step:
    reset()               forget the sequence (conv histories; the KV caches are position-masked)
    feed(token, pos)      run one decode step on `token` (returns before the device is done)
    feed_next(pos)        run one decode step on the argmax of the step at pos - 1, without reading it first
    token_at(pos) -> int  argmax of the step at pos
    read_logits()         fp32 host logits [vocab] of the last step run
    max_ctx
    spec                  True: the runner also has spec_pass (speculative decode, greedy requests only)
    spec_ready() -> bool  False while the runner wants plain steps (its drafts were not paying off lately)
    spec_pass(token, pos, max_rows, stop) -> list[int] | None
                          feed `token` at pos and up to max_rows - 1 drafted tokens the model agrees with, in one
                          verify pass. Returns the model's tokens [t_0 .. t_a]: t_0 .. t_{a-1} are fed (positions
                          pos + 1 .. pos + a), t_a is the next token, not fed. Nothing is fed after a token of
                          `stop`. None: nothing was fed (the drafter is not sure enough: feed() the token instead)
`TTRunner` is the Inkling2 traced decode (tt/generator.py); the tests use a stub.

Greedy decode runs one step ahead: the step on token t is queued before the host has read t, so the card goes from
step to step while the host reads the last token and turns it into text. A queued step is never taken back (the k/v
conv window is a shift register: a step can not be undone), so its token always counts as fed. When the reply ends on
a stop token, that token is fed: the next turn's prompt has it right after the reply (the template closes each model
turn with it), so the reuse below still holds.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import torch

from models.demos.inkling2_mini.server.chat import OutputParser


class TTRunner:
    """Inkling2Model + Generator on the card: prefill = traced decode token by token, then traced decode.

    The step's input token is the head's argmax buffer itself (model.tokens is a view of it), so a replay with no
    token write runs on the argmax of the replay before it (feed_next); feed() writes the host's token over it. Each
    step also queues a 4-byte read of its argmax (two host slots, one per step in flight). Sampled steps read the
    logits through a small tail trace that turns the head's TILE logits into one fp32 row (800 KB instead of the
    25 MB padded TILE tensor)."""

    SPEC_K = 2  # drafts per pass: verify passes of 2 and 3 rows
    SPEC_MIN_CTX = 4096  # the verify traces are captured around position 1536

    # Cost model of a pass, ms (p100a, both modes within 10 %): a plain step of this process takes PLAIN_MS[0] +
    # PLAIN_MS[1] * position; a pass adds SPEC_EXTRA_MS[k] to it for k verified drafts (0: the draft alone, then a
    # plain step). A draft with top probability p is right about SPEC_ACC(p) of the time (the drafter is
    # overconfident; 4 chat requests, 498 passes). The row count of a pass is the one with the most expected
    # tokens per ms. When the last SPEC_WINDOW passes took longer than plain steps would have for their tokens, the
    # next SPEC_OFF[0] tokens run as plain steps (doubling up to SPEC_OFF[1] while it stays so).
    PLAIN_MS = (15.6, 4.4e-5)
    SPEC_EXTRA_MS = (3.9, 11.3, 14.8)
    SPEC_WINDOW, SPEC_OFF = 8, (16, 128)

    @staticmethod
    def SPEC_ACC(p):
        return 0.89 if p >= 0.9 else 0.95 * p * p

    def __init__(self, cache, max_ctx, trace_mib=None, log=print, spec=False, spec_early="auto"):
        """spec: speculative decode for greedy requests (tt/spec_generator.py): a drafter proposes SPEC_K tokens, one
        verify pass checks them; same tokens as the plain step. spec_early: how many drafts a pass verifies.
        "auto": by the cost model above; (t1, t2): draft 1 when its top probability is >= t1, draft 2 when the
        product of both is >= t2; None: always SPEC_K."""
        import ttnn
        from models.demos.inkling2_mini.tt.device import open_device
        from models.demos.inkling2_mini.tt.generator import Generator
        from models.demos.inkling2_mini.tt.model import VOCAB_REAL, Inkling2Model

        self.ttnn, self.vocab, self.max_ctx = ttnn, VOCAB_REAL, max_ctx
        if spec and max_ctx < self.SPEC_MIN_CTX:
            raise ValueError(f"--spec needs --max-ctx >= {self.SPEC_MIN_CTX}")
        self.spec, self.spec_early, self.drafter = spec, spec_early, None
        self._vrow = None  # (verify outputs, row) when the last device step was a verify pass
        self._hist, self._off, self._off_n = [], 0, self.SPEC_OFF[0]  # spec_ready()
        self.spec_log = [] if os.environ.get("INKLING_SPEC_LOG") else None  # per pass: (pos, draft probs, k, a)
        t0 = time.time()
        # spec: the draft, verify (per row count, conv phase, far-chunk case) and commit traces need more room
        self.dev = open_device(trace_region_size=(trace_mib or (1024 if spec else 512)) << 20)
        self.tail, self.tail_pieces = None, 0
        try:
            if spec:
                from models.demos.inkling2_mini.tt.spec_generator import (
                    SpecGenerator,
                    build,
                    dram_free_gib,
                    verify_rows,
                )
                from models.demos.inkling2_mini.tt.verify import VerifyStep

                self.model, self.drafter = build(self.dev, max_ctx, log=lambda s: None, k=self.SPEC_K, root=cache)
                m, d = self.model, self.drafter
                d.head = d.make_head(ttnn.bfloat4_b)  # the drafts' own head copy (BFP4: 3.3 ms per draft pass)
                d.top_probs = True
            else:
                self.model = m = Inkling2Model(self.dev, root=cache, max_ctx=max_ctx, log=lambda s: None)
            log(f"model loaded in {time.time() - t0:.0f} s")
            self.chain = self._alias_token(m)
            m.set_inputs(0, 0)
            lg = m.step()["logits"]  # compiles the step; the head's logits are a module buffer, the same in the traces
            tail = lg.layout == ttnn.TILE_LAYOUT and m.head_tail is not None
            if tail:
                for k in self.TAIL_PIECES:  # compile every tail variant before any trace exists
                    for t in self._rows(lg, k):
                        ttnn.deallocate(t)
            if spec:
                # the verify steps that fit in DRAM at this max_ctx (3-row passes need 0.28 GiB more)
                free = dram_free_gib(self.dev)
                rows = verify_rows(self.dev, max_ctx)
                self.gen = SpecGenerator(m, d, verify=[VerifyStep(m, r) for r in rows])
                self.gen.capture_rows(1536)
            else:
                self.gen = Generator(m)
                self.gen.capture()
            if tail and all(o["logits"].buffer_address() == lg.buffer_address() for o in self.gen.out):
                self.tail = self._capture_tail(lg)
            log(
                f"decode traces captured, {time.time() - t0:.0f} s (token chain {'on' if self.chain else 'off'}, "
                f"logits row tail {f'{self.tail_pieces} pieces' if self.tail else 'off'})"
            )
            if spec:
                log(
                    f"speculative decode on (greedy requests): passes of up to {max(rows)} rows, "
                    f"{free:.2f} GiB DRAM free before the verify steps, {dram_free_gib(self.dev):.2f} now"
                )
        except BaseException:
            self.ttnn.close_device(self.dev)
            raise
        self.out = None
        self.pos_ready = None  # position whose inputs are already written
        host = lambda: ttnn.from_torch(torch.zeros(1, 1, 1, 1, dtype=torch.int32), dtype=ttnn.uint32)  # noqa: E731
        self.slots = [[host(), None, -1] for _ in range(2)]  # [host token, event, pos]

    def _alias_token(self, m):
        """model.tokens = a [1, 1] view of the head's argmax buffer [1, 1, 1, 1] (both uint32 ROW_MAJOR DRAM)."""
        ht = m.head_tail
        if ht is None:
            return False
        view = self.ttnn.reshape(ht.token, [1, 1])
        if view.buffer_address() != ht.token.buffer_address():
            return False
        m.tokens = view
        return True

    # The tail's circular buffers must fit under the decode's L1 staging (with a PF cache also the expert ring):
    # the logits go to rows in this many pieces, the first count that fits.
    TAIL_PIECES = (1, 2, 4, 8, 16)

    def _rows(self, lg, k):
        """[1, 1, 1, W] TILE -> k ROW_MAJOR pieces of whole tiles, in order."""
        ttnn, tiles = self.ttnn, lg.shape[-1] // 32
        out, a = [], 0
        for j in range(k):
            b = tiles * (j + 1) // k * 32
            piece = lg if k == 1 else ttnn.slice(lg, [0, 0, 0, a], [1, 1, 1, b])
            out.append(ttnn.to_layout(piece, ttnn.ROW_MAJOR_LAYOUT))
            a = b
        return out

    def _capture_tail(self, lg):
        """Trace: TILE logits -> fp32 rows. Checked once against the full TILE read; None if no piece count fits or
        the rows are not bit-equal."""
        ttnn = self.ttnn
        for k in self.TAIL_PIECES:
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            try:
                self.rows = self._rows(lg, k)
            except RuntimeError:  # circular buffers clash with L1 buffers
                ttnn.end_trace_capture(self.dev, tid, cq_id=0)
                ttnn.release_trace(self.dev, tid)
                continue
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            self.gen.decode(0, 0)
            ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
            want = ttnn.to_torch(lg).float().reshape(-1)
            got = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in self.rows])
            if torch.equal(want.view(torch.int32), got.view(torch.int32)):
                self.tail_pieces = k
                return tid
            ttnn.release_trace(self.dev, tid)
        return None

    def reset(self):
        self.model.reset()

    def spec_begin(self):
        """A new request: drafting is on again."""
        self._hist, self._off, self._off_n = [], 0, self.SPEC_OFF[0]

    def spec_ready(self):
        return self._off <= 0

    def _spec_note(self, k, tokens, pos):
        """One pass done: k drafts verified, `tokens` tokens gained. Turns drafting off for a while when it loses."""
        if os.environ.get("INKLING_SPEC_GATE") == "0":  # tuning: never turn drafting off
            return
        plain = self.PLAIN_MS[0] + self.PLAIN_MS[1] * pos
        self._hist = (self._hist + [(plain + self.SPEC_EXTRA_MS[k], tokens * plain)])[-self.SPEC_WINDOW :]
        if len(self._hist) < self.SPEC_WINDOW:
            return
        if sum(c for c, _ in self._hist) > sum(t for _, t in self._hist):
            self._off, self._hist = self._off_n, []
            self._off_n = min(2 * self._off_n, self.SPEC_OFF[1])
        else:
            self._off_n = self.SPEC_OFF[0]

    def _run(self, pos, token, read):
        """Inputs (token: None = the head's argmax already in place), replay, argmax read; then the next position's
        inputs, so the host writes them while this step runs (the queue keeps them behind it)."""
        from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

        ttnn, m, ph = self.ttnn, self.model, pos % FusedShortConv.PHASES
        m.phase = ph
        if token is not None:
            t = ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32)
            ttnn.copy_host_to_device_tensor(t, m.tokens)
        if pos != self.pos_ready:
            self._positions(pos)
        ttnn.execute_trace(self.dev, self.gen.trace[ph], cq_id=0, blocking=False)
        self.out, self._vrow = self.gen.out[ph], None
        self._off -= 1
        if read:
            s = self.slots[pos % 2]
            if s[1] is not None:
                ttnn.event_synchronize(s[1])  # the slot's last read has landed before it is read into again
            ttnn.copy_device_to_host_tensor(self.out["token"], s[0], blocking=False, cq_id=0)
            s[1], s[2] = ttnn.record_event(self.dev, cq_id=0), pos
        self.pos_ready = pos + 1 if pos + 1 < self.max_ctx else None
        if self.pos_ready is not None:
            self._positions(self.pos_ready)

    def _positions(self, pos):
        self.model.positions.update(pos)
        if self.drafter is not None:  # the step also commits the drafter's context row of pos
            self.drafter.set_commit_pos(pos)

    def spec_pass(self, token, pos, max_rows, stop=()):
        """See the module docstring. One draft pass, the drafts' probabilities read, then (if a draft is kept) one
        verify pass over token + drafts, the tokens read, and the commit of the accepted rows."""
        ttnn, g, d = self.ttnn, self.gen, self.drafter
        kmax = min(max(g.verifies) - 1, max_rows - 1, self.max_ctx - 1 - pos)
        if kmax < 1 or pos < 1:
            return None
        if self.slots[(pos - 1) % 2][1] is not None:  # a queued token read of the step before: done before new work
            ttnn.event_synchronize(self.slots[(pos - 1) % 2][1])
        # the first draft's anchor: `token` (after a plain step or a pass it is there already; not after a reuse)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32), d.anchor
        )
        d.set_draft_pos(pos)
        ttnn.execute_trace(self.dev, d.trace, cq_id=0, blocking=False)
        for v in g.verifies.values():
            v.set_positions(pos)  # host writes while the draft runs
        k = kmax
        pr = ttnn.to_torch(d.probs).float()[0, 0, 0, :kmax].tolist()
        if self.spec_early == "auto":
            plain = self.PLAIN_MS[0] + self.PLAIN_MS[1] * pos
            a1 = self.SPEC_ACC(pr[0])
            exp = [1.0, 1.0 + a1] + ([1.0 + a1 + a1 * self.SPEC_ACC(pr[1])] if kmax > 1 else [])
            k = max(range(len(exp)), key=lambda i: exp[i] / (plain + self.SPEC_EXTRA_MS[i]))
        elif self.spec_early is not None:
            t1, t2 = self.spec_early
            k = 0 if pr[0] < t1 else 1 if kmax < 2 or pr[0] * pr[1] < t2 else 2
        self.pos_ready = None  # the verify steps wrote their own position inputs
        if k == 0:
            if self.spec_log is not None:
                self.spec_log.append((pos, pr, 0, 0))
            self._spec_note(0, 1, pos)
            return None
        rows, ph = k + 1, pos % 4
        tid, out = g.vtrace[(rows, ph, g._shared(pos, rows))]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        tgt = [int(x) for x in ttnn.to_torch(g.vtok[rows]).reshape(-1)[:rows].tolist()]
        drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()]
        a = 0
        while a < k and drafts[a] == tgt[a]:
            a += 1
            if tgt[a - 1] in stop:  # the reply ends there: nothing after it is fed
                break
        ttnn.execute_trace(self.dev, g.ctrace[(rows, ph, a + 1)], cq_id=0, blocking=False)
        self.out, self._vrow = None, (out, a)
        self._spec_note(k, a + 1, pos)
        if self.spec_log is not None:
            self.spec_log.append((pos, pr, k, a))
        return tgt[: a + 1]

    def feed(self, token, pos, read=True):
        self._run(pos, token, read)

    def feed_next(self, pos):
        if not self.chain:
            return self.feed(self.token_at(pos - 1), pos)
        self._run(pos, None, True)

    def token_at(self, pos):
        s = self.slots[pos % 2]
        assert s[2] == pos, f"no token read queued for pos {pos}"
        self.ttnn.event_synchronize(s[1])
        return int(self.ttnn.to_torch(s[0]).reshape(-1)[0])

    def read_logits(self):
        if self._vrow is not None:  # after a verify pass: the row of the last fed token (one TILE read, 25 MB)
            out, row = self._vrow
            return self.ttnn.to_torch(out["logits"][row]).float().reshape(-1)[: self.vocab]
        if self.tail is None:
            return self.ttnn.to_torch(self.out["logits"]).float().reshape(-1)[: self.vocab]
        self.ttnn.execute_trace(self.dev, self.tail, cq_id=0, blocking=False)
        return torch.cat([self.ttnn.to_torch(t).float().reshape(-1) for t in self.rows])[: self.vocab]

    def close(self):
        if self.spec_log:
            import json

            with open(os.environ["INKLING_SPEC_LOG"], "w") as f:
                json.dump(self.spec_log, f)
        if self.tail is not None:
            self.ttnn.release_trace(self.dev, self.tail)
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
            r.feed(t, len(self.fed), read=i == len(ids) - 1)  # only the last step's argmax is read
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

    def _run(self, job):
        r, prompt = self.runner, job.prompt
        if len(prompt) >= r.max_ctx:
            job.events.put(("error", f"prompt is {len(prompt)} tokens, the context is {r.max_ctx}"))
            return
        t0 = time.perf_counter()
        if getattr(r, "spec", False):
            r.spec_begin()
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
        greedy = job.sampler.greedy
        ahead = False  # a step on the argmax of the step at len(fed) - 1 is queued at len(fed) (greedy)
        # speculative decode (runner.spec, greedy): a pass feeds a token and the drafts the model agrees with, and
        # returns the model's tokens; `known` holds those not yet handled below: (token, already fed)
        known = []
        n_pass = n_pass_tok = 0

        def spec_ok():
            # not while a thinking budget may force tokens: the tokens of a pass are fed before they are handled
            ok = greedy and getattr(r, "spec", False) and not forced and job.thinking_budget < 0
            return ok and r.spec_ready()

        def look_ahead():  # greedy: queue the next step on the card's own argmax before the host has read it
            nonlocal ahead
            if spec_ok() or known:
                return  # the next step is a spec pass (or its tokens are still being handled)
            if greedy and not forced and len(new) + 1 < limit and len(self.fed) < r.max_ctx:
                r.feed_next(len(self.fed))
                ahead = True

        if n_prompt:
            look_ahead()
        while len(new) < limit:
            was_fed = False
            if forced:
                t = forced.pop(0)
            elif known:
                t, was_fed = known.pop(0)
            elif self.last_logits is not None:  # prompt == fed: sample from the kept logits
                lg = self.last_logits
                t = int(lg.argmax()) if greedy else job.sampler(lg)
            elif greedy:
                t = r.token_at(len(self.fed) - 1)
            else:
                t = job.sampler(r.read_logits())
            if t1 is None:
                t1 = time.perf_counter()  # prompt done: first token known
            if t in self.sp.stop and not job.raw:
                finish = "stop"
                if ahead:  # the step on t already ran: t is fed
                    self.fed.append(t)
                    self.last_logits = None
                break
            new.append(t)
            # the step on t goes to the card before the text work below; once queued, t counts as fed
            queued, ahead = ahead, False
            if not queued and not was_fed and len(new) < limit:
                tg = r.spec_pass(t, len(self.fed), limit - len(new) + 1, self.sp.stop) if spec_ok() else None
                if tg is not None:  # t and tg[:-1] are fed now; tg[-1] is the token after them
                    self.fed += [t] + tg[:-1]
                    self.last_logits = None
                    known = [(x, True) for x in tg[:-1]] + [(tg[-1], False)]
                    n_pass, n_pass_tok = n_pass + 1, n_pass_tok + len(tg)
                else:
                    r.feed(t, len(self.fed), read=greedy)
                    queued = True
            if queued:
                self.fed.append(t)
                self.last_logits = None
            for piece in parser.push(t):
                job.events.put(("piece", piece))
            if parser.stopped:
                finish = "stop"
                break
            if job.cancel.is_set():
                finish = "cancelled"
                break
            # (not while tokens of a spec pass after this one are fed: the forced tokens would not follow them)
            if parser.channel == "reasoning" and not forced and not parser.in_header and not any(f for _, f in known):
                if job.end_reasoning.is_set() or 0 <= job.thinking_budget <= parser.n_reasoning:
                    forced = list(self.sp.end_reasoning)
                    job.end_reasoning.clear()
                    job.thinking_budget = -1
            if len(new) == limit:
                break
            look_ahead()
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
        if n_pass:  # speculative decode: verify passes run and the tokens they gave
            timings["spec_passes"], timings["spec_tokens"] = n_pass, n_pass_tok
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
