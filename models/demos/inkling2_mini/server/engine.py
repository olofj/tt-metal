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
    spec                  True: the runner also has spec_pass (speculative decode)
    spec_ready() -> bool  False while the runner wants plain steps (its drafts were not paying off lately)
    spec_pass(token, pos, max_rows, stop, sampler=None) -> list[int] | None
                          feed `token` at pos and up to max_rows - 1 drafted tokens the model agrees with, in one
                          verify pass. Returns the model's tokens [t_0 .. t_a]: t_0 .. t_{a-1} are fed (positions
                          pos + 1 .. pos + a), t_a is the next token, not fed. Nothing is fed after a token of
                          `stop`. None: nothing was fed (the drafter is not sure enough: feed() the token instead).
                          sampler: the request's Sampler (a sampled request). The model's token of row i is then
                          sampler(logits of row i), drawn row by row in order, and draft i is kept only when the
                          draw is that very token: the draws, and so the tokens, are those of plain steps with
                          the same sampler (see "Sampled requests" below)

Sampled requests under speculative decode. The drafter's proposal is one fixed token d per position (its argmax),
i.e. a proposal distribution q with q(d) = 1. Speculative sampling accepts a proposal x ~ q with probability
min(1, p(x) / q(x)) and otherwise draws from norm(max(0, p - q)), p being the sampler's distribution (after
temperature, top-k, top-p, min-p). With q(d) = 1 that is: accept d with probability p(d), else draw from p with d
removed. "Draw t ~ p; accept iff t == d; on a miss the token is t" is the same rule: P(t = d) = p(d), and given
t != d, t is distributed as p without d. So the reply's distribution is the plain sampler's exactly, and since each
row's logits are the plain step's logits bit for bit and the sampler makes one draw per token in the same order,
a seeded request gives the very tokens of the plain path.
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

    # Cost model of a pass, in plain steps of this process: at 4K a draft alone costs SPEC_COST[0], a pass with k
    # verified drafts 1 + SPEC_COST[k] (p100a, default and fast mode within 0.03). A plain step grows with the
    # position (far attention; x 1.38 at 128K) and a pass's extra time hardly does (x 1.1: its rows share the far
    # read), so the costs shrink with it: SPEC_FAR = those two slopes per 131072 positions (at 128K: 0.28, 0.60,
    # 0.84, measured 0.30, 0.59, 0.83).
    # A draft with top probability p is right about SPEC_ACC(p) of the time (the drafter is overconfident), scaled
    # per request by (accepted drafts + n0 s0) / (expected by SPEC_ACC + n0) over the drafts whose row was
    # reached: SPEC_PRIOR = (n0, s0 greedy, s0 sampled). The row count of a pass is the one with the most
    # expected tokens per step; the draft is paid by then, so no draft is verified unless that beats one plain
    # step.
    # Each draft books (tokens gained) - (its cost). Drafting may not cost a request more than SPEC_DEBT[0] +
    # SPEC_DEBT[1] x (its plain steps) over plain steps (not counted before SPEC_DEBT[2] drafts: one failed pass
    # says little): below that the request runs plain steps. It then tries one draft every SPEC_PROBE[0] steps,
    # doubling to SPEC_PROBE[1] while the probes do not gain. A probe that gains gives drafting SPEC_PROBE[2]
    # steps of new credit; once that run has gained a step the probe gap starts over. So text the drafter can not
    # predict costs a few drafts, ever more rarely, and a stretch it can predict is found within the probe gap.
    SPEC_COST = (0.35, 0.75, 1.05)
    SPEC_FAR = (0.1, 0.38)
    SPEC_PRIOR = (6.0, 1.0, 0.95)
    SPEC_DEBT = (0.7, 0.002, 4)
    SPEC_PROBE = (8, 128, 0.7)

    @staticmethod
    def SPEC_ACC(p):
        return 0.89 if p >= 0.9 else 0.95 * p * p

    def __init__(self, cache, max_ctx, trace_mib=None, log=print, spec=False, spec_early="auto"):
        """spec: speculative decode (tt/spec_generator.py): a drafter proposes SPEC_K tokens, one verify pass checks
        them; same tokens as the plain step. spec_early: how many drafts a pass verifies.
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
        self.spec_begin()
        # stash steps (TappedModel.stash): on = the group under way is stashed; pending = its positions so far
        self._stash, self._pending, self._declined, self._next = False, 0, None, 0
        self.vtail = {}  # verify row -> (tail trace, row pieces): the row's logits as fp32 rows (sampled passes)
        # tuning: per pass (pos, draft probs, k, a, start time, sampled), per plain step (pos, None, -1, stash, time)
        self.spec_log = [] if os.environ.get("INKLING_SPEC_LOG") else None
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
                stash = os.environ.get("INKLING_SPEC_STASH", "1") == "1"  # tuning: 0 = every step commits
                self.gen = SpecGenerator(m, d, verify=[VerifyStep(m, r) for r in rows], stash=stash)

                # the tails of the verify rows' logits are compiled before any capture, like every other program
                def row_tails():
                    for x in m.head_tail.rows_tiles[1 : max(rows)] if tail else []:
                        for k in self.TAIL_PIECES:
                            try:
                                for t in self._rows(x, k):
                                    ttnn.deallocate(t)
                            except RuntimeError:  # circular buffers clash with L1 buffers (as in _capture_tail)
                                pass

                self.gen.capture_rows(1536, before_capture=row_tails)
            else:
                self.gen = Generator(m)
                self.gen.capture()
            outs = self.gen.out + (getattr(self.gen, "sout", None) or [])
            if tail and all(o["logits"].buffer_address() == lg.buffer_address() for o in outs):
                self.tail = self._capture_tail(lg)
            if spec and self.tail is not None:
                self._capture_vtails(lg, max(rows))
            log(
                f"decode traces captured, {time.time() - t0:.0f} s (token chain {'on' if self.chain else 'off'}, "
                f"logits row tail {f'{self.tail_pieces} pieces' if self.tail else 'off'})"
            )
            if spec:
                log(
                    f"speculative decode on (greedy{' and sampled' if self.spec_sampled else ''} requests): passes "
                    f"of up to {max(rows)} rows, {free:.2f} GiB DRAM free before the verify steps, "
                    f"{dram_free_gib(self.dev):.2f} now"
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

    def _capture_vtails(self, lg, n):
        """self.vtail: the tail of each verify row's logits (row 0 is the plain step's buffer and tail). The rows
        of a pass are the fused head's row buffers in every verify trace; else no sampled passes."""
        ttnn, tiles = self.ttnn, self.model.head_tail.rows_tiles[:n]
        same = lambda a, b: a.buffer_address() == b.buffer_address()  # noqa: E731
        if len(tiles) < n or not same(tiles[0], lg):
            return
        if not all(same(x, t) for _, o in self.gen.vtrace.values() for x, t in zip(o["logits"], tiles)):
            return
        vt = {0: (self.tail, self.rows)}
        for r in range(1, n):
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            rows = self._rows(tiles[r], self.tail_pieces)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            vt[r] = (tid, rows)
            ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
            want = ttnn.to_torch(tiles[r]).float().reshape(-1)
            got = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in rows])
            if not torch.equal(want.view(torch.int32), got.view(torch.int32)):
                for t, _ in list(vt.values())[1:]:
                    ttnn.release_trace(self.dev, t)
                return
        self.vtail = vt

    @property
    def spec_sampled(self):
        """Sampled requests run verify passes too (else the plain step)."""
        return len(self.vtail) > 1

    def reset(self):
        self.model.reset()
        self._stash, self._pending, self._next = False, 0, 0

    def _seq(self, pos, n=1):
        """The step or pass at pos feeds n positions. A position that does not follow the last one (the state was
        set from outside) drops the stash group under way: its slots belong to the other sequence."""
        if pos != self._next:
            self._stash, self._pending = False, 0
        self._next = pos + n

    def spec_begin(self, sampled=False):
        """A new request: its drafting account and its accept scale start over."""
        self._bal, self._steps, self._n, self._mark = 0.0, 0, 0, None
        self._gap, self._wait = self.SPEC_PROBE[0], 0
        self._hit, self._exp = self.SPEC_PRIOR[0] * self.SPEC_PRIOR[2 if sampled else 1], self.SPEC_PRIOR[0]

    def _room(self):
        """Steps the account may still lose before drafting stops."""
        return self._bal + self.SPEC_DEBT[0] + self.SPEC_DEBT[1] * self._steps

    def spec_ready(self):
        gate = os.environ.get("INKLING_SPEC_GATE")  # tuning: "0" never turns drafting off, "off" never drafts
        if gate in ("0", "off"):
            return gate == "0"
        return self._room() >= 0 or self._n < self.SPEC_DEBT[2] or self._wait <= 0

    def _acc(self, p):
        return min(0.97, self.SPEC_ACC(p) * self._hit / self._exp)

    def _cost(self, pos):
        x = pos / 131072
        return [c * (1 + self.SPEC_FAR[0] * x) / (1 + self.SPEC_FAR[1] * x) for c in self.SPEC_COST]

    def _spec_note(self, k, a, pr, pos):
        """One draft at pos done: k of its tokens verified (0: declined), the first a of them kept (draft a, if
        any, was not the model's)."""
        v = a - self._cost(pos)[k]
        probe = self._room() < 0 and self._n >= self.SPEC_DEBT[2]
        self._bal, self._n = self._bal + v, self._n + 1
        if probe and v > 0:  # new credit; _mark: the account at which this run has gained a step
            self._bal = max(self._bal, self._bal - self._room() + self.SPEC_PROBE[2])
            self._mark = self._bal + 1.0
        elif self._mark is not None and self._bal >= self._mark:
            self._gap, self._mark = self.SPEC_PROBE[0], None
        if self._room() < 0 and self._n >= self.SPEC_DEBT[2]:  # stopped (again): the next probe after the gap
            if probe or self._mark is not None:
                self._gap = min(2 * self._gap, self.SPEC_PROBE[1])
            self._wait, self._mark = self._gap, None
        for i in range(min(k, a + 1)):
            self._exp += self.SPEC_ACC(pr[i])
            self._hit += i < a

    def _run(self, pos, token, read):
        """Inputs (token: None = the head's argmax already in place), replay, argmax read; then the next position's
        inputs, so the host writes them while this step runs (the queue keeps them behind it)."""
        from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

        ttnn, m, ph = self.ttnn, self.model, pos % FusedShortConv.PHASES
        m.phase = ph
        tr, outs = self.gen.trace, self.gen.out
        self._seq(pos)
        if getattr(self.gen, "strace", None):
            # a plain step that is not the fallback of a declined pass is one of a run (prompt feed, drafting off):
            # it stashes its drafter taps, from the next group start on (TappedModel.stash)
            run = self._declined != pos
            if not run:
                self._flush()
            elif self._pending == 0:
                self._stash = pos % len(self.gen.strace) == 0
                if self._stash:
                    self.drafter.set_stash_pos(pos)
            if self._stash:
                slot = pos % len(self.gen.strace)
                tr, outs = self.gen.strace[slot - ph :], self.gen.sout[slot - ph :]  # [ph] = this slot's step
                self._pending = (slot + 1) % len(self.gen.strace)  # the last slot's step commits the group
        if token is not None:
            t = ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32)
            ttnn.copy_host_to_device_tensor(t, m.tokens)
        if pos != self.pos_ready:
            self._positions(pos)
        if self.spec_log is not None:
            self.spec_log.append((pos, None, -1, int(self._stash), time.perf_counter()))
        ttnn.execute_trace(self.dev, tr[ph], cq_id=0, blocking=False)
        self.out, self._vrow = outs[ph], None
        self._steps, self._wait = self._steps + 1, self._wait - 1
        if read:
            s = self.slots[pos % 2]
            if s[1] is not None:
                ttnn.event_synchronize(s[1])  # the slot's last read has landed before it is read into again
            ttnn.copy_device_to_host_tensor(self.out["token"], s[0], blocking=False, cq_id=0)
            s[1], s[2] = ttnn.record_event(self.dev, cq_id=0), pos
        self.pos_ready = pos + 1 if pos + 1 < self.max_ctx else None
        if self.pos_ready is not None:
            self._positions(self.pos_ready)

    def _flush(self):
        """Commit the stashed positions of a group cut short (before a draft, or a committing step)."""
        if self._pending:
            self.ttnn.execute_trace(self.dev, self.gen.ftrace[self._pending], cq_id=0, blocking=False)
        self._stash, self._pending = False, 0

    def _positions(self, pos):
        self.model.positions.update(pos)
        if self.drafter is not None:  # the step also commits the drafter's context row of pos
            self.drafter.set_commit_pos(pos)

    def spec_pass(self, token, pos, max_rows, stop=(), sampler=None):
        """See the module docstring. One draft pass, the drafts' probabilities read, then (if a draft is kept) one
        verify pass over token + drafts, the model's tokens (read, or sampled row by row), and the commit of the
        accepted rows."""
        ttnn, g, d = self.ttnn, self.gen, self.drafter
        t0 = time.perf_counter()
        kmax = min(max(g.verifies) - 1, max_rows - 1, self.max_ctx - 1 - pos)
        if sampler is not None:
            kmax = min(kmax, len(self.vtail) - 1)
        if kmax < 1 or pos < 1:
            self._declined = pos
            return None
        self._seq(pos, 0)
        self._flush()
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
            a1 = self._acc(pr[0])
            exp = [1.0, 1.0 + a1] + ([1.0 + a1 + a1 * self._acc(pr[1])] if kmax > 1 else [])
            cost = self._cost(pos)
            k = max(range(len(exp)), key=lambda i: exp[i] / (1.0 + cost[i] if i else 1.0))
        elif self.spec_early is not None:
            t1, t2 = self.spec_early
            k = 0 if pr[0] < t1 else 1 if kmax < 2 or pr[0] * pr[1] < t2 else 2
        self.pos_ready = None  # the verify steps wrote their own position inputs
        if k == 0:
            if self.spec_log is not None:
                self.spec_log.append((pos, pr, 0, 0, t0, sampler is not None))
            self._spec_note(0, 0, pr, pos)
            self._declined = pos
            return None
        rows, ph = k + 1, pos % 4
        tid, out = g.vtrace[(rows, ph, g._shared(pos, rows))]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        if sampler is None:
            tgt = [int(x) for x in ttnn.to_torch(g.vtok[rows]).reshape(-1)[:rows].tolist()]
            drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()]
            a = 0
            while a < k and drafts[a] == tgt[a]:
                a += 1
                if tgt[a - 1] in stop:  # the reply ends there: nothing after it is fed
                    break
            tgt = tgt[: a + 1]
        else:
            # row r's token = one sampler draw on row r's logits, rows in order, while the draws equal the drafts
            # (the draws of plain steps on the same tokens). The rows' tails run one row ahead of the host's draws.
            for r in range(min(2, rows)):
                ttnn.execute_trace(self.dev, self.vtail[r][0], cq_id=0, blocking=False)
            drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()]
            tgt, a = [], 0
            while True:
                tgt.append(sampler(self._vrow_logits(a)))
                if a == k or tgt[a] != drafts[a] or (a > 0 and tgt[a - 1] in stop):
                    break
                a += 1
                if a + 1 < rows:
                    ttnn.execute_trace(self.dev, self.vtail[a + 1][0], cq_id=0, blocking=False)
        ttnn.execute_trace(self.dev, g.ctrace[(rows, ph, a + 1)], cq_id=0, blocking=False)
        self.out, self._vrow = None, (out, a)
        self._next = pos + a + 1
        self._spec_note(k, a, pr, pos)
        if self.spec_log is not None:
            self.spec_log.append((pos, pr, k, a, t0, sampler is not None))
        return tgt

    def _vrow_logits(self, r):
        return torch.cat([self.ttnn.to_torch(t).float().reshape(-1) for t in self.vtail[r][1]])[: self.vocab]

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
        if self._vrow is not None:  # after a verify pass: the row of the last fed token
            out, row = self._vrow
            if row in self.vtail:
                self.ttnn.execute_trace(self.dev, self.vtail[row][0], cq_id=0, blocking=False)
                return self._vrow_logits(row)
            return self.ttnn.to_torch(out["logits"][row]).float().reshape(-1)[: self.vocab]  # one TILE read, 25 MB
        if self.tail is None:
            return self.ttnn.to_torch(self.out["logits"]).float().reshape(-1)[: self.vocab]
        self.ttnn.execute_trace(self.dev, self.tail, cq_id=0, blocking=False)
        return torch.cat([self.ttnn.to_torch(t).float().reshape(-1) for t in self.rows])[: self.vocab]

    def close(self):
        if self.spec_log:
            import json

            with open(os.environ["INKLING_SPEC_LOG"], "w") as f:
                json.dump(self.spec_log, f)
        for tid in {t for t, _ in self.vtail.values()} | ({self.tail} if self.tail is not None else set()):
            self.ttnn.release_trace(self.dev, tid)
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
        greedy = job.sampler.greedy
        # speculative decode: greedy requests, and sampled ones when the runner can sample a pass's rows
        spec = getattr(r, "spec", False) and (greedy or getattr(r, "spec_sampled", False))
        if spec:
            r.spec_begin(sampled=not greedy)
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
        ahead = False  # a step on the argmax of the step at len(fed) - 1 is queued at len(fed) (greedy)
        # speculative decode: a pass feeds a token and the drafts the model agrees with, and returns the model's
        # tokens (sampled request: the sampler's draws); `known` holds those not yet handled below: (token, already
        # fed)
        known = []
        n_pass = n_pass_tok = 0

        def spec_ok():
            # not while a thinking budget may force tokens: the tokens of a pass are fed before they are handled
            ok = spec and not forced and job.thinking_budget < 0
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
                tg = None
                if spec_ok():
                    tg = r.spec_pass(
                        t, len(self.fed), limit - len(new) + 1, self.sp.stop, None if greedy else job.sampler
                    )
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
