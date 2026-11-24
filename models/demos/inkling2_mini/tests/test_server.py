# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Chat server HTTP layer against a stub runner (server/fake.py): no device. Needs the checkpoint's tokenizer
($INKLING_CKPT), else skips.

    pytest models/demos/inkling2_mini/tests/test_server.py
"""

import json
import os
import threading
import urllib.error
import urllib.request

import pytest
import torch

CKPT = os.environ.get("INKLING_CKPT")
pytestmark = pytest.mark.skipif(not CKPT, reason="needs INKLING_CKPT (tokenizer + chat template)")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


@pytest.fixture
def srv(tok, tmp_path):
    from models.demos.inkling2_mini.server.chat import SpecialIds
    from models.demos.inkling2_mini.server.engine import Engine
    from models.demos.inkling2_mini.server.fake import EchoRunner
    from models.demos.inkling2_mini.server.server import App, serve

    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text("<html>ui</html>")
    (tmp_path / "secret.txt").write_text("no")
    runner = EchoRunner(tok, max_ctx=512)
    engine = Engine(lambda: runner, tok, SpecialIds(tok), log=lambda s: None).start()
    engine.ready.wait(10)
    httpd = serve(App(engine, tok, ui_dir=tmp_path / "ui"), "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, runner, engine
    httpd.shutdown()
    engine.stop()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def sse(raw):
    out = [line[6:] for line in raw.decode().split("\n") if line.startswith("data: ")]
    assert out[-1] == "[DONE]"
    return [json.loads(x) for x in out[:-1]]


USER = [{"role": "user", "content": "hello"}]


def test_info_endpoints(srv):
    base, _, _ = srv
    st, b = get(base + "/health")
    assert st == 200 and json.loads(b)["status"] == "ok"
    st, b = get(base + "/v1/models")
    assert st == 200 and json.loads(b)["data"][0]["id"] == "inkling-2-mini"
    p = json.loads(get(base + "/props")[1])
    assert p["default_generation_settings"]["n_ctx"] == 512 and p["modalities"]["vision"] is False
    assert get(base + "/")[1] == b"<html>ui</html>"
    assert get(base + "/../secret.txt")[0] == 404
    assert get(base + "/%2e%2e/secret.txt")[0] == 404
    assert json.loads(post(base + "/v1/streams/lookup", {"conversation_ids": ["x"]})[1]) == []


def test_garbage_request_line(srv, capsys):
    """A request that fails to parse (here a TLS hello on the plain-HTTP port) gets a 400 and logs one line."""
    import socket

    base, _, _ = srv
    host, port = base[len("http://") :].split(":")
    capsys.readouterr()
    with socket.create_connection((host, int(port)), timeout=10) as s:
        s.sendall(b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03 garbage\r\n\r\n")
        reply = b""
        while chunk := s.recv(4096):
            reply += chunk
    assert b" 400 " in reply.split(b"\r\n", 1)[0]
    err = capsys.readouterr().err
    assert "Traceback" not in err and "AttributeError" not in err
    assert [line for line in err.splitlines() if line.startswith("- - ")] == ["- - 400"]
    assert get(base + "/health")[0] == 200


def test_non_stream(srv):
    base, _, _ = srv
    st, b = post(base + "/v1/chat/completions", {"messages": USER, "temperature": 0})
    assert st == 200
    r = json.loads(b)
    msg = r["choices"][0]["message"]
    assert msg["content"] == "Hello! I am a stub, not the model."
    assert msg["reasoning_content"] == "The user says hello. I answer briefly."
    assert r["choices"][0]["finish_reason"] == "stop"
    t = r["timings"]
    for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"):
        assert k in t
    assert t["prompt_n"] == r["usage"]["prompt_tokens"] and t["predicted_n"] == r["usage"]["completion_tokens"]


def test_stream_and_reuse(srv):
    base, runner, _ = srv
    body = {"messages": USER, "stream": True, "temperature": 0.7, "seed": 1, "timings_per_token": True}
    st, raw = post(base + "/v1/chat/completions", body)
    assert st == 200
    chunks = sse(raw)
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert reasoning == "The user says hello. I answer briefly." and content == "Hello! I am a stub, not the model."
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop" and last["timings"]["predicted_n"] > 0
    first = last["timings"]
    resets = runner.resets

    # next turn: history as the UI sends it (reasoning_content kept) -> only the new tokens are fed
    msgs = USER + [{"role": "assistant", "content": content, "reasoning_content": reasoning}]
    msgs += [{"role": "user", "content": "again"}]
    r = json.loads(post(base + "/v1/chat/completions", {"messages": msgs, "temperature": 0})[1])
    t = r["timings"]
    assert runner.resets == resets
    assert t["cache_n"] == first["prompt_n"] + first["predicted_n"]
    assert t["cache_n"] + t["prompt_n"] == r["usage"]["prompt_tokens"]

    # a prompt that does not extend what was fed: reset, full prefill
    r = json.loads(post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})[1])
    assert runner.resets == resets + 1 and r["timings"]["cache_n"] == 0


def test_pre_encode_then_generate(srv):
    base, runner, _ = srv
    r = json.loads(post(base + "/v1/chat/completions", {"messages": USER, "n_predict": 0})[1])
    assert r["timings"]["predicted_n"] == 0 and r["timings"]["prompt_n"] > 0
    n = r["timings"]["prompt_n"]
    r = json.loads(post(base + "/v1/chat/completions", {"messages": USER, "temperature": 0})[1])
    assert r["timings"]["cache_n"] == n and r["timings"]["prompt_n"] == 0
    assert r["choices"][0]["message"]["content"] == "Hello! I am a stub, not the model."


def test_limits_and_stop(srv):
    base, _, _ = srv
    r = json.loads(post(base + "/v1/chat/completions", {"messages": USER, "max_tokens": 3})[1])
    assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 3
    r = json.loads(post(base + "/v1/chat/completions", {"messages": USER, "stop": ["stub"], "temperature": 0})[1])
    assert r["choices"][0]["message"]["content"] == "Hello! I am a " and r["choices"][0]["finish_reason"] == "stop"
    # thinking budget: reasoning cut after 2 tokens, then the answer
    r = json.loads(post(base + "/v1/chat/completions", {"messages": USER, "thinking_budget_tokens": 2})[1])
    assert r["choices"][0]["message"]["reasoning_content"] == "The user"
    st, b = post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "w " * 600}]})
    assert st == 400 and json.loads(b)["error"]["n_ctx"] == 512
    assert post(base + "/v1/chat/completions", {"messages": [{"role": "bogus", "content": "x"}]})[0] == 400


def test_sampler():
    from models.demos.inkling2_mini.server.chat import Sampler

    lg = torch.tensor([5.0, 4.9, 0.0, -1.0])
    assert {Sampler(1.0, top_k=1, seed=s)(lg) for s in range(20)} | {0} == {0}
    assert {Sampler(1.0, top_p=0.5, seed=s)(lg) for s in range(50)} == {0}  # top token alone holds >= 0.5
    assert {Sampler(1.0, top_k=2, seed=s)(lg) for s in range(50)} == {0, 1}
    assert {Sampler(1.0, min_p=0.5, seed=s)(lg) for s in range(50)} == {0, 1}

    # the draws follow the exact top-p / min-p distribution
    lg = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -1.0])
    for t, top_p, min_p in [(1.0, 1.0, 0.0), (0.8, 0.7, 0.0), (1.2, 0.9, 0.1)]:
        p = torch.softmax(lg / t, -1)  # already sorted
        keep = torch.cumsum(p, 0) - p < top_p
        keep &= p >= min_p * p.max()
        want = torch.where(keep, p, torch.zeros_like(p))
        s = Sampler(t, top_p=top_p, min_p=min_p, seed=0)
        got = torch.bincount(torch.tensor([s(lg) for _ in range(20000)]), minlength=6).float() / 20000
        assert (got - want / want.sum()).abs().max() < 0.02


def test_greedy_feeds_stop_token(srv):
    base, runner, _ = srv
    body = {"messages": USER, "temperature": 0}
    r = json.loads(post(base + "/v1/chat/completions", body)[1])
    t, msg = r["timings"], r["choices"][0]["message"]
    assert msg["content"] == "Hello! I am a stub, not the model." and r["choices"][0]["finish_reason"] == "stop"
    resets = runner.resets
    msgs = USER + [{"role": "assistant", "content": msg["content"], "reasoning_content": msg["reasoning_content"]}]
    r = json.loads(post(base + "/v1/chat/completions", {"messages": msgs + USER, "temperature": 0})[1])
    # the step queued ahead ran on the stop token: it counts as fed, and the next prompt has it after the reply
    assert runner.resets == resets and r["timings"]["cache_n"] == t["prompt_n"] + t["predicted_n"] + 1
    assert r["choices"][0]["message"]["content"] == "Hello! I am a stub, not the model."


def test_pf_cache(tmp_path):
    from models.demos.inkling2_mini.server.server import pf_cache

    def cache(name, layout):
        (tmp_path / name).mkdir()
        lay = {"L00/e": layout, "L00/e.x": layout + "_X"}
        (tmp_path / name / "manifest.json").write_text(json.dumps({"expert_layout": lay}))
        return str(tmp_path / name)

    bank, pf, stick = cache("v1-bank", "BANK"), cache("v1-pf", "PF"), cache("v2-stick", "STICK")
    assert pf_cache(bank, log=lambda s: None) == pf and pf_cache(pf) == pf and pf_cache(stick) == stick
    tile, pf2 = cache("own2-def-tile", "TILE"), cache("own2-def-pf", "PF")  # dashed base names keep their dashes
    assert pf_cache(tile, log=lambda s: None) == pf2


def test_modes(monkeypatch, expect_error):
    from models.demos.inkling2_mini import modes

    d = modes.CACHE_DIR
    for v in ("INKLING_MODE", "INKLING_CACHE", "INKLING_TILE_CACHE"):
        monkeypatch.delenv(v, raising=False)
    assert modes.roots() == (d / "own3-def8-pf", d / "own3-def8-tile")
    assert modes.roots(fast=True) == (d / "own3-l8-pf", d / "own3-l8-tile")
    monkeypatch.setenv("INKLING_MODE", "fast")
    assert modes.roots() == modes.roots(fast=True)
    monkeypatch.setenv("INKLING_CACHE", "/x")  # an explicit cache wins over the mode
    assert modes.roots()[0].as_posix() == "/x" and modes.roots()[1] == d / "own3-l8-tile"
    monkeypatch.setenv("INKLING_MODE", "slow")
    with expect_error(ValueError, "INKLING_MODE"):
        modes.roots()


def test_greedy_thinking_budget_and_limits(srv):
    """The one-step-ahead path with forced tokens, a token limit and a stop string."""
    base, _, _ = srv
    body = {"messages": USER, "temperature": 0}
    r = json.loads(post(base + "/v1/chat/completions", dict(body, thinking_budget_tokens=2))[1])
    assert r["choices"][0]["message"]["reasoning_content"] == "The user"
    assert r["choices"][0]["message"]["content"] == "Hello! I am a stub, not the model."
    for n in (1, 2, 3):
        r = json.loads(post(base + "/v1/chat/completions", dict(body, max_tokens=n))[1])
        assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == n
    r = json.loads(post(base + "/v1/chat/completions", dict(body, stop=["stub"]))[1])
    assert r["choices"][0]["message"]["content"] == "Hello! I am a "


def test_spec_engine(tok):
    """The engine's speculative path on the stub runner (made-up drafts, some wrong, some passes declined): the same
    replies, usage and next-turn reuse as the plain path; a thinking budget, token limits and a stop string too."""
    from models.demos.inkling2_mini.server.chat import SpecialIds
    from models.demos.inkling2_mini.server.engine import Engine
    from models.demos.inkling2_mini.server.fake import EchoRunner
    from models.demos.inkling2_mini.server.server import App, serve

    def run(spec, bodies):
        runner = EchoRunner(tok, max_ctx=512, spec=spec)
        engine = Engine(lambda: runner, tok, SpecialIds(tok), log=lambda s: None).start()
        engine.ready.wait(10)
        httpd = serve(App(engine, tok), "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = []
        try:
            for b in bodies:
                r = json.loads(post(base + "/v1/chat/completions", b)[1])
                m = r["choices"][0]
                out.append((m["message"].get("reasoning_content"), m["message"]["content"], m["finish_reason"],
                            r["usage"]["completion_tokens"], r["timings"].get("spec_passes", 0),
                            r["timings"]["cache_n"], runner.resets))  # fmt: skip
        finally:
            httpd.shutdown()
            engine.stop()
        return out

    body = {"messages": USER, "temperature": 0}
    again = USER + [{"role": "assistant", "content": "Hello! I am a stub, not the model.",
                     "reasoning_content": "The user says hello. I answer briefly."}] + USER  # fmt: skip
    bodies = [body, {"messages": again, "temperature": 0}, dict(body, thinking_budget_tokens=2)]
    bodies += [dict(body, max_tokens=n) for n in (1, 2, 3, 7)] + [dict(body, stop=["stub"])]
    plain, spec = run(False, bodies), run(True, bodies)
    assert [p[:4] for p in plain] == [s[:4] for s in spec]
    assert all(p[4] == 0 for p in plain) and spec[0][4] > 0 and spec[1][4] > 0
    # the second turn reuses the first reply in both: no reset; the stop token is fed by the plain path's step
    # ahead, and by a pass only when the drafter drafted it
    assert spec[1][5] in (plain[1][5] - 1, plain[1][5]) and spec[1][6] == spec[0][6]


class _RandomRunner:
    """A random "model" for the engine: the logits after a token sequence are a fixed function of its last two
    tokens (64 live tokens, one of them a stop token). spec_pass as engine.TTRunner.spec_pass: the drafts are the
    argmax tokens, every 4th one wrong."""

    def __init__(self, vocab, stop, max_ctx=512, spec=False, spec_ctx=None):
        self.vocab, self.max_ctx, self.spec, self.spec_sampled = vocab, max_ctx, spec, spec
        self.spec_ctx, self.top = spec_ctx or max_ctx, 0  # passes below spec_ctx only; top = the highest pass
        self.live = torch.tensor(list(range(100, 163)) + [stop])
        self.seq, self.n_draft, self.passes, self.kept = [], 0, 0, 0

    def reset(self):
        self.seq = []

    def _logits(self, seq):
        g = torch.Generator().manual_seed(1 + seq[-1] * 1009 + (seq[-2] if len(seq) > 1 else 0))
        lg = torch.full((self.vocab,), -30.0)
        lg[self.live] = torch.randn(64, generator=g) * 2.5
        return lg

    def feed(self, token, pos, read=True):
        assert pos == len(self.seq)
        self.seq.append(token)

    def feed_next(self, pos):
        self.feed(self.token_at(pos - 1), pos)

    def token_at(self, pos):
        return int(self._logits(self.seq[: pos + 1]).argmax())

    def read_logits(self):
        return self._logits(self.seq)

    def spec_begin(self, sampled=False):
        pass

    def spec_ready(self):
        return len(self.seq) < self.spec_ctx

    def spec_pass(self, token, pos, max_rows, stop=(), sampler=None):
        assert pos == len(self.seq) and pos < self.spec_ctx
        self.top = max(self.top, pos)
        k = min(2, max_rows - 1)
        if k < 1 or pos % 5 == 4:
            return None
        seq, drafts = self.seq + [token], []
        for _ in range(k):  # the drafter: the model's argmax, every 4th draft another token
            self.n_draft += 1
            d = int(self._logits(seq + drafts).argmax())
            drafts.append(d if self.n_draft % 4 else int(self.live[self.n_draft % 63]))
        tgt, a = [], 0
        while True:
            lg = self._logits(seq + drafts[:a])
            tgt.append(int(lg.argmax()) if sampler is None else sampler(lg))
            if a == k or tgt[a] != drafts[a] or (a > 0 and tgt[a - 1] in stop):
                break
            a += 1
        self.seq = seq + drafts[:a]
        self.passes, self.kept = self.passes + 1, self.kept + a
        return tgt


def test_spec_sampled_engine(tok):
    """Sampled requests through speculative passes: for every seed the very tokens of the plain path (each row's
    token is the sampler's draw on that row's logits, in order), so the same distribution; drafts are both kept and
    dropped on the way."""
    from models.demos.inkling2_mini.server.chat import Sampler, SpecialIds
    from models.demos.inkling2_mini.server.engine import Engine, Job

    sp = SpecialIds(tok)
    stop = sorted(sp.stop)[0]

    def run(spec, raw, spec_ctx=None):
        runner = _RandomRunner(len(tok), stop, spec=spec, spec_ctx=spec_ctx)
        engine = Engine(lambda: runner, tok, sp, log=lambda s: None).start()
        engine.ready.wait(10)
        out = []
        try:
            for seed in range(24):
                for kw in ({}, {"top_k": 40, "top_p": 0.95, "min_p": 0.05}, {"temperature": 0.0}):
                    job = engine.submit(
                        Job([5, 6, 7 + seed], 40, Sampler(**{"temperature": 0.8, "seed": seed, **kw}), raw=raw)
                    )
                    while (ev := job.events.get(timeout=10))[0] != "done":
                        assert ev[0] != "error", ev
                    out.append(
                        (ev[1]["content"], ev[1]["reasoning"], ev[1]["completion_tokens"], ev[1]["finish_reason"])
                    )
        finally:
            engine.stop()
        return out, runner

    for raw in (False, True):
        (plain, _), (spec, r) = run(False, raw), run(True, raw)
        assert plain == spec
        assert r.passes > 100 and 0.2 < r.kept / (2 * r.passes) < 0.9, (r.passes, r.kept)
        # passes below position 20 only, plain steps from there on (the drafter's positions end inside the reply)
        cross, rc = run(True, raw, spec_ctx=20)
        assert plain == cross and 0 < rc.passes < r.passes and rc.top == 19, (rc.passes, rc.top)
    assert len({p[0] for p in plain}) > 40 and any(p[3] == "stop" for p in run(False, False)[0])
