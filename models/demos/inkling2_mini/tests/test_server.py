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
