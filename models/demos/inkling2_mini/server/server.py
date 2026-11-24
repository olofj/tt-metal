# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Chat server for Inkling-2-Mini on one p100a: the OpenAI chat API plus the llama-server endpoints its web UI
(Llama Chat) reads, so that UI, Open WebUI or any OpenAI client can talk to the model.

    python -m models.demos.inkling2_mini.server --ckpt $INKLING_CKPT [--fast | --cache DIR] [--port 8080]

Endpoints: POST /v1/chat/completions (stream or not; SSE), GET /v1/models, /models, /props, /health, /slots;
POST /v1/chat/completions/control (UI "stop reasoning"), /v1/streams/lookup (no resumable streams: []).
Every reply carries llama-server's `timings` (prompt_n, prompt_ms, prompt_per_second, predicted_n, ...), the
reasoning goes in `reasoning_content`. GET / serves the Llama Chat files from --ui-dir (see fetch_ui.py).

Single user: requests queue and run one at a time on one worker thread. No CORS headers, binds 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from models.demos.inkling2_mini import modes
from models.demos.inkling2_mini.server.chat import EFFORT_LEVELS, Sampler, SpecialIds, prompt_ids, template_kwargs
from models.demos.inkling2_mini.server.engine import Engine, Job

MODEL_ID = "inkling-2-mini"
DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}  # the checkpoint's generation_config
NO_CACHE = {"/index.html", "/service-worker.js", "/sw.js", "/manifest.webmanifest", "/_app/version.json", "/build.json"}


class App:
    def __init__(self, engine, tok, model_id=MODEL_ID, ui_dir=None, chat_template="", max_tokens_default=-1):
        self.engine, self.tok, self.model_id = engine, tok, model_id
        self.ui_dir = Path(ui_dir).resolve() if ui_dir else None
        self.chat_template = chat_template
        self.max_tokens_default = max_tokens_default
        self.started = int(time.time())

    # ------------------------------------------------------------ GET bodies
    def model_info(self):
        return {
            "id": self.model_id,
            "object": "model",
            "created": self.started,
            "owned_by": "tt-metal",
            "meta": {"n_ctx": self.engine.max_ctx, "n_vocab": len(self.tok)},
        }

    def models(self):
        return {
            "object": "list",
            "data": [self.model_info()],
            "models": [
                {"name": self.model_id, "model": self.model_id, "type": "model", "capabilities": ["completion"]}
            ],
        }

    def props(self):
        params = dict(DEFAULTS, n_predict=self.max_tokens_default, stream=True)
        return {
            "default_generation_settings": {"params": params, "n_ctx": self.engine.max_ctx},
            "total_slots": 1,
            "model_alias": self.model_id,
            "model_path": self.model_id,
            "modalities": {"vision": False, "video": False, "audio": False},
            "endpoint_slots": True,
            "endpoint_props": False,
            "endpoint_metrics": False,
            "ui": self.ui_dir is not None,
            "chat_template": self.chat_template,
            "chat_template_caps": {"reasoning_effort_levels": EFFORT_LEVELS},
            "bos_token": "",
            "eos_token": "<|content_model_end_sampling|>",
            "build_info": "inkling2_mini tt-metal",
            "is_sleeping": False,
        }

    def slots(self):
        return [{"id": 0, "n_ctx": self.engine.max_ctx, "is_processing": self.engine.busy.is_set()}]

    # ------------------------------------------------------------ chat
    def make_job(self, req):
        """Request -> Job, or raises ValueError (400)."""
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages must be a non-empty list")
        try:
            ids = prompt_ids(self.tok, msgs, template_kwargs(req))
        except Exception as ex:  # noqa: BLE001  (template errors: unknown role, image parts, ...)
            raise ValueError(f"chat template: {ex}") from ex

        def get(k, default):
            v = req.get(k)
            return default if v is None else v

        try:
            sampler = Sampler(
                get("temperature", DEFAULTS["temperature"]),
                get("top_p", DEFAULTS["top_p"]),
                get("top_k", DEFAULTS["top_k"]),
                get("min_p", DEFAULTS["min_p"]),
                req.get("seed"),
            )
            mt = req.get("max_completion_tokens", req.get("max_tokens", req.get("n_predict")))
            mt = self.max_tokens_default if mt is None else int(mt)
            budget = int(get("thinking_budget_tokens", -1))
        except (TypeError, ValueError) as ex:
            raise ValueError(f"bad sampling option: {ex}") from ex
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        raw = req.get("reasoning_format") == "none"
        job = Job(ids, mt, sampler, stops, raw, budget)
        job.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        return job

    def chunk(self, job, delta=None, finish=None, **extra):
        c = {
            "id": job.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        }
        c.update(extra)
        return c

    def final(self, job, done):
        msg = {"role": "assistant", "content": done["content"]}
        if done["reasoning"]:
            msg["reasoning_content"] = done["reasoning"]
        return {
            "id": job.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": msg, "finish_reason": done["finish_reason"]}],
            "usage": usage(done),
            "timings": done["timings"],
        }


def usage(done):
    return {
        "prompt_tokens": done["prompt_tokens"],
        "completion_tokens": done["completion_tokens"],
        "total_tokens": done["prompt_tokens"] + done["completion_tokens"],
        "prompt_tokens_details": {"cached_tokens": done["timings"]["cache_n"]},
    }


def error_body(code, msg, kind="invalid_request_error"):
    return {"error": {"code": code, "message": msg, "type": kind}}


class Handler(BaseHTTPRequestHandler):
    app: App = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: one line per request on stderr
        sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _loading(self):
        e = self.app.engine
        if e.error:
            self._send(500, error_body(500, e.error, "server_error"))
            return True
        if not e.ready.is_set():
            self._send(503, error_body(503, "Loading model", "unavailable_error"))
            return True
        return False

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        app = self.app
        if path in ("/health", "/v1/health"):
            if not self._loading():
                self._send(200, {"status": "ok"})
        elif path in ("/v1/models", "/models"):
            self._send(200, app.models())
        elif path == "/props":
            if not self._loading():
                self._send(200, app.props())
        elif path == "/slots":
            self._send(200, app.slots())
        elif path in ("/tools", "/v1/stream"):
            self._send(404, error_body(404, "not supported", "not_supported_error"))
        else:
            self._static(path)

    def _static(self, path):
        root = self.app.ui_dir
        if root is None:
            self._send(
                404,
                b"no web UI: start with --ui-dir (python -m models.demos.inkling2_mini.server.fetch_ui)\n",
                "text/plain; charset=utf-8",
            )
            return
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file():
            self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        if f.suffix == ".webmanifest":
            ctype = "application/manifest+json; charset=utf-8"
        cache = "no-cache" if "/" + rel in NO_CACHE else "public, max-age=31536000, immutable"
        h = {"Cache-Control": cache}
        if rel == "index.html":  # what llama-server sends with the UI (cross-origin isolated page)
            h.update({"Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Opener-Policy": "same-origin"})
        self._send(200, f.read_bytes(), ctype, h)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as ex:
            self._send(400, error_body(400, f"bad JSON: {ex}"))
            return None

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        req = self._body()
        if req is None:
            return
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._chat(req)
        elif path == "/v1/chat/completions/control":
            cur = self.app.engine.current
            ok = cur is not None and (req.get("id") in (None, cur.id)) and "reason" in str(req.get("action", ""))
            if ok:
                cur.end_reasoning.set()
            self._send(200, {"success": bool(ok)})
        elif path == "/v1/streams/lookup":
            self._send(200, [])
        else:
            self._send(404, error_body(404, f"no route {path}", "not_found_error"))

    def _chat(self, req):
        if self._loading():
            return
        app = self.app
        try:
            job = app.make_job(req)
        except ValueError as ex:
            self._send(400, error_body(400, str(ex)))
            return
        if len(job.prompt) >= app.engine.max_ctx:
            e = error_body(
                400, f"the prompt ({len(job.prompt)} tokens) exceeds the context size", "exceed_context_size_error"
            )
            e["error"].update(n_prompt_tokens=len(job.prompt), n_ctx=app.engine.max_ctx)
            self._send(400, e)
            return
        app.engine.submit(job)
        if req.get("stream"):
            self._stream(job, req)
            return
        while True:
            kind, val = job.events.get()
            if kind == "done":
                self._send(200, app.final(job, val))
                return
            if kind == "error":
                self._send(500, error_body(500, val, "server_error"))
                return

    def _stream(self, job, req):
        app = self.app
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        progress = bool(req.get("return_progress"))
        per_token = bool(req.get("timings_per_token"))
        usage_last = bool((req.get("stream_options") or {}).get("include_usage"))
        t0 = time.perf_counter()
        n = 0

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        try:
            send(app.chunk(job, {"role": "assistant", "content": None}))
            while True:
                kind, val = job.events.get()
                if kind == "progress":
                    if progress:
                        send(app.chunk(job, prompt_progress=val))
                elif kind == "piece":
                    ch, text = val
                    n += 1
                    extra = {}
                    if per_token:  # running numbers; the last chunk has the exact ones
                        dt = time.perf_counter() - t0
                        extra["timings"] = {"predicted_n": n, "predicted_ms": dt * 1e3, "predicted_per_second": n / dt}
                    send(app.chunk(job, {"reasoning_content" if ch == "reasoning" else "content": text}, **extra))
                elif kind == "done":
                    last = app.chunk(job, finish=val["finish_reason"], timings=val["timings"])
                    if usage_last:
                        send(last)
                        last = app.chunk(job, usage=usage(val), timings=val["timings"])
                        last["choices"] = []
                    send(last)
                    break
                else:
                    send(error_body(500, val, "server_error"))
                    break
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            job.cancel.set()  # the client went away: the engine stops at its next token


def serve(app, host, port):
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def pf_cache(cache, log=print):
    """The PF-layout twin of a weight cache when there is one: `<base>-pf` next to `<base>[-<layout>]` (e.g. v1-bank
    -> v1-pf), with every expert tensor in the PF layout. With it the decode streams guessed experts ahead
    (tt/mk/expert_overlap.py, ~0.3 ms/token at 4K). Else the cache as given."""
    c = Path(cache)

    def is_pf(d):
        try:
            lay = json.loads((d / "manifest.json").read_text()).get("expert_layout") or {}
        except (OSError, ValueError):
            return False
        return bool(lay) and all(v.startswith("PF") for v in lay.values())  # PF main + PF_X extra tensors

    if is_pf(c):
        return str(c)
    base = re.sub(r"-(bank|tile|stick\d*)$", "", c.name)  # v1-bank -> v1, own2-def-tile -> own2-def
    twin = c.with_name(base + "-pf")
    if twin != c and is_pf(twin):
        log(f"cache {c} has no PF expert layout; using {twin} (--exact-cache keeps {c.name})")
        return str(twin)
    return str(c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1: this machine only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ckpt", default=os.environ.get("INKLING_CKPT"), help="checkpoint dir (tokenizer, template)")
    ap.add_argument("--cache", default=None, help="converted weight cache (default: the run mode's, see --fast)")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="fast mode (also INKLING_MODE=fast): own3-l8 experts, ~1.5 ms/token faster decode, KL .0096 vs .0067 "
        "(default mode: own3-def8). $INKLING_CACHE or --cache override either",
    )
    ap.add_argument("--max-ctx", type=int, default=16384, help="context size (decode cost grows with it)")
    ap.add_argument("--max-tokens", type=int, default=-1, help="default reply limit (-1: up to the context)")
    ap.add_argument("--ui-dir", default=None, help="Llama Chat static files (fetch_ui.py prints the path)")
    ap.add_argument("--fake", action="store_true", help="no card: a stub model that echoes (HTTP layer tests)")
    ap.add_argument("--exact-cache", action="store_true", help="load --cache as given, not its PF-layout twin")
    ap.add_argument(
        "--spec",
        action="store_true",
        default=os.environ.get("INKLING_SPEC") == "1",
        help="speculative decode (also INKLING_SPEC=1): a small drafter proposes 2 tokens, one verify pass checks "
        "them; the same tokens as without it, more tokens per second on text the drafter predicts. Sampled "
        "requests too: a draft is kept when the sampler draws it, so a seeded reply is the same as without "
        "--spec. Needs --max-ctx >= 4096 and the drafter weights "
        "(convert.mtp); off by default",
    )
    a = ap.parse_args(argv)
    if a.ckpt is None:
        ap.error("--ckpt or INKLING_CKPT is required (the tokenizer and chat template)")
    if a.cache is None:
        a.cache = str(modes.roots(a.fast)[0])
    if a.ui_dir is None:
        from models.demos.inkling2_mini.server.fetch_ui import default_dir

        d = default_dir()
        a.ui_dir = str(d) if (d / "index.html").is_file() else None

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.ckpt)
    special = SpecialIds(tok)
    log = lambda s: print(f"[inkling-server] {s}", flush=True)  # noqa: E731
    if a.fake:
        from models.demos.inkling2_mini.server.fake import EchoRunner

        make = lambda: EchoRunner(tok, a.max_ctx, spec=a.spec)  # noqa: E731
    else:
        from models.demos.inkling2_mini.server.engine import TTRunner

        cache = a.cache if a.exact_cache else pf_cache(a.cache, log)
        early = os.environ.get("INKLING_SPEC_EARLY", "auto")  # tuning: "auto", "t1,t2" thresholds, or "none"
        early = None if early == "none" else early if early == "auto" else tuple(float(x) for x in early.split(","))
        make = lambda: TTRunner(cache, a.max_ctx, log=log, spec=a.spec, spec_early=early)  # noqa: E731
    engine = Engine(make, tok, special, log=log)
    tmpl = Path(a.ckpt, "chat_template.jinja")
    app = App(
        engine,
        tok,
        ui_dir=a.ui_dir,
        chat_template=tmpl.read_text() if tmpl.is_file() else "",
        max_tokens_default=a.max_tokens,
    )
    httpd = serve(app, a.host, a.port)
    log(f"listening on http://{a.host}:{a.port}/ (UI: {a.ui_dir or 'none'}); loading the model")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def on_signal(*_):  # Ctrl-C, kill, a job timeout: finish the step, release the traces, close the device
        log("shutting down")
        # from another thread: the handler may run while this thread holds the job queue's lock
        threading.Thread(target=engine.stop, kwargs={"wait": False}).start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        engine.run()  # main thread: the device opens and closes here
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
