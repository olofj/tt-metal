# Inkling-2-Mini on one Blackhole p100a

Batch-1 decode (and chunked prefill) of
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single Blackhole p100a card
(28 GB DRAM, no Ethernet, 1x1 mesh).

## The model

| | |
|---|---|
| size | ~26B parameters, ~5B active per token |
| layers | 24; layers 6-17 run three times per token (48 layer applications) |
| attention | local (1024-token sliding window) and global (layers 5, 11, 17, 23); learned relative bias, no RoPE |
| residual | block attention residual over depth, cross-pass messages and message gates in the loop, short convs |
| MoE | 128 routed experts, top-8, sigmoid router with bias, softcapped SwiGLU |
| hidden / vocab | 2304 / 200058 (padded to 200064 on device) |
| source weights | NVFP4 experts in layers 0-19, MXFP8 in 20-23, bf16 / fp32 elsewhere |

On the card the experts are BFP4 or BFP8 (per tensor, `convert/default_dtype_map.json`), attention and the
unembed BFP8, router and residual queries fp32. The weights take about 20.5 GiB.

Code map: `convert/` checkpoint -> weight cache, `tt/` device modules (`model.py` wires them,
`generator.py` traces the decode step, `prefill_model.py` chunked prefill), `tests/` per-module checks
against the Hugging Face reference, `demo/` the demo, golden check and perf harness.

## Setup

Build tt-metal and activate its python env as usual. The scripts take their data from these variables
(or the matching command-line flags):

| variable | what |
|---|---|
| `INKLING_CKPT` | the Hugging Face checkpoint directory (safetensors, config.json, tokenizer) |
| `INKLING_MODE` | `default` or `fast`: which expert weights load (see "Run modes"; `--fast` in the demo and server) |
| `INKLING_SPEC` | `0`: no speculative decode in the server (`--no-spec`; it is on by default, see "Speculative decode"); `1`: the demo's check (`--spec`) |
| `INKLING_SPEC_MAX_CTX` | speculative decode below this position (default 143360; `--spec-max-ctx` in the server) |
| `INKLING_CACHE` | the converted weight cache the model loads (overrides the mode's) |
| `INKLING_TILE_CACHE` | a cache with the experts as tiles, for the chunked prefill and its test (overrides the mode's) |
| `INKLING_PF_GATE_BF16` | `0`: prefill keeps the loop gate messages and outputs fp32 (default `1`: bf16, faster) |
| `INKLING_GOLDEN` | CPU golden outputs (per-prompt hidden states and logits), for the golden check and perf gate |
| `INKLING_PERF_DIR` | where perf.py appends its results and keeps its baseline |
| `INKLING_REF_PYTHON` | a python with transformers >= 5.24 (Inkling2), for the HF-reference tests |
| `INKLING_CONFIG` | the model config.json for those tests (default `$INKLING_CKPT/config.json`) |

## Run modes

Two sets of expert weights, picked with `--fast` (demo, server) or `INKLING_MODE=fast` (every script).
They differ only in how many expert layers are 4-bit; the rest of the model is the same.

| mode | experts | decode cache | prefill cache | decode 4K / 128K | KL vs bf16 model |
|---|---|---|---|---:|---:|
| default | own3-def8: GPTQ, BFP4/BFP8 as `convert/default_dtype_map.json` | `own3-def8-pf` | `own3-def8-tile` | 14.6 / 19.9 ms | .0067 |
| fast | own3-l8: GPTQ, layers 0-19 all BFP4 | `own3-l8-pf` | `own3-l8-tile` | 13.1 / 18.5 ms | .0096 |

The caches live in `/work/inkling-bringup/cache/` (`modes.py`). Prefill uses the same LoFi K-blocked
defaults in both modes, with the loop gate messages stored bf16 (`INKLING_PF_GATE_BF16=0` turns that off).

## Speculative decode (off by default)

A small drafter proposes 2 tokens, one verify pass checks them with the full model, and the accepted ones are kept.
The tokens are the same as plain decode; text the drafter predicts well comes out faster.

| where | how |
|---|---|
| chat server | on by default, greedy and sampled requests; `--no-spec` or `INKLING_SPEC=0` turn it off |
| demo | `--spec` or `INKLING_SPEC=1`: each prompt also runs through spec decode; prints `spec_equals_plain`, tok/s |
| code | `tt/spec_generator.py`: `SpecGenerator.generate_rows`, `generate_rows_from` |

It needs the drafter weights: `python -m models.demos.inkling2_mini.convert.mtp` writes them once to
`<cache dir>/mtp`, where both run modes find them (`$INKLING_MTP_CACHE` points elsewhere).

Sampled requests: the model's token of each verify row is one sampler draw on that row's logits, and a draft is
kept only when the draw is that token. That is speculative sampling with a proposal that puts all its mass on the
draft (accept with probability p(draft), else draw from p without the draft), so the reply has the plain sampler's
distribution. The draws are the plain path's draws one for one: the same `seed` gives the same reply with and
without `--spec`.

Served tok/s at 4K (the server's `timings`, one card, 300 tokens unless noted; replies equal to the plain server's
in every row, sampled rows by seed). Sampled = `temperature` 0.8, `top_k` 40, `top_p` 0.95, `min_p` 0.05.

| request | default | default `--spec` | fast | fast `--spec` |
|---|---:|---:|---:|---:|
| one-sentence answer (58 tokens), greedy | 66.9 | 83.2 | 75.1 | 86.3 |
| one-sentence answer, sampled | 62.4 | 77.6 | 70.0 | 88.0 |
| Python function, greedy | 68.5 | 81.0 | 76.9 | 91.3 |
| Python function, sampled | 63.1 | 75.7 | 70.2 | 88.6 |
| short story, greedy | 68.6 | 66.2 | 76.9 | 74.4 |
| short story, sampled | 63.2 | 62.3 | 70.5 | 69.7 |
| short story, `temperature` 1, full vocab | 63.5 | 63.2 | 71.6 | 70.5 |
| short story with reasoning, greedy | 68.4 | 66.1 | 76.9 | 74.3 |
| prompt feed | 68.1 | 66.1 | 76.2 | 74.5 |

At 128K context (greedy, before the stash steps and the sampled passes): continue the text 49.8 -> 58.2 tok/s
(fast 54.1 -> 56.4), a question about the text 50.5 -> 62.7 to 72.1 (fast 54.9 -> 80.0 to 80.3).

- With `--spec` the plain step is ~3 % slower (15.13 against 14.68 ms at 4K): the expert prefetch is off (0.29 ms;
  its L1 ring does not fit beside the drafter) and the step keeps the drafter's context up to date (0.15 ms, by
  stashing its taps and committing four positions at once). The prompt feed and text the drafter can not predict
  (the story) lose that much.
- Drafting may cost a request at most 1.3 plain steps + 0.4 % over plain steps: beyond that the request runs
  plain steps and tries a draft again now and then.
- A thinking budget turns spec off until the budget is used (the forced tokens must follow an exact token).
- It never costs context. The server drafts below position 143,360 (`modes.SPEC_MAX_CTX`, `--spec-max-ctx`,
  `INKLING_SPEC_MAX_CTX`) and runs plain steps from there on, at the plain server's speed and with the same
  tokens; a reply that crosses the position goes on without a pause. The drafter keeps 4,608 B of DRAM per
  position below it (two of its three layers read the last 1,032 positions only and keep rings), beside the far
  KV's 19,584 B per token of `--max-ctx`. At `--max-ctx 196608` in default mode that leaves room for 3-row
  passes; when DRAM is short for the drafter and a verify step the server says so at start and runs without
  spec decode.

A chunked prefill (`tt/prefill_model.py`; the server does not use it) gives the drafter no context for the prompt
unless it is told to: set `Inkling2Prefill.drafter = <the DSparkDrafter>` before `prefill()` and every chunk also
writes the drafter's context (`DSparkDrafter.commit_chunk`). A process that runs that prefill has to load the TILE
cache; spec tokens equal plain there too, but its expert step runs a verify pass row by row, so spec gains nothing
in that process today.

## Convert the weights

Host only, one layer at a time (peak host RSS ~3 GB; the checkpoint is ~19 GB, the cache ~21 GB):

```
python -m models.demos.inkling2_mini.convert --ckpt $INKLING_CKPT --out $INKLING_TILE_CACHE
python -m models.demos.inkling2_mini.convert.relayout_sticks --src $INKLING_TILE_CACHE --dst $INKLING_CACHE \
    --layout bank --dtypes BFP4,BFP8
```

The first step writes a cache with every expert as tiles; the chunked prefill reads that one
(`INKLING_TILE_CACHE`). A rerun converts only the units (global, L00 ... L23) whose settings changed; the model
refuses a cache whose manifest is not complete.

The second step rewrites the expert tensors of every layer (BFP4, BFP8 and mixed) in the bank layout, where each
expert column lives in one DRAM bank and the decode MoE (`tt/fused/moe_bank.py`) reads it from its own bank. It
costs disk only for the rewritten tensors (every other file is hard-linked) and checks each one against its
source. It is ~1.9 ms/token faster than the stick layout (`--layout stick`, read by the older fused kernel).
Both layouts sum the down projection expert by expert, a different order than the tile path:
the BFP8 layers round differently (token-only 127 -> 125 on the short gate), with equal code1500 results.

Sticks and bank layouts are laid out for the card's DRAM bank count, 7 on a p100a (`convert/stick.py`
`N_BANKS`). The loader refuses a cache made for another count; for a p150 (8 banks) use `relayout_sticks --n-banks 8`.

ttnn opens the card even for host-side tensor conversions. To convert or run the host-only tests without
the card, point it at a mock cluster:

```
export TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml
```

## Run the demo

```
python -m models.demos.inkling2_mini.demo.demo --prompt "Write a haiku about a mountain lake at dawn." \
    --max-ctx 4096 --gen 64 --skip-app-trace --perf-pos
```

The prompt goes through the model's chat template and the answer is printed. Without `--prompt`, the demo
generates the golden prompts (`--golden`) and compares prompt 0 with the golden greedy tokens. It also
checks that a traced decode step replays bit-equal and matches eager, and measures decode speed at
`--perf-pos` positions.

## Chat in a browser

`server/` is a small single-user chat server: the OpenAI chat API (`/v1/chat/completions`, streamed or not,
`/v1/models`) plus the llama-server endpoints that llama.cpp's web UI (Llama Chat) reads. Each reply carries
llama-server's `timings` (prompt and generation tokens, ms, tok/s), which the UI shows under the reply; the
model's reasoning comes as `reasoning_content`.

```
python -m models.demos.inkling2_mini.server.fetch_ui       # once: Llama Chat (MIT), pinned and sha256-checked
python -m models.demos.inkling2_mini.server --ckpt $INKLING_CKPT --max-ctx 16384 [--fast] [--no-spec]
```

Then open http://127.0.0.1:8080/ once the log says the traces are captured (the UI waits while the model
loads). It binds 127.0.0.1 (`--host` to change) and holds the card until Ctrl-C. Open WebUI or any OpenAI
client works too: base URL `http://127.0.0.1:8080/v1`, any API key. The prompt is fed by the traced
decode step token by token (~decode speed); a next turn that extends the previous conversation feeds only its
new tokens. `--fake` runs the HTTP layer with a stub model and no card (`tests/test_server.py`).

## Run the tests

```
pytest models/demos/inkling2_mini/tests models/demos/inkling2_mini/convert
```

- Without `INKLING_REF_PYTHON` (and transformers < 5.24 in the tt-metal env) the HF-reference tests skip;
  without the mode's caches (or `INKLING_CACHE`) and `INKLING_GOLDEN` the full-model and prefill tests skip.
- Host only (with the mock cluster above): `tests/test_convert_pack.py` (converter packing vs the modules),
  `tests/test_perf_gate.py`, `convert/test_cache.py`. `convert/test_dequant.py` runs in the reference python.
- `tests/test_model.py`: full model vs the golden, teacher-forced per-layer PCC >= 0.99, logits PCC >= 0.98.
- `tests/test_prefill.py` takes over 5 minutes on a cold kernel cache and needs a tile-layout cache.
- `tests/test_attention_fast.py::test_use_time` is a benchmark (marker `models_performance_bare_metal`).

## Run perf

```
python -m models.demos.inkling2_mini.demo.perf --golden $INKLING_GOLDEN --results $INKLING_PERF_DIR \
    --label my-change [--long code1500 prose8k] [--decode 4096 131072]
```

Each part (golden gate, decode at 4K / 32K / 128K, long-prompt prefill) runs in its own process and prints
one JSON line, appended to `results.jsonl`. The gate fails below absolute bars (forced per-layer PCC 0.99,
logits PCC 0.98, top-1 90 %, top-5 98 %) and, if `baseline.json` exists, on a regression against it.
`--set-baseline` writes the baseline only from a run that passes the gate. Set `INKLING_CARD_LOCK` to a lock
file each part should hold, and `INKLING_CARD_RESET` to a command to run after a part times out.

## Expected numbers

Measured on one p100a (7 DRAM banks), traced decode, median of 5 x 10 steps:

| | 4K context (pos 2048) | 128K context (pos 122880) |
|---|---|---|
| decode step | ~40.2 ms (~24.9 tok/s) | ~47.1 ms (~21.2 tok/s) |

Quality against the CPU golden (bf16 HF reference), four short prompts (143 positions): teacher-forced
per-layer PCC >= 0.999 at the last layer applications, token-only greedy top-1 agreement at 124 positions,
logits PCC >= 0.99. Chunked prefill of
the 1478-token code prompt matches the teacher-forced golden at logits PCC 0.9999.
