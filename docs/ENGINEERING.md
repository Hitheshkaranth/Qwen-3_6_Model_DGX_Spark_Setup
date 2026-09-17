# Engineering Notes

This is the technical log behind the configuration in this repository — what
was measured, what was changed, why, and the raw data behind every number in
the README. Everything here was captured from the actual running deployment
on **NVIDIA DGX Spark (GB10)**, not derived from documentation alone.

## 1. Hardware profile

```
$ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
NVIDIA GB10, 580.173.02, 12.1

$ free -h
              total        used        free      shared  buff/cache   available
Mem:          121Gi        98Gi       1.7Gi       188Mi        22Gi        22Gi
```

DGX Spark's GB10 is a **single-GPU, unified-memory** system: CPU and GPU
share the same 128GB physical memory pool (reported ~121GiB usable to the
OS). This has one direct consequence that shaped every tuning decision
below: there is no separate "VRAM" to overcommit — GPU memory pressure and
host memory pressure are the same pressure.

## 2. Model architecture internals

From the checkpoint's `config.json`
([nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)):

| Field | Value |
|---|---|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` |
| `model_type` | `qwen3_5` |
| `num_hidden_layers` | 40 |
| `mtp_num_hidden_layers` | 1 (native MTP/speculative-decoding head present) |
| `max_position_embeddings` | 262144 |
| Quantization | Mixed: NVFP4 (MLP linears) + FP8 (attention linears, KV cache) |
| Total / active params | 35B / 3B (MoE) |

**MTP was deliberately left disabled.** The checkpoint does ship a native
multi-token-prediction head (confirmed by `mtp_num_hidden_layers: 1`), and
we successfully validated MTP (`--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`) on the sibling dense model
`nvidia/Qwen3.8-27B-NVFP4` on this same box — 65% of drafted tokens were
accepted in production traffic (47,527 / 72,927,
`vllm:spec_decode_num_accepted_tokens_total` /
`vllm:spec_decode_num_draft_tokens_total`). But for *this* 35B-A3B MoE
checkpoint, the model's own published vLLM recipe does not include
`--speculative-config` in **any** of its per-hardware launch commands (GB10,
GB300, RTX 5090, RTX Pro 6000) — only the smaller dense `Qwen3.6-27B`
sibling has a validated MTP recipe. Enabling it here would be running
unvalidated configuration in production, so it was left off. This is the
single largest known unexploited performance lever for this deployment if
someone wants to benchmark it.

## 3. Configuration timeline

The server did not start in its current state — it was tuned in three
concrete, measured steps.

### 3.1 Baseline (initial deployment)

```
--model nvidia/Qwen3.6-35B-A3B-NVFP4
--gpu-memory-utilization 0.70
--max-model-len 131072
--max-num-seqs 12
--kv-cache-dtype fp8
--moe-backend marlin
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

`--gpu-memory-utilization 0.70` was already a *learned* constraint at this
point, not a default choice — earlier experimentation on this unified-memory
box found that pushing utilization higher caused concurrency to collapse
under concurrent load (symptom: throughput falling off a cliff as more
requests queued, rather than degrading gracefully). This ceiling has been
held constant through every change since.

### 3.2 Fix 1 — batched-token scheduling budget

At startup, vLLM logs (for a comparable sibling deployment on this box):

```
WARNING max_num_scheduled_tokens is set to 2048 based on the speculative
decoding settings. This may lead to suboptimal performance. Consider
increasing max_num_batched_tokens...
```

Without an explicit `--max-num-batched-tokens`, vLLM's scheduler derives a
small internal budget that throttles how much prefill work can be batched
into a single scheduling step — directly hurting time-to-first-token under
concurrent load, since one large prompt's prefill can't be chunked
efficiently against everyone else's decode steps.

**Fix applied:** `--max-num-batched-tokens 8192` — the exact value the
model's own published DGX Spark (GB10) hardware recipe specifies. Confirmed
active post-restart via the server's own startup log:

```
non-default args: {..., 'max_num_batched_tokens': 8192, ...}
INFO [scheduler.py:252] Chunked prefill is enabled with max_num_batched_tokens=8192.
```

### 3.3 Fix 2 — prefix caching

`enable_prefix_caching` was `False` in the engine's own startup config dump
— confirmed directly from the V1 engine init log line before the fix. The
model's own GB10 recipe turns this on. Prefix caching reuses KV cache across
requests that share a prompt prefix (system prompts, multi-turn
conversations), which is exactly the traffic shape of a chat deployment.

**Fix applied:** `--enable-prefix-caching`.

### 3.4 Fix 3 — context window

The model's native maximum is 262144 tokens (`max_position_embeddings`).
The server had been running with `--max-model-len 131072` — half of what
the checkpoint supports. Before raising it, KV-cache headroom was checked
at both settings using the engine's own startup log:

| `--max-model-len` | GPU KV cache size (tokens) | Max concurrency at full context |
|---|---|---|
| 131,072 | 5,656,991 | 43.16x |
| **262,144 (current)** | **5,912,141** | **22.55x** |

Doubling the context roughly halves the theoretical max concurrent
full-length requests, as expected — but 22.55x still comfortably covers the
12-concurrent-user target with margin, so the change was applied and kept.
No OOM, no errors, verified with a live chat completion immediately after
restart.

## 4. Benchmark methodology

### 4.1 Throughput history

Pulled directly from Prometheus, not synthesized:

```
GET /api/v1/query_range
  query=sum(rate(vllm:generation_tokens_total[1m]))
  query=sum(rate(vllm:prompt_tokens_total[1m]))
  step=15s, range=1h
```

Peak output observed in this window: **218.5 tok/s** (rounds to the 219
tok/s also independently visible in the Grafana panel's own `Max` legend
column for the same metric/window — two independent read paths agree).

### 4.2 12-concurrent load test

Script: [`benchmarks/load_test_12_concurrent.py`](../benchmarks/load_test_12_concurrent.py).

- 12 threads, each firing one `/v1/chat/completions` request simultaneously
  against the live server.
- 12 distinct prompts (varied topics, 20-30 prompt tokens each) to avoid
  prefix-cache and prompt-cache artifacts skewing the result.
- `max_tokens: 400`, `enable_thinking: false` (isolates decode throughput
  from variable-length reasoning traces).
- Peak instantaneous throughput independently cross-checked via Prometheus
  (`sum(rate(vllm:generation_tokens_total{instance="host.docker.internal:8000"}[15s]))`)
  sampled every 3s for the duration of the run.

Raw result:

```json
{
  "total_wall_s": 40.81,
  "total_completion_tokens": 4541,
  "total_prompt_tokens": 284,
  "aggregate_output_tps_sustained": 111.27
}
```

Per-request wall time ranged 22-41s for 400 output tokens (11/12 requests
hit the token cap; one stopped naturally at 141 tokens). The spread reflects
GPU compute being shared across 12 simultaneous decode streams — individual
per-request speed drops under contention even as aggregate throughput rises,
which is the expected and correct behavior of a batching inference server.

Prometheus-observed peak during this specific run: **224.3 tok/s**
(slightly above the 1-hour-history figure, consistent with a deliberately
saturating synthetic load vs. organic traffic).

### 4.3 Reasoning benchmarks

Not independently re-run — cited directly from NVIDIA's own published
evaluation in the
[model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4#evaluation),
BF16-vs-NVFP4, across MMLU Pro, GPQA Diamond, τ²-Bench Telecom, SciCode,
AIME 2025, AA-LCR, IFBench, and MMMU Pro. Reproducing an independent eval
run was out of scope for this deployment exercise; the model card's numbers
are reported here with full attribution rather than re-measured, so treat
them as NVIDIA's claim, not this repo's own measurement.

### 4.4 Dashboard screenshot capture

The screenshot in [`assets/grafana-dashboard.png`](../assets/grafana-dashboard.png)
is a genuine headless-browser capture of the live dashboard, not a mockup.
Notes for anyone reproducing this:

- Grafana's SPA session is cookie-based; a `user:pass@host` URL does **not**
  authenticate it (modern Chrome strips embedded URL credentials on
  top-level navigation). The working approach: `POST /login` via `curl` to
  obtain a real `grafana_session` cookie, then inject that cookie into a
  headless Chrome tab via the DevTools Protocol (`Network.setCookie`)
  before navigating.
- `Page.captureScreenshot` with `captureBeyondViewport: true` reliably
  produced a **visually correct but chart-empty** capture — panel titles
  and layout rendered, but canvas-based chart content (uPlot) never
  painted, even after long waits and explicit `resize` event dispatches.
  Switching to `captureBeyondViewport: false` (a plain in-viewport capture)
  fixed this immediately. This looks like a Chromium bug/limitation around
  compositing live canvas content into the extended "beyond viewport"
  capture mode, not a Grafana issue — worth knowing if you automate this
  again.

## 5. Monitoring stack

Sibling containers on the same host (not part of this repo, referenced for
completeness):

| Container | Role |
|---|---|
| `prometheus` | Scrapes vLLM `/metrics` (`:8000`), `dcgm-exporter` (`:9400`), `node-exporter` (`:9100`) |
| `dcgm-exporter` | GPU utilization, temperature, memory metrics via NVIDIA DCGM |
| `node-exporter` | Host CPU/RAM metrics |
| `grafana` | Dashboard visualization (the screenshot in this repo) |

Prometheus scrape config points at the vLLM job by hostname/port rather
than a fixed model name, so the same dashboard keeps working across model
swaps on this box without editing queries.

## 6. Known limitations / open items

- **MTP is unvalidated for this checkpoint** (see §2) — the largest
  plausible remaining throughput lever, untested by design.
- **vLLM version**: this deployment runs vLLM `0.24.0` (via
  `nvcr.io/nvidia/vllm:26.07-py3`). The model's own recipe elsewhere
  references `vLLM ≥ 0.28.0` for some NVFP4 code paths on other hardware
  targets — not confirmed to be a hard requirement for this specific
  GB10 + 35B-A3B combination (it has run stably on 0.24.0 throughout this
  exercise), but worth revisiting on a future vLLM upgrade.
- **Reasoning benchmarks are cited, not reproduced** (see §4.3).
