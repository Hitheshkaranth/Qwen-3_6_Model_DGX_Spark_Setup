import concurrent.futures
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOKENIZE_URL = "http://127.0.0.1:8000/tokenize"
METRICS_URL = "http://127.0.0.1:8000/metrics"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
N = 12
MAX_TOKENS = 150

# Deterministic filler paragraph, repeated to build a long, precisely-sized prefix.
PARAGRAPH = (
    "The scheduler evaluates every candidate request against the current "
    "resource budget before admitting it into the running batch, weighing "
    "KV cache block availability, the configured maximum sequence count, "
    "and the token budget for the step. "
)


def build_text(min_tokens: int, seed: str = "") -> str:
    text = seed
    while count_tokens(text) < min_tokens:
        text += PARAGRAPH
    return text


def count_tokens(text: str) -> int:
    if not text:
        return 0
    req = urllib.request.Request(
        TOKENIZE_URL,
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["count"]


def get_metrics() -> dict:
    with urllib.request.urlopen(METRICS_URL, timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "vllm:prompt_tokens_by_source_total" in line or "vllm:time_to_first_token_seconds_sum" in line or "vllm:time_to_first_token_seconds_count" in line:
            name_labels, value = line.rsplit(" ", 1)
            out[name_labels] = float(value)
    return out


def fire(prompt: str, question: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}],
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    t1 = time.time()
    return {"wall_s": t1 - t0, "usage": body.get("usage", {})}


def run_concurrent(prompts_and_questions):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts_and_questions)) as ex:
        futs = [ex.submit(fire, p, q) for p, q in prompts_and_questions]
        return [f.result() for f in futs]


def diff_metrics(before, after, key_substr):
    total = 0.0
    for k, v in after.items():
        if key_substr in k:
            total += v - before.get(k, 0.0)
    return total


if __name__ == "__main__":
    shared_prefix = build_text(4200, seed="[SHARED SYSTEM PROMPT] ")
    print("shared_prefix tokens:", count_tokens(shared_prefix))

    questions = [f"Question variant {i}: summarize the scheduling policy in one sentence." for i in range(N)]

    # ---- Warm-up: populate the cache with the shared prefix ----
    warm = fire(shared_prefix, questions[0])
    print("warmup wall_s:", round(warm["wall_s"], 2))

    # ---- Test A: N concurrent requests sharing the now-warm prefix ----
    before_a = get_metrics()
    t0 = time.time()
    results_a = run_concurrent([(shared_prefix, q) for q in questions])
    wall_a = time.time() - t0
    after_a = get_metrics()

    cache_hit_a = diff_metrics(before_a, after_a, "local_cache_hit")
    compute_a = diff_metrics(before_a, after_a, "local_compute")
    ttft_sum_a = diff_metrics(before_a, after_a, "time_to_first_token_seconds_sum")
    ttft_count_a = diff_metrics(before_a, after_a, "time_to_first_token_seconds_count")

    # ---- Test B: N concurrent requests, each with a UNIQUE prefix (no sharing) ----
    unique_prefixes = [build_text(4200, seed=f"[UNIQUE PROMPT {i} — nonce {i*7919}] ") for i in range(N)]
    before_b = get_metrics()
    t0 = time.time()
    results_b = run_concurrent(list(zip(unique_prefixes, questions)))
    wall_b = time.time() - t0
    after_b = get_metrics()

    cache_hit_b = diff_metrics(before_b, after_b, "local_cache_hit")
    compute_b = diff_metrics(before_b, after_b, "local_compute")
    ttft_sum_b = diff_metrics(before_b, after_b, "time_to_first_token_seconds_sum")
    ttft_count_b = diff_metrics(before_b, after_b, "time_to_first_token_seconds_count")

    report = {
        "shared_prefix_tokens": count_tokens(shared_prefix),
        "test_A_shared_prefix": {
            "wall_s_total": round(wall_a, 3),
            "avg_ttft_s": round(ttft_sum_a / ttft_count_a, 3) if ttft_count_a else None,
            "cache_hit_tokens": cache_hit_a,
            "computed_tokens": compute_a,
            "cache_hit_ratio": round(cache_hit_a / (cache_hit_a + compute_a), 4) if (cache_hit_a + compute_a) else None,
            "per_request_wall_s": [round(r["wall_s"], 2) for r in results_a],
        },
        "test_B_unique_prefixes": {
            "wall_s_total": round(wall_b, 3),
            "avg_ttft_s": round(ttft_sum_b / ttft_count_b, 3) if ttft_count_b else None,
            "cache_hit_tokens": cache_hit_b,
            "computed_tokens": compute_b,
            "cache_hit_ratio": round(cache_hit_b / (cache_hit_b + compute_b), 4) if (cache_hit_b + compute_b) else None,
            "per_request_wall_s": [round(r["wall_s"], 2) for r in results_b],
        },
    }
    print(json.dumps(report, indent=2))
