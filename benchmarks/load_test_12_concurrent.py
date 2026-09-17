import concurrent.futures
import time
import json
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
CONCURRENCY = 12
MAX_TOKENS = 400

PROMPTS = [
    "Explain how a hash table resolves collisions, in detail.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List and explain five design patterns used in backend systems.",
    "Describe the water cycle and its major stages in depth.",
    "Compare TCP and UDP for a real-time multiplayer game.",
    "Explain the difference between supervised and unsupervised learning with examples.",
    "Write a product description for a smart home thermostat.",
    "Summarize the causes of the fall of the Roman Empire.",
    "Explain how a compiler turns source code into machine code.",
    "Describe best practices for designing a REST API.",
    "Explain photosynthesis at the molecular level.",
    "Write a persuasive paragraph on why remote work benefits productivity.",
]

def one_request(i):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    t1 = time.time()
    usage = body.get("usage", {})
    return {
        "idx": i,
        "wall_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }

print(f"Firing {CONCURRENCY} concurrent requests...")
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    results = list(ex.map(one_request, range(CONCURRENCY)))
total_wall = time.time() - start

total_completion = sum(r["completion_tokens"] for r in results)
total_prompt = sum(r["prompt_tokens"] for r in results)

print(json.dumps({
    "total_wall_s": round(total_wall, 2),
    "total_completion_tokens": total_completion,
    "total_prompt_tokens": total_prompt,
    "aggregate_output_tps_sustained": round(total_completion / total_wall, 2),
    "per_request": results,
}, indent=2))
