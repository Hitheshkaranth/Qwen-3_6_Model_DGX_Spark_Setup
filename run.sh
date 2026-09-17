#!/usr/bin/env bash
# Build and run nvidia/Qwen3.6-35B-A3B-NVFP4 on a single Blackwell GPU
# (validated on NVIDIA DGX Spark / GB10). See README.md and docs/ENGINEERING.md
# for why each flag below is set the way it is.
#
# GB10 (or any unified-memory Blackwell box): keep --gpu-memory-utilization
# <= 0.70 or concurrency collapses under load — see docs/ENGINEERING.md §3.1.
set -euo pipefail

IMAGE="qwen3.6-dgx-spark:local"
NAME="qwen3.6-dgx-spark"
VOLUME="qwen3.6-dgx-spark-hf-cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the patched image (adds xgrammar==0.2.4 for tool calling; see Dockerfile).
docker build -t "$IMAGE" "$SCRIPT_DIR"

docker rm -f "$NAME" 2>/dev/null || true

exec docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8000:8000 \
  -v "$VOLUME":/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 262144 \
    --max-num-seqs 12 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 8000
