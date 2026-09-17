#!/usr/bin/env bash
# One-line installer for nvidia/Qwen3.6-35B-A3B-NVFP4 on a DGX Spark (GB10)
# or any single Blackwell GPU box.
#
#   curl -fsSL https://raw.githubusercontent.com/<owner>/Qwen-3_6_Model_DGX_Spark_Setup/main/install.sh | bash
#
# Clones this repo (or updates it if already present in the current
# directory) and builds + starts the server via run.sh. Safe to re-run.
set -euo pipefail

REPO_URL="${QWEN36_REPO_URL:-https://github.com/<owner>/Qwen-3_6_Model_DGX_Spark_Setup.git}"
DIR="Qwen-3_6_Model_DGX_Spark_Setup"

echo "==> Checking prerequisites..."

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install it first (e.g. 'sudo apt-get install -y git')." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install it first: https://docs.docker.com/engine/install/" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon isn't reachable (is it running, and do you have permission to use it?)." >&2
  exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "GPU not visible to Docker. Install the NVIDIA Container Toolkit, then re-run this script:" >&2
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  exit 1
fi

echo "==> Prerequisites OK."

if [ -d "$DIR/.git" ]; then
  echo "==> $DIR already exists here, pulling latest..."
  git -C "$DIR" pull --ff-only
else
  echo "==> Cloning $REPO_URL..."
  git clone "$REPO_URL" "$DIR"
fi

cd "$DIR"
echo "==> Building image and starting the server (this also downloads the ~21.8GB model on first run)..."
./run.sh

echo
echo "==> Done. Tail logs with: docker logs -f qwen3.6-dgx-spark"
echo "==> Server will be reachable at http://localhost:8000/v1 once startup completes (3-5 minutes)."
