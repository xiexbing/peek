#!/bin/bash
# Reproduction wrapper: run W3 vLLM DP=1 (cells B+C) and DP=2 (cells B+C),
# both policies, seed=42 only. Uses the unmodified peek-nips drivers.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/peek/peek-nips}"
export REPO_ROOT
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PY=/workspace/envs/vllm/bin/python
export SITECUSTOMIZE_DIR="$REPO_ROOT/scripts/peek_sitecustomize"

cd "$REPO_ROOT"

# DP=1, both cells, both policies, seed 42
echo "[wrapper] === vLLM DP=1 ==="
SEEDS=42 bash benchmarks/w3/run_w3_vllm.sh

# DP=2 cell C
echo "[wrapper] === vLLM DP=2 cell C ==="
SEEDS=42 CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
  CONCURRENCY=360 DECODE_MIX="" \
  bash benchmarks/w3/run_w3_vllm_dp2.sh

# DP=2 cell B
echo "[wrapper] === vLLM DP=2 cell B ==="
SEEDS=42 CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
  CONCURRENCY=360 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" \
  bash benchmarks/w3/run_w3_vllm_dp2.sh

echo "[wrapper] === vLLM DONE ==="
