#!/bin/bash
# Reproduction wrapper: run W3 SGLang DP=1 (cells B+C) and DP=2 (cells B+C),
# both policies, seed=42 only. Uses the unmodified peek-nips drivers.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/peek/peek-nips}"
export REPO_ROOT
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PY=/workspace/envs/sglang/bin/python

# Make peek importable in any python child the bench scripts spawn
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/scripts/peek_sitecustomize:${PYTHONPATH:-}"

cd "$REPO_ROOT"

# DP=1, both cells, both policies, seed 42
echo "[wrapper] === SGLang DP=1 ==="
SEEDS=42 bash benchmarks/w3/run_w3_sglang.sh

# DP=2 cell C
echo "[wrapper] === SGLang DP=2 cell C ==="
SEEDS=42 CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
  CONCURRENCY=180 DECODE_MIX="" \
  bash benchmarks/w3/run_w3_sglang_dp2.sh

# DP=2 cell B
echo "[wrapper] === SGLang DP=2 cell B ==="
SEEDS=42 CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
  CONCURRENCY=180 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" \
  bash benchmarks/w3/run_w3_sglang_dp2.sh

echo "[wrapper] === SGLang DONE ==="
