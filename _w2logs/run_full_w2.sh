#!/bin/bash
# Detached driver: full W2 matrix on sglang, then vllm. Sequential (single GPU).
# Resumable: each engine script skips result JSONs that already exist.
set -uo pipefail

cd /workspace/peek
. "$HOME/.cargo/env" 2>/dev/null || true
export PATH="/usr/local/cuda/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p /workspace/peek/_w2logs

STAMP="/workspace/peek/_w2logs/STATUS"
echo "START $(date -u +%FT%TZ)" > "$STAMP"

# ---- 1. SGLang full matrix (cells A B C D0-D4 x moderate,heavy x 3 seeds x 8 policies)
echo "SGLANG_START $(date -u +%FT%TZ)" >> "$STAMP"
PY=/root/venvs/peek-sglang/bin/python \
  bash benchmarks/w2/run_w2_sglang.sh > /workspace/peek/_w2logs/full_sglang.log 2>&1
echo "SGLANG_DONE rc=$? $(date -u +%FT%TZ)" >> "$STAMP"

# ---- Free the GPU before switching engines.
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 15

# ---- 2. vLLM full matrix (cell B x moderate,heavy x 3 seeds x 2 policies)
echo "VLLM_START $(date -u +%FT%TZ)" >> "$STAMP"
PY=/root/venvs/peek-vllm/bin/python \
  bash benchmarks/w2/run_w2_vllm.sh > /workspace/peek/_w2logs/full_vllm.log 2>&1
echo "VLLM_DONE rc=$? $(date -u +%FT%TZ)" >> "$STAMP"

pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
echo "ALL_DONE $(date -u +%FT%TZ)" >> "$STAMP"
