#!/bin/bash
# Autonomous W3 reproduction chain: SGLang DP=1+DP=2, then vLLM DP=1+DP=2,
# then compare_to_paper.py. Seed=42 only.
#
# Designed to run unattended in the background — no per-stage user input
# required. Logs:
#   /tmp/w3_chain.log         (this script)
#   /tmp/w3_sglang.log        (SGLang wrapper)
#   /tmp/w3_vllm.log          (vLLM wrapper)
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/peek/peek-nips}"
export REPO_ROOT
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export SERVER_READY_TIMEOUT_S=3600

cd "$REPO_ROOT"

echo "[chain] === START === $(date)"

# 1) SGLang — activate the sglang venv so its bin (incl. ninja) is on PATH
echo "[chain] === SGLang stage ==="
( source /workspace/envs/sglang/bin/activate
  bash benchmarks/w3/_run_all_sglang_seed42.sh ) 2>&1 | tee /tmp/w3_sglang.log

# 2) vLLM
echo
echo "[chain] === vLLM stage ==="
( source /workspace/envs/vllm/bin/activate
  bash benchmarks/w3/_run_all_vllm_seed42.sh ) 2>&1 | tee /tmp/w3_vllm.log

# 3) Compare
echo
echo "[chain] === comparison vs paper ==="
/workspace/envs/vllm/bin/python benchmarks/w3/compare_to_paper.py 2>&1 | tee /tmp/w3_compare.log
cmp_rc=${PIPESTATUS[0]}

echo
echo "[chain] compare exit=$cmp_rc"
echo "[chain] === DONE === $(date)"
