#!/bin/bash
# Targeted rerun: 9 W3 configs that failed in the overnight chain.
# - vLLM DP=1 cell C clpm_gm_dl_pe (1 config; retry just this one)
# - SGLang DP=2 (4 configs: cell C+B, both policies)
# - vLLM DP=2 (4 configs: cell C+B, both policies)
#
# Drivers skip configs whose JSON already exists, so the previously-passing
# 7 configs are not re-run.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/peek/peek-nips}"
export REPO_ROOT
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export SERVER_READY_TIMEOUT_S=3600

cd "$REPO_ROOT"

echo "[rerun] === START === $(date)"

# 1) vLLM DP=1 — re-launch will only run the 1 missing JSON (cell C PEEK)
echo "[rerun] === vLLM DP=1 (only the missing PEEK cell C will run) ==="
( source /workspace/envs/vllm/bin/activate
  SEEDS=42 PY=/workspace/envs/vllm/bin/python \
    bash benchmarks/w3/run_w3_vllm.sh ) 2>&1 | tee /tmp/w3_rerun_vllm_dp1.log

# 2) SGLang DP=2
echo
echo "[rerun] === SGLang DP=2 ==="
( source /workspace/envs/sglang/bin/activate
  SEEDS=42 CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
    CONCURRENCY=180 DECODE_MIX="" PY=/workspace/envs/sglang/bin/python \
    bash benchmarks/w3/run_w3_sglang_dp2.sh
  SEEDS=42 CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
    CONCURRENCY=180 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" PY=/workspace/envs/sglang/bin/python \
    bash benchmarks/w3/run_w3_sglang_dp2.sh ) 2>&1 | tee /tmp/w3_rerun_sglang_dp2.log

# 3) vLLM DP=2
echo
echo "[rerun] === vLLM DP=2 ==="
( source /workspace/envs/vllm/bin/activate
  SEEDS=42 CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
    CONCURRENCY=360 DECODE_MIX="" PY=/workspace/envs/vllm/bin/python \
    bash benchmarks/w3/run_w3_vllm_dp2.sh
  SEEDS=42 CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
    CONCURRENCY=360 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" PY=/workspace/envs/vllm/bin/python \
    bash benchmarks/w3/run_w3_vllm_dp2.sh ) 2>&1 | tee /tmp/w3_rerun_vllm_dp2.log

# 4) Compare
echo
echo "[rerun] === comparison vs paper ==="
/workspace/envs/vllm/bin/python benchmarks/w3/compare_to_paper.py 2>&1 | tee /tmp/w3_compare.log
cmp_rc=${PIPESTATUS[0]}
echo "[rerun] compare exit=$cmp_rc"
echo "[rerun] === DONE === $(date)"
