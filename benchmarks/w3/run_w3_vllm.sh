#!/bin/bash
# W3 vLLM DP=1 driver (paper §4.3, multi-GPU 70B).
#
# Llama-3.1-70B-Instruct at TP=2 on a single replica (2xH100 80GB).
#   - PEEK hooks injected into vLLM's spawn child via a sitecustomize.py shim
#     on PYTHONPATH (vllm v1 spawns EngineCore as a fresh Python process --
#     parent monkey-patches don't inherit). See
#     scripts/peek_sitecustomize/sitecustomize.py.
#   - vLLM has no LPM scheduler, so its stock baseline is FCFS+APC+LRU
#     (labelled fcfs_apc_lru here).
#
# Cells (paper §4.3; same workload shapes as W1 cell C and W2 cell B,
# but on Llama-70B at TP=2):
#   C    chat:   G=88,  prefix=1500, decode=fixed 128                   admission-bound
#   B    RAG:    G=14,  prefix=4096, decode=mix(128,512,1024,2048,4096) decode-bound
#
# Policies
#   fcfs_apc_lru   vLLM stock FCFS + APC + LRU                              baseline
#   clpm_gm_dl_pe  cLPM + group_major + dynamic_lane + cluster eviction    full PEEK
#
# Usage
#   bash benchmarks/w3/run_w3_vllm.sh                                       # full matrix
#   CELLS=C POLICIES=fcfs_apc_lru SEEDS=42 bash benchmarks/w3/run_w3_vllm.sh # smoke
#   CELLS=B POLICIES="fcfs_apc_lru clpm_gm_dl_pe" SEEDS=42 bash benchmarks/w3/run_w3_vllm.sh
#
# Prerequisites:
#   1. vllm 0.19.1 installed in your active Python environment.
#   2. peek built and importable in the same env (`maturin develop --release`).
#   3. PYTHONPATH includes scripts/peek_sitecustomize/ so the patch hook fires
#      in every spawned EngineCore worker.

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# vllm installed in /workspace/peek/ (sglang uninstalled to make room).
VLLM_VENV="${VLLM_VENV:-/workspace/peek}"
PY="${PY:-python3}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w3/results_vllm}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"

# vLLM caps per-request KV reservation; set generously for cell B's 4096-prefix
# + 4096-decode worst case.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9728}"

CELLS="${CELLS:-C B}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-fcfs_apc_lru clpm_gm_dl_pe}"
RATE_LABEL="${RATE_LABEL:-heavy}"
CONCURRENCY="${CONCURRENCY:-180}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell parameters ---------------------------
# Match the sglang side exactly so cell-by-cell deltas can be compared.

declare -A CELL_GROUPS CELL_PREFIX CELL_DECODE_MIX CELL_MAX_TOKENS CELL_N CELL_WARMUP

# Cell C (chat, admission-bound) -- same as benchmarks/w3/README.md cell C.
CELL_GROUPS[C]=88
CELL_PREFIX[C]=1500
CELL_DECODE_MIX[C]=""           # empty -> use --max-tokens (fixed)
CELL_MAX_TOKENS[C]=128
CELL_N[C]=1000
CELL_WARMUP[C]=200

# Cell B (RAG, decode-bound) -- same as benchmarks/w3/README.md cell B.
CELL_GROUPS[B]=14
CELL_PREFIX[B]=4096
CELL_DECODE_MIX[B]="10:128,25:512,30:1024,25:2048,10:4096"
CELL_MAX_TOKENS[B]=4096         # ceiling; the mix dominates
CELL_N[B]=500
CELL_WARMUP[B]=100

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    fcfs_apc_lru) echo "" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  pkill -9 -if "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -if "VLLM::Worker" 2>/dev/null || true
  sleep 8
}

launch_server() {
  local policy="$1" slog="$2"
  local env_pref; env_pref="$(policy_env "$policy")"
  echo "[w3-vllm] launching $policy (env='$env_pref')"

  # PYTHONPATH includes the sitecustomize shim AND the peek source (so
  # `import peek` works inside the spawned EngineCore child).
  local pp="${SITECUSTOMIZE_DIR}:${REPO_ROOT}/python"
  env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="$pp" \
      VLLM_USE_V1=1 \
      $env_pref "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --enable-prefix-caching \
      --max-model-len "$MAX_MODEL_LEN" \
      --host 127.0.0.1 --port "$PORT" \
      >>"$slog" 2>&1 &

  local t0; t0="$(date +%s)"
  while true; do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[w3-vllm]   ready after $(( $(date +%s) - t0 ))s"
      sleep 3
      return 0
    fi
    if (( $(date +%s) - t0 > 1800 )); then
      echo "[w3-vllm]   server failed to start in 1800s"
      return 1
    fi
    sleep 5
  done
}

# vLLM does not expose /flush_cache; rely on per-policy server cycle.

# ------------------------------ bench -------------------------------------

run_bench() {
  local cell="$1" policy="$2" seed="$3"
  local outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}/rate_${RATE_LABEL}"
  mkdir -p "$outdir"
  local out="$outdir/${policy}.json"
  local blog="$outdir/_run_${policy}.log"
  if [[ -f "$out" ]]; then
    echo "[w3-vllm]   skip: $out already exists"
    return 0
  fi
  local groups="${CELL_GROUPS[$cell]}"
  local prefix="${CELL_PREFIX[$cell]}"
  local mix="${CELL_DECODE_MIX[$cell]}"
  local max_tok="${CELL_MAX_TOKENS[$cell]}"
  local n="${CELL_N[$cell]}"
  local warm="${CELL_WARMUP[$cell]}"

  local decode_args=()
  if [[ -n "$mix" ]]; then
    decode_args=(--decode-mix "$mix")
  else
    decode_args=(--max-tokens "$max_tok")
  fi

  echo "[w3-vllm]   benching $policy seed=$seed cell=$cell G=$groups prefix=$prefix concurrency=$CONCURRENCY decode='${mix:-fixed=$max_tok}' -> $out"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$n" \
    --groups "$groups" --prefix-tokens "$prefix" \
    --dataset auto \
    --distribution zipf --zipf-alpha 1.0 \
    "${decode_args[@]}" \
    --rate 0 --concurrency "$CONCURRENCY" \
    --seed "$seed" --warmup-reqs "$warm" \
    --ttft-slo-ms 2000 --tpot-slo-ms 100 --e2e-slo-ms 60000 \
    --label "${policy}_cell${cell}_seed${seed}_${RATE_LABEL}" \
    --output "$out" \
    > "$blog" 2>&1
  echo "[w3-vllm]   $policy cell=$cell seed=$seed done"
}

# ------------------------------ main loop ---------------------------------
# Policy-major loop: one vLLM server per policy, all (cell x seed) for that
# policy run back-to-back. vLLM doesn't expose /flush_cache, so we kill and
# relaunch the server BETWEEN cells to ensure each cell starts cold.

echo "[w3-vllm] === START === $(date)"
echo "[w3-vllm] cells=$CELLS policies=$POLICIES seeds=$SEEDS"

for policy in $POLICIES; do
  echo
  echo "[w3-vllm] ##### policy: $policy #####"
  for cell in $CELLS; do
    kill_server
    launch_server "$policy" "$RESULTS_DIR/_server_${policy}_cell${cell}.log" || exit 1
    for seed in $SEEDS; do
      run_bench "$cell" "$policy" "$seed"
    done
  done
done

kill_server
echo
echo "[w3-vllm] === DONE === $(date)"
