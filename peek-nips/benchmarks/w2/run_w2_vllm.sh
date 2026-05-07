#!/bin/bash
# W2 vLLM driver (paper §4.2, Table 14): long-document RAG on
# Qwen2.5-32B-Instruct + vLLM 0.19.1.
#
# Cell B (4× KV pressure):
#   G=40 documents × 8192-token prefixes
#   N=500 requests per seed
#   decode mix: 10:128, 25:512, 30:1024, 25:2048, 10:4096 (mean ≈1460)
#   moderate rate=0.15 req/s, heavy rate=0.20 req/s
#   concurrency=256
#
# Policies (paper Table 2 labels):
#   fcfs_apc_lru   vLLM stock FCFS + APC + LRU                        baseline
#   clpm_gm_dl_pe  cLPM + group_major + dynamic_lane + cluster evict  full PEEK
#
# Usage:
#   bash benchmarks/w2/run_w2_vllm.sh                                       # full matrix
#   POLICIES=fcfs_apc_lru SEEDS=42 RATES=heavy bash benchmarks/w2/run_w2_vllm.sh   # smoke
#
# Prerequisites:
#   1. vllm 0.19.1 installed in your active Python environment.
#   2. peek built and importable (`maturin develop --release`).
#   3. PYTHONPATH includes scripts/peek_sitecustomize/ so the patch hook
#      fires in every spawned EngineCore worker.

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
PORT="${PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w2/results_vllm}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"

# vLLM caps per-request KV reservation; cell B's 8192 prefix + worst-case
# 4096 decode = 12288, leave headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-13312}"

CELLS="${CELLS:-B}"
RATES="${RATES:-moderate heavy}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-fcfs_apc_lru clpm_gm_dl_pe}"

# Cell B fixed (only paper-reported W2 cell)
CELL_G_B=40
CELL_PREFIX_B=8192
CELL_N_B=500
CELL_WARMUP_B=100
DECODE_MIX="${DECODE_MIX:-10:128, 25:512, 30:1024, 25:2048, 10:4096}"
ZIPF_ALPHA="${ZIPF_ALPHA:-1.0}"
CONCURRENCY="${CONCURRENCY:-256}"

# Calibrated against fcfs_apc_lru baseline (paper Table 14).
cell_rate() {
  case "$1-$2" in
    B-moderate) echo 0.15 ;;
    B-heavy)    echo 0.20 ;;
    *) echo "ERR"; return 1 ;;
  esac
}

TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    fcfs_apc_lru) echo "" ;;
    fcfs_apc_pe)  echo "PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm)            echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1" ;;
    clpm_gm)         echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1" ;;
    clpm_gm_dl)      echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_pe)      echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm_gm_dl_pe)   echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  pkill -9 -if "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -if "VLLM::Worker"     2>/dev/null || true
  sleep 8
}

launch_server() {
  local policy="$1" slog="$2"
  local env_pref; env_pref="$(policy_env "$policy")"
  echo "[w2-vllm] launching $policy (env='$env_pref') -> $slog"

  local pp="${SITECUSTOMIZE_DIR}:${REPO_ROOT}/python"
  env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
      PYTHONPATH="$pp" \
      VLLM_USE_V1=1 \
      $env_pref "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --enable-prefix-caching \
      --max-model-len "$MAX_MODEL_LEN" \
      --host 127.0.0.1 --port "$PORT" \
      >>"$slog" 2>&1 &

  local t0; t0="$(date +%s)"
  while true; do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[w2-vllm]   ready after $(( $(date +%s) - t0 ))s"
      sleep 3
      return 0
    fi
    if (( $(date +%s) - t0 > SERVER_READY_TIMEOUT_S )); then
      echo "[w2-vllm]   server failed to start in ${SERVER_READY_TIMEOUT_S}s"
      return 1
    fi
    sleep 5
  done
}

# ------------------------------ bench -------------------------------------

run_bench() {
  local cell="$1" rate_label="$2" policy="$3" seed="$4"
  local outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}/rate_${rate_label}"
  mkdir -p "$outdir"
  local out="$outdir/${policy}.json"
  local blog="$outdir/_run_${policy}.log"
  if [[ -f "$out" ]]; then
    echo "[w2-vllm]   skip: $out already exists"
    return 0
  fi

  local rate; rate="$(cell_rate "$cell" "$rate_label")"
  local label; label="${policy}_cell${cell}_seed${seed}_${rate_label}"

  echo "[w2-vllm]   benching $policy (cell=$cell G=$CELL_G_B prefix=$CELL_PREFIX_B N=$CELL_N_B rate=$rate cap=$CONCURRENCY seed=$seed)"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$CELL_N_B" \
    --groups "$CELL_G_B" \
    --prefix-tokens "$CELL_PREFIX_B" \
    --decode-mix "$DECODE_MIX" \
    --rate "$rate" \
    --concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --warmup-reqs "$CELL_WARMUP_B" \
    --ttft-slo-ms "$TTFT_SLO" \
    --tpot-slo-ms "$TPOT_SLO" \
    --e2e-slo-ms "$E2E_SLO" \
    --dataset repobench \
    --distribution zipf \
    --zipf-alpha "$ZIPF_ALPHA" \
    --label "$label" \
    --output "$out" \
    --save-per-request \
    > "$blog" 2>&1
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "[w2-vllm]   BENCH FAILED rc=$rc, tail of log:"; tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w2-vllm]   wrote $out"
  return 0
}

# ------------------------------ main loop ---------------------------------

echo "[w2-vllm] === START === $(date)"
echo "[w2-vllm] cells=$CELLS rates=$RATES seeds=$SEEDS policies=$POLICIES"
echo "[w2-vllm] decode_mix=$DECODE_MIX zipf_alpha=$ZIPF_ALPHA"
echo "[w2-vllm] results -> $RESULTS_DIR"

for policy in $POLICIES; do
  echo
  echo "[w2-vllm] ##### policy: $policy #####"
  kill_server
  launch_server "$policy" "$RESULTS_DIR/_server_${policy}.log" || exit 1
  for seed in $SEEDS; do
    for cell in $CELLS; do
      for rate_label in $RATES; do
        run_bench "$cell" "$rate_label" "$policy" "$seed" || true
      done
    done
  done
done

kill_server
echo
echo "[w2-vllm] === DONE === $(date)"
