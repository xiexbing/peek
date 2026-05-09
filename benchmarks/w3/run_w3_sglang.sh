#!/bin/bash
# W3 SGLang DP=1 driver (paper §4.3, Table 15): Llama-3.1-70B-Instruct at
# TP=2 on a single replica (2xH100 80GB). Closed-loop bench at concurrency
# cap 180 (no Poisson rate cap; --rate 0).
#
# Cells (paper §4.3, mirroring W1 cell C and W2 cell B at 70B scale):
#   C    chat:   G=88,  prefix=1500, decode=fixed 128                     N=1000  admission-bound
#   B    RAG:    G=14,  prefix=4096, decode=mix(128,512,1024,2048,4096)   N=500   decode-bound
#
# Policies (paper Table 2 labels):
#   lpm_lru       SGLang LPM + LRU                                  baseline
#   clpm_gm_dl_pe cLPM + group_major + dynamic_lane + cluster evict full PEEK
#
# Usage:
#   bash benchmarks/w3/run_w3_sglang.sh                                # full matrix
#   CELLS=C POLICIES=lpm_lru SEEDS=42 bash benchmarks/w3/run_w3_sglang.sh    # smoke
#   CELLS=B POLICIES="lpm_lru clpm_gm_dl_pe" SEEDS=42 bash benchmarks/w3/run_w3_sglang.sh
#
# Prerequisites:
#   1. sglang 0.5.9 installed in your active Python environment.
#   2. peek built and importable (`maturin develop --release`).
#   3. PYTHONPATH includes scripts/peek_sitecustomize/ so the patch hook fires.
#   4. 2 x H100 80GB available (TP=2 single replica).

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.88}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w3/results_sglang}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"

TP="${TP:-2}"

CELLS="${CELLS:-C B}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-lpm_lru clpm_gm_dl_pe}"
RATE_LABEL="${RATE_LABEL:-heavy}"
CONCURRENCY="${CONCURRENCY:-180}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell parameters ---------------------------
# Match the W3 vLLM driver exactly so per-cell deltas can be compared.

declare -A CELL_GROUPS CELL_PREFIX CELL_DECODE_MIX CELL_MAX_TOKENS CELL_N CELL_WARMUP

# Cell C -- chat-like, admission-bound (mirrors W1 cell C at 70B).
CELL_GROUPS[C]=88
CELL_PREFIX[C]=1500
CELL_DECODE_MIX[C]=""        # empty -> use --max-tokens (fixed)
CELL_MAX_TOKENS[C]=128
CELL_N[C]=1000
CELL_WARMUP[C]=200

# Cell B -- RAG-like, decode-bound (mirrors W2 cell B at 70B).
CELL_GROUPS[B]=14
CELL_PREFIX[B]=4096
CELL_DECODE_MIX[B]="10:128,25:512,30:1024,25:2048,10:4096"
CELL_MAX_TOKENS[B]=4096      # ceiling; the mix dominates
CELL_N[B]=500
CELL_WARMUP[B]=100

ZIPF_ALPHA="${ZIPF_ALPHA:-1.0}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    lpm_lru)         echo "" ;;
    fcfs_lru)        echo "" ;;
    lpm_pe)          echo "PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm)            echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1" ;;
    clpm_gm)         echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1" ;;
    clpm_gm_dl)      echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_pe)      echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm_gm_dl_pe)   echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

policy_sched() {
  case "$1" in
    fcfs_lru) echo "fcfs" ;;
    *)        echo "lpm"  ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
  ps -ef | awk '/sglang::/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/torch._inductor.compile_worker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/multiprocessing.resource_tracker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 5
}

launch_server() {
  local policy="$1" slog="$2"
  local env_pref; env_pref="$(policy_env "$policy")"
  local sched;    sched="$(policy_sched "$policy")"

  echo "[w3-sglang] launching $policy (sched=$sched tp=$TP env='$env_pref') -> $slog"
  env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME" \
      PYTHONPATH="$SITECUSTOMIZE_DIR:${PYTHONPATH:-}" \
      $env_pref "$PY" -m sglang.launch_server \
      --model "$MODEL" \
      --tp "$TP" \
      --mem-fraction-static "$MEM_FRAC" \
      --schedule-policy "$sched" \
      --enable-cache-report \
      --host 127.0.0.1 --port "$PORT" \
      --log-level warning \
      >>"$slog" 2>&1 &

  local t0; t0="$(date +%s)"
  while true; do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[w3-sglang]   ready after $(( $(date +%s) - t0 ))s"
      sleep 3
      return 0
    fi
    if (( $(date +%s) - t0 > SERVER_READY_TIMEOUT_S )); then
      echo "[w3-sglang]   server failed to start in ${SERVER_READY_TIMEOUT_S}s"
      return 1
    fi
    sleep 5
  done
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null 2>&1 \
    && echo "[w3-sglang]   /flush_cache ok" || echo "[w3-sglang]   /flush_cache failed (ignored)"
  sleep 2
}

# ------------------------------ bench -------------------------------------

run_bench() {
  local cell="$1" policy="$2" seed="$3"
  local outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}/rate_${RATE_LABEL}"
  mkdir -p "$outdir"
  local out="$outdir/${policy}.json"
  local blog="$outdir/_run_${policy}.log"
  if [[ -f "$out" ]]; then
    echo "[w3-sglang]   skip: $out already exists"
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

  echo "[w3-sglang]   benching $policy seed=$seed cell=$cell G=$groups prefix=$prefix concurrency=$CONCURRENCY decode='${mix:-fixed=$max_tok}' -> $out"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$n" \
    --groups "$groups" --prefix-tokens "$prefix" \
    --dataset auto \
    --distribution zipf --zipf-alpha "$ZIPF_ALPHA" \
    "${decode_args[@]}" \
    --rate 0 --concurrency "$CONCURRENCY" \
    --seed "$seed" --warmup-reqs "$warm" \
    --ttft-slo-ms "$TTFT_SLO" --tpot-slo-ms "$TPOT_SLO" --e2e-slo-ms "$E2E_SLO" \
    --label "${policy}_cell${cell}_seed${seed}_${RATE_LABEL}" \
    --output "$out" \
    --save-per-request \
    > "$blog" 2>&1
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "[w3-sglang]   BENCH FAILED rc=$rc, tail of log:"; tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w3-sglang]   wrote $out"
}

# ------------------------------ main loop ---------------------------------

echo "[w3-sglang] === START === $(date)"
echo "[w3-sglang] tp=$TP cells=$CELLS seeds=$SEEDS policies=$POLICIES"
echo "[w3-sglang] results -> $RESULTS_DIR"

for policy in $POLICIES; do
  echo
  echo "[w3-sglang] ##### policy: $policy #####"
  kill_server
  launch_server "$policy" "$RESULTS_DIR/_server_${policy}.log" || exit 1
  for cell in $CELLS; do
    for seed in $SEEDS; do
      flush_cache
      run_bench "$cell" "$policy" "$seed" || true
    done
  done
done

kill_server
echo
echo "[w3-sglang] === DONE === $(date)"
