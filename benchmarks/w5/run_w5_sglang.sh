#!/bin/bash
# W5 benchmark driver — singleton chat workload (LMSYS-Chat-1M, Gemma 2 27B-it).
# No-regression test for peek vs stock LPM.
#
# Cells (sharing structure: ~0% by construction)
#   C_short  prompts mean ~512 tok   (lognormal-distributed natural chat)
#   C_long   prompts mean ~2048 tok  (longer-form chat / instruction tasks)
#
# Policies
#   lpm_lru   stock LPM + LRU              baseline (LPM scheduling axis)
#   fcfs_lru   stock FCFS + LRU             baseline (FCFS scheduling axis)
#   clpm_gm_dl_pe   peek_flpm + group_major + dynamic_lane + peek_evict (demand_cluster)
#
# Rates: per-cell moderate + heavy, set by the W5 probe (see /tmp/w5_probe_then_run.sh)
#
# Usage
#   bash run_w5_sglang.sh                                     # full matrix (24 runs)
#   CELLS=C_short POLICIES=lpm_lru SEEDS=42 RATES=moderate bash run_w5_sglang.sh   # smoke

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:-google/gemma-2-27b-it}"
MEM_FRAC="${MEM_FRAC:-0.85}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w5/results}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_lmsys_singleton.py}"

CELLS="${CELLS:-C_short C_long}"
RATES="${RATES:-moderate heavy}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-lpm_lru fcfs_lru clpm_gm_dl_pe}"

CONCURRENCY="${CONCURRENCY:-256}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
FULL_RESTART="${FULL_RESTART:-0}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell params -------------------------------

declare -A CELL_PROMPT_MIN CELL_PROMPT_MAX CELL_DECODE_MIN CELL_DECODE_MAX CELL_N CELL_WARMUP
CELL_PROMPT_MIN[C_short]=32;   CELL_PROMPT_MAX[C_short]=1024;  CELL_DECODE_MIN[C_short]=64;  CELL_DECODE_MAX[C_short]=256;  CELL_N[C_short]=1500; CELL_WARMUP[C_short]=100
CELL_PROMPT_MIN[C_long]=512;   CELL_PROMPT_MAX[C_long]=4096;   CELL_DECODE_MIN[C_long]=64;   CELL_DECODE_MAX[C_long]=256;   CELL_N[C_long]=1500;  CELL_WARMUP[C_long]=100

cell_rate() {
  # Set by the W5 probe; placeholders below get patched in-place.
  case "$1-$2" in
    C_short-moderate)  echo 102.0 ;;
    C_short-heavy)     echo 120.0 ;;
    C_long-moderate)  echo 15.0 ;;
    C_long-heavy)     echo 25.0 ;;
    *) echo "ERR"; return 1 ;;
  esac
}

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    lpm_lru) echo "" ;;
    fcfs_lru) echo "" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    # Pa* variants: replace dynamic_lane with static biglane_share to characterize
    # the fairness/aggressiveness tradeoff. Lower share = more fairness toward
    # non-cluster reqs = smaller tail regression but smaller cache-hit gain.
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

policy_sched() {
  case "$1" in
    fcfs_lru) echo "fcfs" ;;
    *)  echo "lpm"  ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
  sleep 3
}

launch_server() {
  local policy="$1" slog="$2"
  kill_server
  local env_pref; env_pref="$(policy_env "$policy")"
  local sched;    sched="$(policy_sched "$policy")"

  local extra_args=()
  if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    extra_args+=(--disable-cuda-graph)
  fi

  echo "[w5] launching $policy (sched=$sched env='$env_pref') -> $slog"
  env \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME" \
    $env_pref \
    "$PY" -m sglang.launch_server \
      --model "$MODEL" \
      --mem-fraction-static "$MEM_FRAC" \
      --schedule-policy "$sched" \
      --enable-cache-report \
      --enable-metrics \
      --host 127.0.0.1 --port "$PORT" \
      --log-level warning \
      "${extra_args[@]}" \
      >"$slog" 2>&1 &
  local pid=$!
  local iters=$(( SERVER_READY_TIMEOUT_S / 3 ))
  for i in $(seq 1 $iters); do
    sleep 3
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w5]   ready after $((i*3))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w5]   server died; tail of log:"; tail -n 20 "$slog" || true
      return 1
    fi
    if (( i % 20 == 0 )); then
      local nvmem
      nvmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
      echo "[w5]   ...still loading at $((i*3))s, gpu_mem=${nvmem}MiB"
    fi
  done
  echo "[w5]   FAILED to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 1
}

snapshot_metrics() {
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^sglang:(prompt_tokens_total|cached_tokens_total|generation_tokens_total|num_retractions_sum|num_retractions_count|realtime_tokens_total)[ {]' \
    > "$1" 2>/dev/null || true
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local cell="$1" rate_label="$2" policy="$3" seed="$4" out="$5"

  local pmin="${CELL_PROMPT_MIN[$cell]}" pmax="${CELL_PROMPT_MAX[$cell]}"
  local dmin="${CELL_DECODE_MIN[$cell]}" dmax="${CELL_DECODE_MAX[$cell]}"
  local N="${CELL_N[$cell]}" warmup="${CELL_WARMUP[$cell]}"
  local rate; rate="$(cell_rate "$cell" "$rate_label")"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"

  flush_cache

  local blog="$outdir/_run_${policy}.log"
  local label; label="$(basename "$out" .json)"
  echo "[w5]   benching $policy (cell=$cell prompt=[$pmin,$pmax] decode=[$dmin,$dmax] N=$N rate=$rate seed=$seed)"

  snapshot_metrics "$outdir/_metrics_pre_${policy}.prom"

  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" \
    --rate "$rate" \
    --concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --warmup-reqs "$warmup" \
    --ttft-slo-ms "$TTFT_SLO" \
    --tpot-slo-ms "$TPOT_SLO" \
    --e2e-slo-ms "$E2E_SLO" \
    --prompt-len-min "$pmin" --prompt-len-max "$pmax" \
    --max-tokens-min "$dmin" --max-tokens-max "$dmax" \
    --label "$label" \
    --output "$out" \
    --save-per-request \
    > "$blog" 2>&1
  local rc=$?

  snapshot_metrics "$outdir/_metrics_post_${policy}.prom"

  if [[ $rc -ne 0 ]]; then
    echo "[w5]   BENCH FAILED rc=$rc, tail of log:"
    tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w5]   wrote $out"
  return 0
}

# ------------------------------ main loop ---------------------------------
# Policy-major: one server per policy, all (seed × cell × rate) benches
# back-to-back with /flush_cache between.

declare -A policy_plan
declare -a policy_order
for seed in $SEEDS; do
  for cell in $CELLS; do
    for rate_label in $RATES; do
      for policy in $POLICIES; do
        if [[ -z "${policy_plan[$policy]:-}" ]]; then
          policy_order+=("$policy")
          policy_plan[$policy]="$seed|$cell|$rate_label"
        else
          policy_plan[$policy]+=$'\n'"$seed|$cell|$rate_label"
        fi
      done
    done
  done
done

total_runs=0
for p in "${policy_order[@]}"; do
  n=$(echo -n "${policy_plan[$p]}" | awk 'END{print NR}')
  total_runs=$((total_runs + n))
done

echo "[w5] plan: $total_runs runs across ${#policy_order[@]} distinct policies"
echo "[w5] cells=$CELLS  rates=$RATES  seeds=$SEEDS  policies=$POLICIES"
echo "[w5] model=$MODEL  results -> $RESULTS_DIR"
echo

idx=0
for policy in "${policy_order[@]}"; do
  entries="${policy_plan[$policy]}"
  echo
  echo "##### policy: $policy (runs: $(echo -n "$entries" | awk 'END{print NR}')) #####"

  server_up=0
  slog_base="$RESULTS_DIR/_server_${policy}.log"

  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    idx=$((idx+1))
    IFS='|' read -r seed cell rate_label <<< "$entry"
    outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}/rate_${rate_label}"
    out="$outdir/${policy}.json"
    mkdir -p "$outdir"

    echo
    echo "----- [$idx/$total_runs] policy=$policy seed=$seed cell=$cell rate=$rate_label -----"

    if [[ -f "$out" ]]; then
      echo "[w5]   skip: $out already exists"
      continue
    fi

    if [[ "$server_up" == 0 || "$FULL_RESTART" == "1" ]]; then
      slog="$slog_base"
      if ! launch_server "$policy" "$slog"; then
        echo "[w5]   LAUNCH FAILED for $policy — skipping remaining benches"
        server_up=0
        break
      fi
      server_up=1
    fi

    if ! run_one "$cell" "$rate_label" "$policy" "$seed" "$out"; then
      echo "[w5]   BENCH FAILED — restarting server"
      kill_server
      server_up=0
    fi

    if [[ "$FULL_RESTART" == "1" ]]; then
      kill_server
      server_up=0
    fi
  done <<< "$entries"

  if [[ "$server_up" == 1 ]]; then
    kill_server
  fi
done

echo
echo "[w5] done."
