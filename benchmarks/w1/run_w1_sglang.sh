#!/bin/bash
# W1 benchmark driver (sglang-based policies) — "Shared-prompt co-design
# across oversubscription".
#
# This driver handles everything served via an sglang process: stock SGLang
# baselines (lpm_lru, fcfs_lru) and all peek policies (lpm_pe–clpm_gm_dl_pe). External baselines that
# use a different inference engine have their own drivers:
#   - fcfs_apc_lru vLLM → run_w1_vllm.sh
#
# Matrix
#   Cells
#     A  G=100 prefix=1024 N=1000 warmup=200   KV footprint 102K    → ~2×  oversub
#     B  G=200 prefix=1024 N=2000 warmup=400   KV footprint 205K    → ~4×  oversub
#     C  G=100 prefix=4096 N=1000 warmup=200   KV footprint 410K    → ~8×  oversub   [PRIMARY]
#     D  G=200 prefix=4096 N=2000 warmup=400   KV footprint 820K    → ~16× oversub (extreme)
#
#   Rates (req/s) per cell (planning estimates — recalibrate from r_sat probe)
#     A  moderate=8  heavy=16
#     B  moderate=6  heavy=12
#     C  moderate=3  heavy=6
#     D  moderate=2  heavy=4
#
#   Seeds   42, 142, 242
#
#   Policies
#     lpm_lru   SGLang LPM   + LRU                 — baseline
#     fcfs_lru   SGLang FCFS  + LRU                 — scheduling-axis baseline
#     lpm_pe   SGLang LPM   + peek_evict          — eviction-only ablation
#     clpm   peek_flpm (per-req, 0.7) + LRU     — scheduling stage 1
#     clpm_gm   peek_flpm_gm + LRU                 — scheduling stage 2 (+ grouping)
#     clpm_gm_dl   peek_flpm_gm_dyn + LRU             — scheduling stage 3 (+ dynamic lane)
#     clpm_gm_pe   peek_flpm_gm + peek_evict          — co-design (primary claim)
#     clpm_gm_dl_pe   peek_flpm_gm_dyn + peek_evict      — co-design + fairness
#
#   Layout
#     Cell C: all 8 policies (full ablation + baselines)
#     Cells A/B/D: 3-policy subset (lpm_lru, clpm_gm, clpm_gm_pe) for the oversubscription-sensitivity trend
#
#   Budget ~20 GPU-hours on 1×H100 (single-seed) / ~60 hrs for 3 seeds.
#
# Usage
#   bash benchmarks/w1/run_w1_sglang.sh                               # full matrix, 3 seeds
#   CELLS="C"      bash benchmarks/w1/run_w1_sglang.sh                # just primary cell
#   CELLS="C" POLICIES_FULL="lpm_lru clpm_gm_pe" SEEDS="42" bash run_w1_sglang.sh  # smoke test

set -euo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MEM_FRAC="${MEM_FRAC:-0.88}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w1/results}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"

CELLS="${CELLS:-A B C D}"
RATES="${RATES:-moderate heavy}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES_CORE="${POLICIES_CORE:-lpm_lru lpm_pe clpm_gm clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe}"
POLICIES_FULL="${POLICIES_FULL:-lpm_lru fcfs_lru lpm_pe clpm clpm_gm clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe}"
PRIMARY_CELL="${PRIMARY_CELL:-C}"

# Zipf / decode / concurrency fixed across W1
ZIPF_ALPHA="${ZIPF_ALPHA:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-64}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

# CUDA graphs: ENABLED by default for W1 (production-realistic configuration).
# Set DISABLE_CUDA_GRAPH=1 to revert to graphs-off for debugging peek hooks.
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell params -------------------------------

declare -A CELL_G CELL_PREFIX CELL_N CELL_WARMUP CELL_OVERSUB
CELL_G[A]=100;   CELL_PREFIX[A]=1024;  CELL_N[A]=1000; CELL_WARMUP[A]=200;  CELL_OVERSUB[A]=2
CELL_G[B]=200;   CELL_PREFIX[B]=1024;  CELL_N[B]=2000; CELL_WARMUP[B]=400;  CELL_OVERSUB[B]=4
CELL_G[C]=100;   CELL_PREFIX[C]=4096;  CELL_N[C]=1000; CELL_WARMUP[C]=200;  CELL_OVERSUB[C]=8
CELL_G[D]=200;   CELL_PREFIX[D]=4096;  CELL_N[D]=2000; CELL_WARMUP[D]=400;  CELL_OVERSUB[D]=16

cell_rate() {
  # $1=cell A/B/C/D, $2=label moderate/heavy
  case "$1-$2" in
    A-moderate) echo 8  ;;
    A-heavy)    echo 16 ;;
    B-moderate) echo 6  ;;
    B-heavy)    echo 12 ;;
    C-moderate) echo 3  ;;
    C-heavy)    echo 6  ;;
    D-moderate) echo 2  ;;
    D-heavy)    echo 4  ;;
    *) echo "ERR"; return 1 ;;
  esac
}

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    lpm_lru) echo "" ;;
    fcfs_lru) echo "" ;;  # sglang --schedule-policy fcfs handled via policy_sched
    lpm_pe) echo "PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1" ;;
    clpm_gm) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1" ;;
    clpm_gm_dl) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
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
  # $1=policy; writes to $2 (server log path)
  local policy="$1" slog="$2"
  kill_server
  local env_pref; env_pref="$(policy_env "$policy")"
  local sched;    sched="$(policy_sched "$policy")"

  local extra_args=()
  if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    extra_args+=(--disable-cuda-graph)
  fi

  if [[ "${TP:-1}" -gt 1 ]]; then
    extra_args+=(--tp "${TP}")
  fi

  echo "[w1] launching $policy (sched=$sched tp=${TP:-1} cuda_graph=$([[ "$DISABLE_CUDA_GRAPH" == "1" ]] && echo off || echo on) env='$env_pref') → $slog"
  env \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME" \
    $env_pref \
    "$PY" -m sglang.launch_server \
      --model "$MODEL" \
      --mem-fraction-static "$MEM_FRAC" \
      --schedule-policy "$sched" \
      --enable-cache-report \
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
      echo "[w1]   ready after $((i*3))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w1]   server died; tail of log:"; tail -n 20 "$slog" || true
      return 1
    fi
    if (( i % 20 == 0 )); then
      local nvmem
      nvmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
      echo "[w1]   ...still loading at $((i*3))s, gpu_mem=${nvmem}MiB"
    fi
  done
  echo "[w1]   failed to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 1
}

# ------------------------------ one bench run -----------------------------

run_one() {
  # $1=cell $2=rate_label $3=policy $4=seed $5=outpath
  local cell="$1" rate_label="$2" policy="$3" seed="$4" out="$5"

  local G="${CELL_G[$cell]}" prefix="${CELL_PREFIX[$cell]}"
  local N="${CELL_N[$cell]}" warmup="${CELL_WARMUP[$cell]}"
  local rate; rate="$(cell_rate "$cell" "$rate_label")"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"

  flush_cache

  local blog="$outdir/_run_${policy}.log"
  local label; label="$(basename "$out" .json)"
  echo "[w1]   benching $policy (cell=$cell G=$G prefix=$prefix N=$N rate=$rate seed=$seed)"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" \
    --groups "$G" \
    --prefix-tokens "$prefix" \
    --max-tokens "$MAX_TOKENS" \
    --rate "$rate" \
    --concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --warmup-reqs "$warmup" \
    --ttft-slo-ms "$TTFT_SLO" \
    --tpot-slo-ms "$TPOT_SLO" \
    --e2e-slo-ms "$E2E_SLO" \
    --dataset auto \
    --distribution zipf \
    --zipf-alpha "$ZIPF_ALPHA" \
    --label "$label" \
    --output "$out" \
    --save-per-request \
    > "$blog" 2>&1

  echo "[w1]   wrote $out"
}

# ------------------------------ main loop ---------------------------------
#
# Policy-major loop: one sglang launch per unique policy, all of that
# policy's (seed × cell × rate) benches run back-to-back on the same server
# with `/flush_cache` between benches. Saves ~12 GPU-hrs vs restart-per-run
# for the full W1 matrix (156 runs → 8 launches instead of 156).
#
# State-hygiene note: sglang's `/flush_cache` clears the radix cache, but
# peek's module-level globals (e.g. _dyn_lane_state EMA in lpm_integration,
# _phase_timings in patch_hook) persist inside the server process across
# benches of the same policy. For most metrics this is harmless — the peek
# state keys on `id(tree)` and sglang recreates the tree on flush. If a
# future claim needs cold-start equivalence between benches of the same
# policy, set FULL_RESTART=1 to get one launch per run (paper-ironclad,
# ~156 launches, +12 GPU-hrs).
FULL_RESTART="${FULL_RESTART:-0}"

# Build per-policy plan maps: policy -> list of "seed|cell|rate_label" entries.
declare -A policy_plan   # policy -> newline-separated list
declare -a policy_order  # first-seen policy order (bash assoc arrays don't preserve)
for seed in $SEEDS; do
  for cell in $CELLS; do
    local_policies="$POLICIES_CORE"
    if [[ "$cell" == "$PRIMARY_CELL" ]]; then local_policies="$POLICIES_FULL"; fi
    for rate_label in $RATES; do
      for policy in $local_policies; do
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
launches_expected=${#policy_order[@]}
if [[ "$FULL_RESTART" == "1" ]]; then
  launches_expected=$total_runs
fi

echo "[w1] plan: $total_runs runs across ${#policy_order[@]} distinct policies"
echo "[w1] cells=$CELLS  rates=$RATES  seeds=$SEEDS"
echo "[w1] primary-cell=$PRIMARY_CELL  policies_full=$POLICIES_FULL"
echo "[w1]                        policies_core=$POLICIES_CORE"
echo "[w1] restart mode: $([[ "$FULL_RESTART" == "1" ]] && echo 'FULL_RESTART=1 (one launch per run)' || echo 'policy-major (one launch per policy)')"
echo "[w1] expected launches: $launches_expected"
echo "[w1] results → $RESULTS_DIR"
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
      echo "[w1]   skip: $out already exists"
      continue
    fi

    # Launch server if not up (or always, under FULL_RESTART).
    if [[ "$server_up" == 0 || "$FULL_RESTART" == "1" ]]; then
      slog="$slog_base"
      if ! launch_server "$policy" "$slog"; then
        echo "[w1]   LAUNCH FAILED for $policy — skipping remaining benches of this policy"
        server_up=0
        break
      fi
      server_up=1
    fi

    if ! run_one "$cell" "$rate_label" "$policy" "$seed" "$out"; then
      echo "[w1]   BENCH FAILED for $policy (seed=$seed cell=$cell rate=$rate_label) — server log tail:"
      tail -n 30 "$slog_base" || true
      # Treat bench failure as server-possibly-bad; restart on next bench.
      kill_server
      server_up=0
    fi

    # Under FULL_RESTART, kill after each bench so next iteration relaunches.
    if [[ "$FULL_RESTART" == "1" ]]; then
      kill_server
      server_up=0
    fi
  done <<< "$entries"

  # Kill at end of policy regardless of mode.
  if [[ "$server_up" == 1 ]]; then
    kill_server
  fi
done

echo
echo "[w1] done. aggregate with: $PY $REPO_ROOT/benchmarks/w1/aggregate.py"
