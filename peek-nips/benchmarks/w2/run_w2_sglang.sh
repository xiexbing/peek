#!/bin/bash
# W3 benchmark driver — Long-document RAG (decode-dominant co-design).
#
# Adapted from benchmarks/w1/run_w1_sglang.sh with W2-validated additions:
#   - Prometheus /metrics snapshot per cell (for hit-rate computation)
#   - fcfs_lru alias for FCFS (= W1's fcfs_lru)
#   - Per-policy server boot, sequential cells reuse server
#   - Decode-mix support via bench_shared_prompts --decode-mix
#   - Fixed-decode support for decode-length sensitivity cells D0-D4
#
# Cells
#   A     G=40  prefix=4096   N=500 warmup=100  KV~3×  oversub  (short-RAG ref)
#   B     G=40  prefix=8192   N=500 warmup=100  KV~7×  oversub  PRIMARY (mainstream RAG)
#   C     G=40  prefix=16384  N=500 warmup=100  KV~14× oversub  (long-doc RAG)
#   D0    G=40  prefix=8192   fixed_decode=128  decode-length sensitivity
#   D1    G=40  prefix=8192   fixed_decode=512
#   D2    G=40  prefix=8192   fixed_decode=1024
#   D3    G=40  prefix=8192   fixed_decode=2048
#   D4    G=40  prefix=8192   fixed_decode=4096
#
# Policies (W1 8-policy lattice; fcfs_lru alias = fcfs_lru = FCFS)
#   lpm_lru   stock SGLang LPM   + LRU                  baseline
#   fcfs_lru   stock SGLang FCFS  + LRU                  scheduling-axis baseline
#   lpm_pe   stock LPM          + peek_evict cluster   eviction-only ablation
#   clpm   peek_flpm          + LRU                  scheduling stage 1
#   clpm_gm   peek_flpm + group_major + LRU             scheduling stage 2
#   clpm_gm_dl   peek_flpm + group_major + dynamic_lane + LRU   scheduling stage 3
#   clpm_gm_pe   peek_flpm + group_major + peek_evict      co-design (primary claim)
#   clpm_gm_dl_pe   peek_flpm + group_major + dynamic_lane + peek_evict   co-design + fairness
#
# Usage
#   bash run_w3_sglang.sh                                   # full matrix
#   CELLS=B POLICIES=lpm_lru SEEDS=42 RATES=moderate bash run_w3_sglang.sh   # phase 0 probe
#   CELLS=B POLICIES="lpm_lru fcfs_lru lpm_pe clpm clpm_gm clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe" SEEDS=42 bash run_w3_sglang.sh  # phase 1a
#   CELLS="D0 D1 D2 D3 D4" POLICIES="lpm_lru clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe" SEEDS=42 RATES=moderate bash run_w3_sglang.sh  # phase 2

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MEM_FRAC="${MEM_FRAC:-0.88}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w3/results}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"

CELLS="${CELLS:-A B C D0 D1 D2 D3 D4}"
RATES="${RATES:-moderate heavy}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-lpm_lru fcfs_lru lpm_pe clpm clpm_gm clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe}"

# Canonical decode mix per W3 README (mean ≈ 1460 tokens, max 4096)
DECODE_MIX="${DECODE_MIX:-10:128, 25:512, 30:1024, 25:2048, 10:4096}"
# Cell-D fixed decode lengths handled in cell_args()

# Workload knobs (Zipf, concurrency, SLOs)
ZIPF_ALPHA="${ZIPF_ALPHA:-1.0}"
CONCURRENCY="${CONCURRENCY:-256}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-180000}"

DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
FULL_RESTART="${FULL_RESTART:-0}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell params -------------------------------

declare -A CELL_G CELL_PREFIX CELL_N CELL_WARMUP CELL_FIXED_DECODE
# Primary cells (canonical decode mix)
CELL_G[A]=40 ;  CELL_PREFIX[A]=4096  ; CELL_N[A]=500 ; CELL_WARMUP[A]=100 ; CELL_FIXED_DECODE[A]=0
CELL_G[B]=40 ;  CELL_PREFIX[B]=8192  ; CELL_N[B]=1000 ; CELL_WARMUP[B]=100 ; CELL_FIXED_DECODE[B]=0
CELL_G[C]=40 ;  CELL_PREFIX[C]=16384 ; CELL_N[C]=500 ; CELL_WARMUP[C]=100 ; CELL_FIXED_DECODE[C]=0
# Decode-length sensitivity cells (G=40 prefix=8192, fixed decode)
CELL_G[D0]=40 ; CELL_PREFIX[D0]=8192 ; CELL_N[D0]=500 ; CELL_WARMUP[D0]=100 ; CELL_FIXED_DECODE[D0]=128
CELL_G[D1]=40 ; CELL_PREFIX[D1]=8192 ; CELL_N[D1]=500 ; CELL_WARMUP[D1]=100 ; CELL_FIXED_DECODE[D1]=512
CELL_G[D2]=40 ; CELL_PREFIX[D2]=8192 ; CELL_N[D2]=500 ; CELL_WARMUP[D2]=100 ; CELL_FIXED_DECODE[D2]=1024
CELL_G[D3]=40 ; CELL_PREFIX[D3]=8192 ; CELL_N[D3]=500 ; CELL_WARMUP[D3]=100 ; CELL_FIXED_DECODE[D3]=2048
CELL_G[D4]=40 ; CELL_PREFIX[D4]=8192 ; CELL_N[D4]=500 ; CELL_WARMUP[D4]=100 ; CELL_FIXED_DECODE[D4]=4096

cell_rate() {
  # Planning rates per W3 README; recalibrate from r_sat probe if needed.
  case "$1-$2" in
    A-moderate)  echo 3   ;;
    A-heavy)     echo 5   ;;
    B-moderate)  echo 0.40 ;;
    B-heavy)     echo 0.45 ;;
    C-moderate)  echo 0.8 ;;
    C-heavy)     echo 1.5 ;;
    D0-moderate) echo 5   ;;
    D0-heavy)    echo 10  ;;
    D1-moderate) echo 2.5 ;;
    D1-heavy)    echo 5   ;;
    D2-moderate) echo 1.5 ;;
    D2-heavy)    echo 3   ;;
    D3-moderate) echo 0.8 ;;
    D3-heavy)    echo 1.5 ;;
    D4-moderate) echo 0.4 ;;
    D4-heavy)    echo 0.8 ;;
    *) echo "ERR"; return 1 ;;
  esac
}

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    lpm_lru) echo "" ;;
    fcfs_lru) echo "" ;;
    fcfs_lru) echo "" ;;  # alias for fcfs_lru (FCFS)
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
    fcfs_lru|fcfs_lru) echo "fcfs" ;;
    *)     echo "lpm"  ;;
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
  if [[ "${TP:-1}" -gt 1 ]]; then
    extra_args+=(--tp "${TP}")
  fi

  echo "[w2] launching $policy (sched=$sched tp=${TP:-1} env='$env_pref') -> $slog"
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
      echo "[w2]   ready after $((i*3))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w2]   server died; tail of log:"; tail -n 20 "$slog" || true
      return 1
    fi
    if (( i % 20 == 0 )); then
      local nvmem
      nvmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
      echo "[w2]   ...still loading at $((i*3))s, gpu_mem=${nvmem}MiB"
    fi
  done
  echo "[w2]   FAILED to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 1
}

snapshot_metrics() {
  # $1=output file path
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^sglang:(prompt_tokens_total|cached_tokens_total|generation_tokens_total|num_retractions_sum|num_retractions_count|realtime_tokens_total)[ {]' \
    > "$1" 2>/dev/null || true
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local cell="$1" rate_label="$2" policy="$3" seed="$4" out="$5"

  local G="${CELL_G[$cell]}" prefix="${CELL_PREFIX[$cell]}"
  local N="${CELL_N[$cell]}" warmup="${CELL_WARMUP[$cell]}"
  local fixed_dec="${CELL_FIXED_DECODE[$cell]}"
  local rate; rate="$(cell_rate "$cell" "$rate_label")"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"

  flush_cache

  local blog="$outdir/_run_${policy}.log"
  local label; label="$(basename "$out" .json)"
  echo "[w2]   benching $policy (cell=$cell G=$G prefix=$prefix decode=$([[ "$fixed_dec" == "0" ]] && echo "MIX" || echo "$fixed_dec") N=$N rate=$rate seed=$seed)"

  snapshot_metrics "$outdir/_metrics_pre_${policy}.prom"

  local decode_args=()
  if [[ "$fixed_dec" == "0" ]]; then
    decode_args+=(--decode-mix "$DECODE_MIX")
  else
    decode_args+=(--max-tokens "$fixed_dec")
  fi

  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" \
    --groups "$G" \
    --prefix-tokens "$prefix" \
    "${decode_args[@]}" \
    --rate "$rate" \
    --concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --warmup-reqs "$warmup" \
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

  snapshot_metrics "$outdir/_metrics_post_${policy}.prom"

  if [[ $rc -ne 0 ]]; then
    echo "[w2]   BENCH FAILED rc=$rc, tail of log:"
    tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w2]   wrote $out"
  return 0
}

# ------------------------------ main loop ---------------------------------
# Policy-major loop: one server per policy, all (seed × cell × rate) benches
# of that policy run back-to-back with /flush_cache between.

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

echo "[w2] plan: $total_runs runs across ${#policy_order[@]} distinct policies"
echo "[w2] cells=$CELLS  rates=$RATES  seeds=$SEEDS  policies=$POLICIES"
echo "[w2] decode_mix=$DECODE_MIX  zipf_alpha=$ZIPF_ALPHA"
echo "[w2] results -> $RESULTS_DIR"
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
      echo "[w2]   skip: $out already exists"
      continue
    fi

    if [[ "$server_up" == 0 || "$FULL_RESTART" == "1" ]]; then
      slog="$slog_base"
      if ! launch_server "$policy" "$slog"; then
        echo "[w2]   LAUNCH FAILED for $policy — skipping remaining benches"
        server_up=0
        break
      fi
      server_up=1
    fi

    if ! run_one "$cell" "$rate_label" "$policy" "$seed" "$out"; then
      echo "[w2]   BENCH FAILED — restarting server"
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
echo "[w2] done."
