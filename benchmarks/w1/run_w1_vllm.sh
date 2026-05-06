#!/bin/bash
# W1 benchmark driver (vllm-based policies). Mirror of run_w1_sglang.sh
# adapted to vllm v1: peek hooks are injected into vllm's spawn child via
# a sitecustomize.py shim on PYTHONPATH (vllm v1 spawns EngineCore as a
# fresh Python process — parent monkey-patches don't inherit).
#
# Matrix is identical to the sglang driver — see run_w1_sglang.sh for the
# G/prefix/N/warmup/oversub table and the rate plan.
#
# Policies map to peek env flags:
#   fcfs_lru   vllm vanilla (FCFS, stock LRU eviction)        — baseline
#   fcfs_apc_lru   not applicable (vllm has no LPM scheduler analogue) — skipped
#   fcfs_apc_pe   PEEK_ONLINE_EVICTION=1                                — eviction-only ablation
#   clpm   PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1                                    — sched stage 1
#   clpm_gm   PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1            — sched stage 2
#   clpm_gm_dl   PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 GROUP_MAJOR=1 DYNAMIC_LANE=1       — sched stage 3
#   clpm_gm_pe   FLPM + GROUP_MAJOR + EVICTION                  — co-design (primary claim)
#   clpm_gm_dl_pe   FLPM + GROUP_MAJOR + DYNAMIC_LANE + EVICTION   — co-design + fairness
#
# Note on fcfs_apc_pe vs sglang: sglang's fcfs_apc_pe used PEEK_ONLINE_EVICTION_MODE=cluster.
# vllm's eviction patch supports only the `plain` mode (the recency/cluster/
# decay variants need per-node access time + ancestor walks specific to
# sglang's RadixCache). PEEK_ONLINE_EVICTION_MODE is silently ignored on vllm.
#
# Usage
#   bash benchmarks/w1/run_w1_vllm.sh                                # full matrix
#   CELLS="C" bash benchmarks/w1/run_w1_vllm.sh                      # primary cell
#   CELLS="C" POLICIES_FULL="fcfs_lru clpm_gm_pe" SEEDS="42" bash run_w1_vllm.sh   # smoke

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w1/results_vllm}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
# Wait progressively for either /v1/models 200 or a log-marker. Polling
# /health alone is fragile across vllm versions.
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"

CELLS="${CELLS:-A B C D}"
RATES="${RATES:-moderate heavy}"
SEEDS="${SEEDS:-42}"
# Policy order: best/worst extremes first so the user sees the upper bound
# (clpm_gm_dl_pe) and lower bound (fcfs_lru) early; intermediates fill in. fcfs_apc_lru (APC+LRU stock
# baseline) anchors the delta.
POLICIES_CORE="${POLICIES_CORE:-fcfs_apc_lru clpm_gm_dl_pe clpm_gm_pe clpm_gm fcfs_apc_pe}"
POLICIES_FULL="${POLICIES_FULL:-fcfs_apc_lru clpm_gm_dl_pe clpm_gm_pe clpm_gm_dl clpm_gm clpm fcfs_apc_pe}"
PRIMARY_CELL="${PRIMARY_CELL:-C}"

ZIPF_ALPHA="${ZIPF_ALPHA:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

# vllm-side: max-model-len caps the KV reservation per req; set generously
# for the longest cell (D: prefix=4096 + decode=128 + slack).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell params -------------------------------

declare -A CELL_G CELL_PREFIX CELL_N CELL_WARMUP CELL_OVERSUB
CELL_G[A]=100;   CELL_PREFIX[A]=1024;  CELL_N[A]=1000; CELL_WARMUP[A]=200;  CELL_OVERSUB[A]=2
CELL_G[B]=200;   CELL_PREFIX[B]=1024;  CELL_N[B]=2000; CELL_WARMUP[B]=400;  CELL_OVERSUB[B]=4
CELL_G[C]=100;   CELL_PREFIX[C]=4096;  CELL_N[C]=1000; CELL_WARMUP[C]=200;  CELL_OVERSUB[C]=8
CELL_G[D]=200;   CELL_PREFIX[D]=4096;  CELL_N[D]=2000; CELL_WARMUP[D]=400;  CELL_OVERSUB[D]=16

cell_rate() {
  ## CALIBRATED FROM PROBE ##  (2026-04-25; queue-depth target: moderate=60-100,
  ## heavy=150-200; rates set well above engine saturation so the bench's
  ## --concurrency cap is the binding constraint and queue depth is determined
  ## by cap, not λ. Probes in results_vllm/_calibration/.)
  case "$1-$2" in
    A-moderate) echo 20 ;;
    A-heavy)    echo 30 ;;
    B-moderate) echo 20 ;;
    B-heavy)    echo 30 ;;
    C-moderate) echo 5  ;;
    C-heavy)    echo 10 ;;
    D-moderate) echo 8  ;;
    D-heavy)    echo 15 ;;
    *) echo "ERR"; return 1 ;;
  esac
}

# Per-(rate-band) bench concurrency cap. Sets the steady-state pending-queue
# depth at vLLM (Waiting = cap - Running): moderate→~60-100, heavy→~150-200.
# Uniform across cells; differing μ between cells is absorbed by the cell_rate
# table above.
cell_concurrency() {
  case "$2" in
    moderate) echo 100 ;;
    heavy)    echo 210 ;;
    *) echo "ERR"; return 1 ;;
  esac
}

policy_env() {
  case "$1" in
    fcfs_lru)  echo "" ;;                                                # APC off
    fcfs_apc_lru)  echo "" ;;                                                # APC on, no peek
    # fcfs_apc_pe/clpm_gm_pe/clpm_gm_dl_pe use cluster mode for parity with sglang's demand_cluster
    # canonical pick. plain/recency/decay variants below are optional
    # ablation policies (not in the default POLICIES_FULL).
    fcfs_apc_pe)  echo "PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1" ;;
    clpm_gm)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1" ;;
    clpm_gm_dl)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_pe)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    clpm_gm_dl_pe)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    # Optional ablation: same as fcfs_apc_pe/clpm_gm_pe but with the other eviction modes.
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

# Per-policy prefix-cache flag. fcfs_lru only is the no-cache control; everything
# else (including fcfs_apc_lru and all peek policies) needs APC enabled.
policy_apc_flag() {
  case "$1" in
    fcfs_lru) echo "--no-enable-prefix-caching" ;;
    *)  echo "--enable-prefix-caching"    ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  pkill -9 -f "vllm.*serve.*--port $PORT" 2>/dev/null || true
  sleep 3
  # vllm v1 spawns VLLM::EngineCore as a child of APIServer via spawn; SIGKILL
  # to the parent does not always reap it, and the orphan pins ~75 GiB of GPU
  # so the next cold-start sees insufficient free memory and fails. Reap
  # explicitly. Also wait briefly for nvidia driver to release.
  local orphans
  orphans=$(pgrep -f "VLLM::EngineCore" || true)
  if [ -n "$orphans" ]; then
    kill -TERM $orphans 2>/dev/null || true
    sleep 5
    orphans=$(pgrep -f "VLLM::EngineCore" || true)
    if [ -n "$orphans" ]; then
      kill -9 $orphans 2>/dev/null || true
      sleep 3
    fi
  fi
}

launch_server() {
  local policy="$1" slog="$2"
  kill_server
  local env_pref; env_pref="$(policy_env "$policy")"
  local apc_flag; apc_flag="$(policy_apc_flag "$policy")"

  echo "[w1] launching $policy (env='$env_pref' apc='$apc_flag') → $slog"
  env \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME" \
    PYTHONPATH="$SITECUSTOMIZE_DIR:${PYTHONPATH:-}" \
    $env_pref \
    "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --enable-prompt-tokens-details \
      "$apc_flag" \
      >"$slog" 2>&1 &
  local pid=$!
  local iters=$(( SERVER_READY_TIMEOUT_S / 3 ))
  for i in $(seq 1 $iters); do
    sleep 3
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w1]   ready after $((i*3))s (via /v1/models)"
      return 0
    fi
    if grep -qE "Application startup complete|Uvicorn running on|Started server process" "$slog" 2>/dev/null; then
      sleep 1
      echo "[w1]   ready after $((i*3))s (log marker)"
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
  # vllm's reset_prefix_cache endpoint requires no in-flight blocks; we
  # call it best-effort and ignore failure. Restart-per-policy is the
  # safer way to get a clean cache when this matters.
  curl -s -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null || true
  sleep 1
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local cell="$1" rate_label="$2" policy="$3" seed="$4" out="$5"
  local G="${CELL_G[$cell]}" prefix="${CELL_PREFIX[$cell]}"
  local N="${CELL_N[$cell]}" warmup="${CELL_WARMUP[$cell]}"
  local rate; rate="$(cell_rate "$cell" "$rate_label")"
  local cap;  cap="$(cell_concurrency "$cell" "$rate_label")"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"
  flush_cache

  local blog="$outdir/_run_${policy}.log"
  local label; label="$(basename "$out" .json)"
  echo "[w1]   benching $policy (cell=$cell G=$G prefix=$prefix N=$N rate=$rate cap=$cap seed=$seed)"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" --groups "$G" --prefix-tokens "$prefix" \
    --max-tokens "$MAX_TOKENS" --rate "$rate" \
    --concurrency "$cap" --seed "$seed" \
    --warmup-reqs "$warmup" \
    --ttft-slo-ms "$TTFT_SLO" --tpot-slo-ms "$TPOT_SLO" --e2e-slo-ms "$E2E_SLO" \
    --dataset auto --distribution zipf --zipf-alpha "$ZIPF_ALPHA" \
    --label "$label" --output "$out" --save-per-request \
    > "$blog" 2>&1
  echo "[w1]   wrote $out"
}

# ------------------------------ main loop ---------------------------------
# Same policy-major loop as the sglang driver: one launch per unique policy,
# all (seed × cell × rate) cells of that policy back-to-back.

FULL_RESTART="${FULL_RESTART:-0}"

declare -A policy_plan
declare -a policy_order
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

echo "[w1] plan: $total_runs runs across ${#policy_order[@]} policies"
echo "[w1] cells=$CELLS rates=$RATES seeds=$SEEDS"
echo "[w1] results → $RESULTS_DIR"
echo

# Preflight: peek import + sitecustomize present.
"$PY" -c "import peek.engines.vllm.patch_hook" 2>/dev/null \
  || { echo "[w1] preflight FAIL: peek.engines.vllm.patch_hook not importable — run 'maturin develop --release' first"; exit 1; }
[[ -f "$SITECUSTOMIZE_DIR/sitecustomize.py" ]] \
  || { echo "[w1] preflight FAIL: sitecustomize shim missing: $SITECUSTOMIZE_DIR/sitecustomize.py"; exit 1; }

# Env fingerprint — written to the results dir so reviewers can check
# which engine/torch/python the W1 numbers were collected against.
VLLM_VERSION="$("$PY" -c 'import vllm; print(vllm.__version__)')"
TORCH_VERSION="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
PY_VERSION="$("$PY" --version 2>&1)"
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
ENV_FILE="$RESULTS_DIR/env.txt"
{
  echo "# vllm W1 run env fingerprint"
  echo "timestamp:      $(date -u +%FT%TZ)"
  echo "vllm:           $VLLM_VERSION"
  echo "torch:          $TORCH_VERSION"
  echo "python:         $PY_VERSION"
  echo "model:          $MODEL"
  echo "gpu_mem_util:   $GPU_MEM_UTIL"
  echo "max_model_len:  $MAX_MODEL_LEN"
  echo "cuda_graphs:    on (vllm 0.19 default)"
  echo "gpu:            $GPU_INFO"
  echo "peek_commit:    $(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "peek_branch:    $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "cells:          $CELLS"
  echo "rates:          $RATES"
  echo "seeds:          $SEEDS"
  echo "policies_full:  $POLICIES_FULL"
  echo "policies_core:  $POLICIES_CORE"
  echo "primary_cell:   $PRIMARY_CELL"
} > "$ENV_FILE"
echo "[w1] env fingerprint → $ENV_FILE"
echo "[w1]   vllm=$VLLM_VERSION torch=$TORCH_VERSION python=$PY_VERSION"
echo "[w1]   model=$MODEL gpu_mem_util=$GPU_MEM_UTIL"

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

    if [[ "$server_up" == 0 || "$FULL_RESTART" == "1" ]]; then
      slog="$slog_base"
      if ! launch_server "$policy" "$slog"; then
        echo "[w1]   LAUNCH FAILED for $policy — skipping"
        server_up=0; break
      fi
      server_up=1
    fi

    if ! run_one "$cell" "$rate_label" "$policy" "$seed" "$out"; then
      echo "[w1]   BENCH FAILED for $policy — server log tail:"
      tail -n 30 "$slog_base" || true
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
echo "[w1] done. aggregate with: $PY $REPO_ROOT/benchmarks/w1/aggregate.py"
