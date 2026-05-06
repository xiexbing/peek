#!/bin/bash
# W2 — Agentic LLM serving on vllm 0.19.1 (peek's flpm-vllm engine).
#
# Mirror of run_w2_agentic.sh adapted for vllm:
#   - Server: vllm.entrypoints.openai.api_server (peek hooks via sitecustomize)
#   - Bench: bench_mooncake_vllm.py (this directory; no sglang dependency)
#   - kill_server reaps orphan VLLM::EngineCore subprocesses (W1 lesson)
#
# Policies (3-policy lattice as agreed for vllm):
#   fcfs_apc_lru       APC + LRU (vanilla vllm)                                     baseline
#   clpm_gm_dl   peek_flpm + group_major + dynamic_lane                        scheduling-only
#   clpm_gm_dl_pe   peek_flpm + group_major + dynamic_lane + peek_evict cluster   co-design
#
# Cells (calibrated against fcfs_apc_lru — start from the sglang W2 numbers and
# adjust after a smoke run):
#   moderate  num_prompts=30   ~ engaged steady-state
#   heavy     num_prompts=120  ~ engaged peak load
#
# Usage:
#   bash run_w2_vllm.sh                                            # full matrix
#   POLICIES="fcfs_apc_lru clpm_gm_dl" SEEDS=42 CELLS=moderate bash run_w2_vllm.sh   # subset
#   SHARED_SYSTEM_PROMPT_PATH="" RESULTS_DIR=$PWD/results/agentic_only_vllm bash run_w2_vllm.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
W4_ROOT="$REPO_ROOT/benchmarks/w4"
W4_DATA="$W4_ROOT/data"
RESULTS_DIR="${RESULTS_DIR:-$W4_ROOT/results_vllm/agentic_shared}"
MOONCAKE_PATH="$W4_DATA/conversation_trace_le6k.jsonl"

MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
PORT="${PORT:-30000}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PY="${PY:-python3}"
BENCH="${BENCH:-$W4_ROOT/bench_mooncake_vllm.py}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

POLICIES="${POLICIES:-fcfs_apc_lru clpm_gm_dl clpm_gm_dl_pe}"
SEEDS="${SEEDS:-42 142 242}"
CELLS="${CELLS:-moderate heavy}"

# Mooncake burst params (match flpm-w2's run_w2_agentic.sh defaults)
NUM_ROUNDS="${NUM_ROUNDS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
INTER_TURN_MEDIAN_MS="${INTER_TURN_MEDIAN_MS:-50}"
INTER_TURN_SIGMA="${INTER_TURN_SIGMA:-0.5}"
SHARED_SYSTEM_PROMPT_PATH="${SHARED_SYSTEM_PROMPT_PATH:-$W4_DATA/shared_system_prompt.txt}"

# SLOs — re-using W1's defaults for consistency
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-200}"
E2E_SLO="${E2E_SLO:-60000}"

# Per-cell num_prompts (sessions). Same as sglang W2 to start; adjust
# after a smoke run if vllm saturates differently.
declare -A CELL_PROMPTS CELL_CONCURRENCY
CELL_PROMPTS[moderate]="${CELL_PROMPTS_MODERATE:-30}"
CELL_CONCURRENCY[moderate]="${CELL_CONCURRENCY_MODERATE:-64}"
CELL_PROMPTS[heavy]="${CELL_PROMPTS_HEAVY:-120}"
CELL_CONCURRENCY[heavy]="${CELL_CONCURRENCY_HEAVY:-200}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    fcfs_apc_lru)      echo "" ;;
    clpm_gm_dl)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_dl_pe)  echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *) echo "ERR_POLICY"; return 1 ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  pkill -9 -f "vllm.*serve.*--port $PORT" 2>/dev/null || true
  sleep 3
  # Reap the spawn-child EngineCore (W1 lesson: SIGKILL to APIServer
  # does not always reap it; orphan pins ~75 GiB GPU and the next
  # cold-start fails with "Free memory on device cuda:0 < gpu_mem_util").
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

  echo "[w4] launching $policy (env='$env_pref') → $slog"
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
      --enable-prefix-caching \
      >"$slog" 2>&1 &
  local pid=$!
  local iters=$(( SERVER_READY_TIMEOUT_S / 3 ))
  for i in $(seq 1 $iters); do
    sleep 3
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w4]   ready after $((i*3))s (via /v1/models)"
      return 0
    fi
    if grep -qE "Application startup complete|Uvicorn running on|Started server process" "$slog" 2>/dev/null; then
      sleep 1
      echo "[w4]   ready after $((i*3))s (log marker)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w4]   server died; tail of log:"; tail -n 20 "$slog" || true
      return 1
    fi
    if (( i % 20 == 0 )); then
      local nvmem
      nvmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
      echo "[w4]   ...still loading at $((i*3))s, gpu_mem=${nvmem}MiB"
    fi
  done
  echo "[w4]   FAILED to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null || true
  sleep 1
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local policy="$1" cell="$2" seed="$3" out="$4"
  local n="${CELL_PROMPTS[$cell]}" cap="${CELL_CONCURRENCY[$cell]}"
  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"
  flush_cache

  local blog="$outdir/_run.log"
  local label="$policy"
  echo "[w4]   benching $policy (cell=$cell n=$n cap=$cap rounds=$NUM_ROUNDS seed=$seed)"
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --dataset-path "$MOONCAKE_PATH" \
    --num-prompts "$n" \
    --num-rounds "$NUM_ROUNDS" \
    --output-len "$OUTPUT_LEN" \
    --inter-turn-gap-median-ms "$INTER_TURN_MEDIAN_MS" \
    --inter-turn-gap-sigma "$INTER_TURN_SIGMA" \
    --shared-system-prompt-path "$SHARED_SYSTEM_PROMPT_PATH" \
    --seed "$seed" \
    --concurrency "$cap" \
    --ttft-slo-ms "$TTFT_SLO" --tpot-slo-ms "$TPOT_SLO" --e2e-slo-ms "$E2E_SLO" \
    --label "$label" --output "$out" \
    > "$blog" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[w4]   BENCH FAILED rc=$rc tail of log:"; tail -n 30 "$blog" || true
    return 1
  fi
  echo "[w4]   wrote $out"
}

# ------------------------------ main loop ---------------------------------
# Policy-major: one server launch per policy; all (seed, cell) combos for
# that policy run back-to-back, then the server is torn down.

total_runs=0
for p in $POLICIES; do for s in $SEEDS; do for c in $CELLS; do total_runs=$((total_runs+1)); done; done; done

echo "[w4] plan: $total_runs runs across $(echo $POLICIES | wc -w) policies"
echo "[w4] policies=$POLICIES  seeds=$SEEDS  cells=$CELLS"
echo "[w4] inter-turn gap: LogNormal(median=${INTER_TURN_MEDIAN_MS}ms, sigma=${INTER_TURN_SIGMA})"
[[ -n "${SHARED_SYSTEM_PROMPT_PATH}" && -f "${SHARED_SYSTEM_PROMPT_PATH}" ]] \
  && echo "[w4] shared system prompt: $SHARED_SYSTEM_PROMPT_PATH" \
  || echo "[w4] no shared system prompt"
echo "[w4] results → $RESULTS_DIR"

# Preflight
"$PY" -c "import peek.online.engines.vllm.patch_hook" 2>/dev/null \
  || { echo "[w4] preflight FAIL: peek.online.engines.vllm.patch_hook not importable"; exit 1; }
[[ -f "$SITECUSTOMIZE_DIR/sitecustomize.py" ]] \
  || { echo "[w4] preflight FAIL: sitecustomize shim missing: $SITECUSTOMIZE_DIR/sitecustomize.py"; exit 1; }
[[ -f "$MOONCAKE_PATH" ]] \
  || { echo "[w4] preflight FAIL: mooncake trace missing: $MOONCAKE_PATH"; exit 1; }

# Env fingerprint
ENV_FILE="$RESULTS_DIR/env.txt"
{
  echo "# vllm W2 run env fingerprint"
  echo "timestamp:    $(date -u +%FT%TZ)"
  echo "vllm:         $($PY -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "torch:        $($PY -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
  echo "python:       $($PY --version 2>&1)"
  echo "model:        $MODEL"
  echo "gpu_mem_util: $GPU_MEM_UTIL"
  echo "max_model_len:$MAX_MODEL_LEN"
  echo "policies:     $POLICIES"
  echo "cells:        $CELLS"
  echo "seeds:        $SEEDS"
  echo "num_rounds:   $NUM_ROUNDS"
  echo "output_len:   $OUTPUT_LEN"
  echo "inter_turn_median_ms: $INTER_TURN_MEDIAN_MS"
  echo "inter_turn_sigma:     $INTER_TURN_SIGMA"
  echo "shared_sys:   ${SHARED_SYSTEM_PROMPT_PATH:-(none)}"
  echo "peek_commit:  $(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "peek_branch:  $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
} > "$ENV_FILE"
echo "[w4] env → $ENV_FILE"

idx=0
for policy in $POLICIES; do
  echo
  echo "##### policy: $policy #####"
  slog="$RESULTS_DIR/_server_${policy}.log"
  server_up=0
  for seed in $SEEDS; do
    for cell in $CELLS; do
      idx=$((idx+1))
      outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}"
      out="$outdir/${policy}.json"
      mkdir -p "$outdir"

      echo
      echo "----- [$idx/$total_runs] policy=$policy seed=$seed cell=$cell -----"
      if [[ -f "$out" ]]; then
        echo "[w4]   skip: $out already exists"
        continue
      fi

      if [[ "$server_up" == 0 ]]; then
        if ! launch_server "$policy" "$slog"; then
          echo "[w4]   LAUNCH FAILED for $policy — skipping"
          break 2
        fi
        server_up=1
      fi

      if ! run_one "$policy" "$cell" "$seed" "$out"; then
        echo "[w4]   bench failed; restarting server"
        kill_server; server_up=0
        if ! launch_server "$policy" "$slog"; then
          echo "[w4]   relaunch failed; skipping rest of $policy"
          break 2
        fi
        server_up=1
      fi
    done
  done
  if [[ "$server_up" == 1 ]]; then
    kill_server
    server_up=0
  fi
done

echo
echo "[w4] done at $(date -Iseconds)"
