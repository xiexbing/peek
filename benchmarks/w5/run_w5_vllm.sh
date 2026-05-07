#!/bin/bash
# W5 benchmark driver — vLLM side, shared-prefix variant for cache-hit study.
#
# Cells (shared system-prompt → measurable cache hit rate)
#   C_short  prefix=512 tok,  decode=128, groups=20, zipf(α=1.2)
#   C_long   prefix=2048 tok, decode=128, groups=20, zipf(α=1.2)
#
# Policies (vLLM scheduler axis)
#   fcfs_lru   stock vLLM: FCFS + APC + LRU                       (no PEEK env)
#   clpm   pure peek FLPM (PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1) + LRU                 (no group_major)
#   clpm_gm_dl_pe   peek_flpm + group_major + dynamic_lane + peek_evict (demand_cluster)
#
# Closed-loop: rate=0, concurrency=256 → measures saturation throughput.
#
# Usage
#   bash run_w5_vllm.sh                                          # full (18 runs: 3p × 2c × 3s)
#   POLICIES=fcfs_lru CELLS=C_short SEEDS=42 bash run_w5_vllm.sh       # smoke

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_ROOT="${VENV_ROOT:-/workspace/peek}"
MODEL="${MODEL:-google/gemma-2-27b-it}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w5/results_vllm}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"

# HF token: bench streams the gated lmsys/lmsys-chat-1m dataset. Pull from
# ~/.cache/huggingface/token if HF_TOKEN isn't already in env.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN

CELLS="${CELLS:-C_short C_long}"
SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-fcfs_lru clpm clpm_gm_dl_pe}"
NUM_GROUPS="${NUM_GROUPS:-20}"
DISTRIBUTION="${DISTRIBUTION:-zipf}"
ZIPF_ALPHA="${ZIPF_ALPHA:-1.2}"

CONCURRENCY="${CONCURRENCY:-256}"
TTFT_SLO="${TTFT_SLO:-2000}"
TPOT_SLO="${TPOT_SLO:-100}"
E2E_SLO="${E2E_SLO:-60000}"

FULL_RESTART="${FULL_RESTART:-0}"

mkdir -p "$RESULTS_DIR"

# ------------------------------ cell params -------------------------------

declare -A CELL_PREFIX_TOK CELL_DECODE CELL_N CELL_WARMUP
CELL_PREFIX_TOK[C_short]=512;   CELL_DECODE[C_short]=128;  CELL_N[C_short]=1500; CELL_WARMUP[C_short]=100
CELL_PREFIX_TOK[C_long]=2048;   CELL_DECODE[C_long]=128;   CELL_N[C_long]=1500;  CELL_WARMUP[C_long]=100

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    fcfs_lru) echo "" ;;
    clpm) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  # Parent api_server (matches on cmdline).
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  # vLLM v1 spawns an EngineCore child that survives the parent — its
  # process name is set via setproctitle to "VLLM::EngineCore", so `pkill
  # -f` matches it on the proc title. Without this, the orphan keeps the
  # GPU and the next launch fails with "Engine core initialization failed".
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

launch_server() {
  local policy="$1" slog="$2"
  kill_server
  local env_pref; env_pref="$(policy_env "$policy")"

  echo "[w5-vllm] launching $policy (env='$env_pref') -> $slog"
  # PYTHONPATH always includes peek_sitecustomize — the shim is a no-op when
  # no PEEK_* env is set, so it's safe for fcfs_lru.
  env \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME" \
    HF_TOKEN="${HF_TOKEN:-}" \
    PYTHONPATH="$SITECUSTOMIZE_DIR:${PYTHONPATH:-}" \
    $env_pref \
    "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --enable-prefix-caching \
      --enable-prompt-tokens-details \
      >"$slog" 2>&1 &
  local pid=$!
  local iters=$(( SERVER_READY_TIMEOUT_S / 3 ))
  for i in $(seq 1 $iters); do
    sleep 3
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w5-vllm]   ready after $((i*3))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w5-vllm]   server died; tail of log:"; tail -n 30 "$slog" || true
      return 1
    fi
    if (( i % 20 == 0 )); then
      local nvmem
      nvmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
      echo "[w5-vllm]   ...still loading at $((i*3))s, gpu_mem=${nvmem}MiB"
    fi
  done
  echo "[w5-vllm]   FAILED to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1 || true
  sleep 1
}

snapshot_metrics() {
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(prompt_tokens_total|generation_tokens_total|num_preemptions_total|gpu_cache_usage_perc|num_requests_waiting|num_requests_running|prefix_cache_hit_rate)[ {]' \
    > "$1" 2>/dev/null || true
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local cell="$1" policy="$2" seed="$3" out="$4"

  local prefix_tok="${CELL_PREFIX_TOK[$cell]}"
  local decode="${CELL_DECODE[$cell]}"
  local N="${CELL_N[$cell]}" warmup="${CELL_WARMUP[$cell]}"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"

  flush_cache

  local blog="$outdir/_run_${policy}.log"
  local label; label="$(basename "$out" .json)"
  echo "[w5-vllm]   benching $policy (cell=$cell prefix=$prefix_tok decode=$decode groups=$NUM_GROUPS dist=$DISTRIBUTION N=$N seed=$seed closed-loop conc=$CONCURRENCY)"

  snapshot_metrics "$outdir/_metrics_pre_${policy}.prom"

  BENCH_NO_SYSTEM_ROLE=1 "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" \
    --rate 0 \
    --concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --warmup-reqs "$warmup" \
    --ttft-slo-ms "$TTFT_SLO" \
    --tpot-slo-ms "$TPOT_SLO" \
    --e2e-slo-ms "$E2E_SLO" \
    --groups "$NUM_GROUPS" \
    --prefix-tokens "$prefix_tok" \
    --max-tokens "$decode" \
    --distribution "$DISTRIBUTION" \
    --zipf-alpha "$ZIPF_ALPHA" \
    --dataset synthetic \
    --label "$label" \
    --output "$out" \
    --save-per-request \
    > "$blog" 2>&1
  local rc=$?

  snapshot_metrics "$outdir/_metrics_post_${policy}.prom"

  if [[ $rc -ne 0 ]]; then
    echo "[w5-vllm]   BENCH FAILED rc=$rc, tail of log:"
    tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w5-vllm]   wrote $out"
  return 0
}

# ------------------------------ main loop ---------------------------------
# Policy-major: one server per policy, all (seed × cell × rate) benches
# back-to-back with /reset_prefix_cache between.

declare -A policy_plan
declare -a policy_order
for seed in $SEEDS; do
  for cell in $CELLS; do
    for policy in $POLICIES; do
      if [[ -z "${policy_plan[$policy]:-}" ]]; then
        policy_order+=("$policy")
        policy_plan[$policy]="$seed|$cell"
      else
        policy_plan[$policy]+=$'\n'"$seed|$cell"
      fi
    done
  done
done

total_runs=0
for p in "${policy_order[@]}"; do
  n=$(echo -n "${policy_plan[$p]}" | awk 'END{print NR}')
  total_runs=$((total_runs + n))
done

echo "[w5-vllm] plan: $total_runs runs across ${#policy_order[@]} distinct policies"
echo "[w5-vllm] cells=$CELLS  seeds=$SEEDS  policies=$POLICIES  groups=$NUM_GROUPS dist=$DISTRIBUTION"
echo "[w5-vllm] model=$MODEL  results -> $RESULTS_DIR"
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
    IFS='|' read -r seed cell <<< "$entry"
    outdir="$RESULTS_DIR/seed_${seed}/cell_${cell}"
    out="$outdir/${policy}.json"
    mkdir -p "$outdir"

    echo
    echo "----- [$idx/$total_runs] policy=$policy seed=$seed cell=$cell -----"

    if [[ -f "$out" ]]; then
      echo "[w5-vllm]   skip: $out already exists"
      continue
    fi

    if [[ "$server_up" == 0 || "$FULL_RESTART" == "1" ]]; then
      slog="$slog_base"
      if ! launch_server "$policy" "$slog"; then
        echo "[w5-vllm]   LAUNCH FAILED for $policy — skipping remaining benches"
        server_up=0
        break
      fi
      server_up=1
    fi

    if ! run_one "$cell" "$policy" "$seed" "$out"; then
      echo "[w5-vllm]   BENCH FAILED — restarting server"
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
echo "[w5-vllm] done."
