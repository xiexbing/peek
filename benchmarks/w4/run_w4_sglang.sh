#!/bin/bash
# W4 — Agentic LLM serving runner (paper §4.4).
#
# Workload: Mooncake conversation_trace, num_rounds=4 burst with tight
# inter-turn gap — models tool-chain bursts as in Cursor / Copilot /
# Claude Code / RAG pipelines.
#
# Policies (paper Table 2 labels):
#   lpm_lru        SGLang LPM + LRU                                 — baseline
#   clpm_gm_dl     cLPM + GM + DL + LRU                             — scheduling-only
#   clpm_gm_dl_pe  cLPM + GM + DL + queue-aware (cluster) eviction  — full PEEK
#
# Cells (calibrated against lpm_lru):
#   moderate  num_prompts=30   → mean queue ~30–50,  peak ~80
#   heavy     num_prompts=120  → mean queue ~80–150, peak ~200
#
# Usage:
#   bash benchmarks/w4/run_w4_sglang.sh                              # full matrix
#   POLICIES="lpm_lru clpm_gm_dl" SEEDS=42 bash run_w4_sglang.sh     # subset

set -uo pipefail

# ------------------------------ config ------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
W4_ROOT="$REPO_ROOT/benchmarks/w4"
W4_DATA="$W4_ROOT/data"
RESULTS_DIR="${RESULTS_DIR:-$W4_ROOT/results}"
MOONCAKE_PATH="$W4_DATA/conversation_trace_le6k.jsonl"

MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
MEM_FRAC="${MEM_FRAC:-0.88}"
PORT="${PORT:-30000}"
PY="${PY:-python3}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"

POLICIES="${POLICIES:-lpm_lru clpm_gm_dl clpm_gm_dl_pe}"
SEEDS="${SEEDS:-42 142 242}"
DATASETS="${DATASETS:-mooncake}"
CELLS="${CELLS:-moderate heavy}"

# Inter-turn gap (passed to bench harness via env, applied to
# get_mooncake_request_over_time after PEEK PATCH applied).
INTER_TURN_MEDIAN_MS="${INTER_TURN_MEDIAN_MS:-50}"
INTER_TURN_SIGMA="${INTER_TURN_SIGMA:-0.5}"
SHARED_SYSTEM_PROMPT_PATH="${SHARED_SYSTEM_PROMPT_PATH:-$W4_DATA/shared_system_prompt.txt}"

# Agentic burst cells
declare -A MC_NUM_PROMPTS MC_NUM_ROUNDS MC_SLOWDOWN
MC_NUM_PROMPTS[moderate]=30  ; MC_NUM_ROUNDS[moderate]=4 ; MC_SLOWDOWN[moderate]=1.0
MC_NUM_PROMPTS[heavy]=120    ; MC_NUM_ROUNDS[heavy]=4    ; MC_SLOWDOWN[heavy]=1.0

# ------------------------------ policy env --------------------------------

policy_env() {
  case "$1" in
    fcfs_lru)            echo "" ;;
    lpm_lru)            echo "" ;;
    clpm_gm_dl)        echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1" ;;
    clpm_gm_dl_pe)        echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *) echo "ERR_POLICY"; return 1 ;;
  esac
}

policy_schedule() {
  case "$1" in
    fcfs_lru) echo "fcfs" ;;
    *)  echo "lpm" ;;
  esac
}

# ------------------------------ server lifecycle --------------------------

kill_server() {
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
  sleep 4
}

launch_server() {
  local policy="$1" slog="$2"
  kill_server
  local env_pref; env_pref="$(policy_env "$policy")"
  local sched; sched="$(policy_schedule "$policy")"

  echo "[w4] launching $policy  schedule=$sched  env='$env_pref'  -> $slog"
  env $env_pref "$PY" -m sglang.launch_server \
    --model-path "$MODEL" \
    --mem-fraction-static "$MEM_FRAC" \
    --schedule-policy "$sched" \
    --enable-cache-report \
    --enable-metrics \
    --host 127.0.0.1 --port "$PORT" \
    --log-level warning \
    >"$slog" 2>&1 &

  local pid=$!
  local iters=$(( SERVER_READY_TIMEOUT_S / 3 ))
  for i in $(seq 1 $iters); do
    sleep 3
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w4]   ready after $((i*3))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[w4]   server died; tail of log:"; tail -n 20 "$slog" || true
      return 1
    fi
  done
  echo "[w4]   FAILED to be ready in ${SERVER_READY_TIMEOUT_S}s"
  kill_server
  return 1
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 1
}

# ------------------------------ one bench run -----------------------------

run_one() {
  local policy="$1" dataset="$2" cell="$3" seed="$4" out="$5"

  local outdir; outdir="$(dirname "$out")"
  mkdir -p "$outdir"

  flush_cache
  local blog="$outdir/_run.log"
  echo "[w4]   benching policy=$policy dataset=$dataset cell=$cell seed=$seed"

  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^sglang:(prompt_tokens_total|cached_tokens_total|generation_tokens_total|num_retractions_sum|num_retractions_count|realtime_tokens_total)[ {]' \
    > "$outdir/_metrics_pre.prom" 2>/dev/null || true

  local args=(
    --backend sglang
    --model "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --seed "$seed"
    --disable-tqdm
    --output-file "$out"
    --output-details
  )

  case "$dataset" in
    mooncake)
      args+=(
        --dataset-name mooncake
        --dataset-path "$MOONCAKE_PATH"
        --num-prompts "${MC_NUM_PROMPTS[$cell]}"
        --mooncake-num-rounds "${MC_NUM_ROUNDS[$cell]}"
        --mooncake-slowdown-factor "${MC_SLOWDOWN[$cell]}"
      )
      ;;
    *) echo "ERR_DATASET=$dataset"; return 1 ;;
  esac

  PEEK_AGENT_INTER_TURN_MEDIAN_MS="$INTER_TURN_MEDIAN_MS" \
  PEEK_AGENT_INTER_TURN_SIGMA="$INTER_TURN_SIGMA" \
  PEEK_SHARED_SYSTEM_PROMPT_PATH="$SHARED_SYSTEM_PROMPT_PATH" \
    "$PY" -m sglang.bench_serving "${args[@]}" > "$blog" 2>&1
  local rc=$?

  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^sglang:(prompt_tokens_total|cached_tokens_total|generation_tokens_total|num_retractions_sum|num_retractions_count|realtime_tokens_total)[ {]' \
    > "$outdir/_metrics_post.prom" 2>/dev/null || true

  if [[ $rc -ne 0 ]]; then
    echo "[w4]   BENCH FAILED rc=$rc  tail of log:"
    tail -n 40 "$blog" || true
    return 1
  fi
  echo "[w4]   wrote $out"
  return 0
}

# ------------------------------ main loop ---------------------------------

mkdir -p "$RESULTS_DIR"

total_runs=0
for p in $POLICIES; do
  for s in $SEEDS; do
    for d in $DATASETS; do
      for c in $CELLS; do
        total_runs=$((total_runs+1))
      done
    done
  done
done

echo "[w4] plan: $total_runs runs across $(echo $POLICIES | wc -w) policies"
echo "[w4] policies=$POLICIES  seeds=$SEEDS  datasets=$DATASETS  cells=$CELLS"
echo "[w4] inter-turn gap: LogNormal(median=${INTER_TURN_MEDIAN_MS}ms, sigma=${INTER_TURN_SIGMA})"
echo "[w4] shared system prompt: $SHARED_SYSTEM_PROMPT_PATH"
echo "[w4] results -> $RESULTS_DIR"
echo

idx=0
for policy in $POLICIES; do
  echo
  echo "##### policy: $policy #####"

  slog="$RESULTS_DIR/_server_${policy}.log"
  if ! launch_server "$policy" "$slog"; then
    echo "[w4]   LAUNCH FAILED for $policy — skipping all $policy runs"
    continue
  fi

  for seed in $SEEDS; do
    for dataset in $DATASETS; do
      for cell in $CELLS; do
        idx=$((idx+1))
        outdir="$RESULTS_DIR/$policy/${dataset}_${cell}/seed_${seed}"
        out="$outdir/result.jsonl"
        mkdir -p "$outdir"

        echo
        echo "----- [$idx/$total_runs] policy=$policy dataset=$dataset cell=$cell seed=$seed -----"

        if [[ -f "$out" ]]; then
          echo "[w4]   skip: $out already exists"
          continue
        fi

        if ! run_one "$policy" "$dataset" "$cell" "$seed" "$out"; then
          echo "[w4]   bench failed; restarting server for $policy"
          if ! launch_server "$policy" "$slog"; then
            echo "[w4]   relaunch failed; skipping rest of $policy"
            break 3
          fi
        fi
      done
    done
  done

  kill_server
done

echo
echo "[w4] done. aggregate with: $PY $W4_ROOT/aggregate.py"
