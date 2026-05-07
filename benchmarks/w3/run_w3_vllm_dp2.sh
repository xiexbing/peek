#!/bin/bash
# W3 cell C/B vLLM-side DP=2 runner (paper §4.3, multi-GPU 70B, 4 GPUs).
#
# Architecture:
#   vllm-A: TP=2 on GPUs 0,1 at port 31000  (--enable-prefix-caching)
#   vllm-B: TP=2 on GPUs 2,3 at port 31001  (--enable-prefix-caching)
#   sglang_router: port 30000, --policy cache_aware,
#                  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001
#
# Bench client hits the router at port 30000. Cache hits land in the OpenAI
# response usage block (`prompt_tokens_details.cached_tokens`) -- vllm v1
# populates this when prefix caching is on, so the bench's existing parser
# captures hit-rate without code changes.
#
# Peek hooks:
#   peek.online.engines.vllm.patch_hook is loaded via the sitecustomize.py
#   shim on PYTHONPATH, so each vllm worker (parent + spawned EngineCore
#   child) inherits the hooks gated by PEEK_ONLINE_* env vars.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"

# Public router port (bench client hits this).
PORT="${PORT:-30000}"
# Internal vllm worker ports.
WORKER_A_PORT="${WORKER_A_PORT:-31000}"
WORKER_B_PORT="${WORKER_B_PORT:-31001}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w3/results_vllm_dp2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9728}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"

# Cell C defaults; CELL=B + DECODE_MIX overrides for RAG.
CELL="${CELL:-C}"
NGROUPS="${NGROUPS:-88}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1500}"
N="${N:-1000}"
WARMUP="${WARMUP:-200}"
CONCURRENCY="${CONCURRENCY:-360}"
DECODE_MIX="${DECODE_MIX:-}"
RATE_LABEL="${RATE_LABEL:-heavy}"

SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-fcfs_apc_lru clpm_gm_dl_pe}"

mkdir -p "$RESULTS_DIR"

# --- server lifecycle ----------------------------------------------------

kill_server() {
  pkill -9 -f "sglang_router.launch_router.*--port $PORT" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  # setproctitle hides the original cmdline; match by ps comm instead.
  ps -ef | awk '/sglang::|VLLM::|sglang_router/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/torch._inductor.compile_worker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/multiprocessing.resource_tracker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 8
}

policy_env() {
  case "$1" in
    fcfs_apc_lru) echo "" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

launch_one_vllm() {
  # $1 = port, $2 = CUDA_VISIBLE_DEVICES, $3 = log path, $4 = policy
  local port="$1" gpus="$2" slog="$3" policy="$4"
  local env_pref; env_pref="$(policy_env "$policy")"
  echo "[w4-vllm-dp2]   spawning vllm worker on port $port (CUDA=$gpus, env='$env_pref')"
  env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_USE_V1=1 \
      CUDA_VISIBLE_DEVICES="$gpus" \
      $env_pref "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --enable-prefix-caching \
      --enable-prompt-tokens-details \
      --max-model-len "$MAX_MODEL_LEN" \
      --host 127.0.0.1 --port "$port" \
      >>"$slog" 2>&1 &
}

wait_worker_ready() {
  local port="$1" t0
  t0="$(date +%s)"
  while true; do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
      "http://127.0.0.1:$port/v1/chat/completions" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[w4-vllm-dp2]   worker $port ready after $(( $(date +%s) - t0 ))s"
      return 0
    fi
    if (( $(date +%s) - t0 > 1800 )); then
      echo "[w4-vllm-dp2]   worker $port FAILED to be ready in 1800s"
      return 1
    fi
    sleep 10
  done
}

launch_router() {
  local slog="$1"
  echo "[w4-vllm-dp2]   launching router (policy=$ROUTER_POLICY)"
  "$PY" -m sglang_router.launch_router \
    --host 127.0.0.1 --port "$PORT" \
    --policy "$ROUTER_POLICY" \
    --worker-urls "http://127.0.0.1:$WORKER_A_PORT" "http://127.0.0.1:$WORKER_B_PORT" \
    --log-level warn \
    >>"$slog" 2>&1 &
  local t0; t0="$(date +%s)"
  while true; do
    local resp
    resp="$(curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"
    if echo "$resp" | grep -q '"id"'; then
      echo "[w4-vllm-dp2]   router ready after $(( $(date +%s) - t0 ))s"
      sleep 2
      return 0
    fi
    if (( $(date +%s) - t0 > 120 )); then
      echo "[w4-vllm-dp2]   router failed to come up in 120s"
      return 1
    fi
    sleep 5
  done
}

launch_server() {
  local policy="$1" slog_a="$2" slog_b="$3" slog_r="$4"
  launch_one_vllm "$WORKER_A_PORT" "0,1" "$slog_a" "$policy"
  launch_one_vllm "$WORKER_B_PORT" "2,3" "$slog_b" "$policy"
  wait_worker_ready "$WORKER_A_PORT" || return 1
  wait_worker_ready "$WORKER_B_PORT" || return 1
  launch_router "$slog_r" || return 1
}

# vllm has no /flush_cache; per-policy server cycle handles cold-start.

# --- bench ---------------------------------------------------------------

run_bench() {
  local policy="$1" seed="$2"
  local outdir="$RESULTS_DIR/seed_${seed}/cell_${CELL}/rate_${RATE_LABEL}"
  mkdir -p "$outdir"
  local out="$outdir/${policy}.json"
  local blog="$outdir/_run_${policy}.log"
  if [[ -f "$out" ]]; then
    echo "[w4-vllm-dp2]   skip: $out already exists"
    return 0
  fi
  echo "[w4-vllm-dp2]   benching $policy seed=$seed cell=$CELL G=$NGROUPS prefix=$PREFIX_TOKENS concurrency=$CONCURRENCY decode='${DECODE_MIX:-fixed=128}' -> $out"
  local decode_args=()
  if [[ -n "$DECODE_MIX" ]]; then
    decode_args=(--decode-mix "$DECODE_MIX")
  else
    decode_args=(--max-tokens 128)
  fi
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" \
    --groups "$NGROUPS" --prefix-tokens "$PREFIX_TOKENS" \
    --dataset auto \
    --distribution zipf --zipf-alpha 1.0 \
    "${decode_args[@]}" \
    --rate 0 --concurrency "$CONCURRENCY" \
    --seed "$seed" --warmup-reqs "$WARMUP" \
    --ttft-slo-ms 2000 --tpot-slo-ms 100 --e2e-slo-ms 60000 \
    --label "${policy}_cell${CELL}_seed${seed}_${RATE_LABEL}_vllm_dp2" \
    --output "$out" \
    > "$blog" 2>&1
  echo "[w4-vllm-dp2]   $policy seed=$seed done"
}

# --- main ----------------------------------------------------------------

echo "[w4-vllm-dp2] === START === $(date)"
echo "[w4-vllm-dp2] cell=$CELL G=$NGROUPS prefix=$PREFIX_TOKENS N=$N warmup=$WARMUP concurrency=$CONCURRENCY"
echo "[w4-vllm-dp2] policies=$POLICIES seeds=$SEEDS results=$RESULTS_DIR"

for policy in $POLICIES; do
  echo
  echo "[w4-vllm-dp2] ##### policy: $policy #####"
  kill_server
  slog_a="$RESULTS_DIR/_server_${policy}_workerA.log"
  slog_b="$RESULTS_DIR/_server_${policy}_workerB.log"
  slog_r="$RESULTS_DIR/_server_${policy}_router.log"
  launch_server "$policy" "$slog_a" "$slog_b" "$slog_r" || exit 1
  for seed in $SEEDS; do
    run_bench "$policy" "$seed"
  done
done

kill_server
echo
echo "[w4-vllm-dp2] === DONE === $(date)"
