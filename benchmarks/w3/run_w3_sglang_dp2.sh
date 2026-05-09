#!/bin/bash
# W3 SGLang DP=2 (paper §4.3, multi-GPU 70B). Server is launched via sglang's
# router with 2 DP replicas (each TP=2). Total 4 GPUs.
#
# Architecture:
#   - launcher: `python -m sglang_router.launch_server` (router co-launches
#     workers and load-balances in front of them)
#   - --tp 2 --dp-size 2 (cluster capacity ~2x; KV pool per replica ~16,754)
#   - --router-policy cache_aware (default; prefix-aware)
#   - results dir: $REPO_ROOT/benchmarks/w3/results_sglang_dp2/
#
# Cells C and B mirror W1 (chat-like, admission-bound) and W2 (RAG-like,
# decode-bound); see benchmarks/w3/README.md. Concurrency cap stays at 180
# by default for apples-to-apples comparison.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python3}"
BENCH="${BENCH:-$REPO_ROOT/scripts/bench/bench_shared_prompts.py}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.88}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/w3/results_sglang_dp2}"
SITECUSTOMIZE_DIR="${SITECUSTOMIZE_DIR:-$REPO_ROOT/scripts/peek_sitecustomize}"

TP="${TP:-2}"
DP="${DP:-2}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"

# Default cell C (chat). Override CELL=B and DECODE_MIX for cell B (RAG).
CELL="${CELL:-C}"
NGROUPS="${NGROUPS:-88}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1500}"
N="${N:-1000}"
WARMUP="${WARMUP:-200}"
CONCURRENCY="${CONCURRENCY:-180}"
DECODE_MIX="${DECODE_MIX:-}"

RATE_LABEL="${RATE_LABEL:-heavy}"

SEEDS="${SEEDS:-42 142 242}"
POLICIES="${POLICIES:-lpm_lru clpm_gm_dl_pe}"

mkdir -p "$RESULTS_DIR"

# --- server lifecycle ----------------------------------------------------

kill_server() {
  # sglang renames argv via setproctitle to "sglang::router/server/scheduler"
  # so pkill -f against the original cmdline misses them. Match by ps comm.
  pkill -9 -f "sglang_router.launch_server.*--port $PORT" 2>/dev/null || true
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  ps -ef | awk '/sglang::/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/torch._inductor.compile_worker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  ps -ef | awk '/multiprocessing.resource_tracker/ && !/awk/ {print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 8
}

policy_env() {
  case "$1" in
    lpm_lru) echo "" ;;
    clpm_gm_dl_pe) echo "PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster" ;;
    *)  echo "ERR_POLICY"; return 1 ;;
  esac
}

launch_server() {
  local policy="$1" slog="$2"
  local env_pref; env_pref="$(policy_env "$policy")"
  echo "[w3-dp2] launching $policy router (tp=$TP dp=$DP policy=$ROUTER_POLICY env='$env_pref')"
  env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="$SITECUSTOMIZE_DIR:${PYTHONPATH:-}" \
      $env_pref "$PY" -m sglang_router.launch_server \
      --model-path "$MODEL" \
      --tp "$TP" --dp-size "$DP" \
      --mem-fraction-static "$MEM_FRAC" \
      --schedule-policy lpm \
      --enable-cache-report \
      --router-policy "$ROUTER_POLICY" \
      --host 127.0.0.1 --port "$PORT" \
      --log-level warning >>"$slog" 2>&1 &
  local t0; t0="$(date +%s)"
  # The router's /health returns 200 immediately even before workers are
  # loaded. Wait for /v1/models to list the model AND a real generation
  # request to succeed before declaring ready.
  while true; do
    local models_resp
    models_resp="$(curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"
    if echo "$models_resp" | grep -q '"id"'; then
      # Workers registered. Now verify a 1-token completion goes through.
      local code
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
      if [[ "$code" == "200" ]]; then
        echo "[w3-dp2]   ready after $(( $(date +%s) - t0 ))s (models listed + 1-tok smoke OK)"
        sleep 3
        return 0
      fi
    fi
    if (( $(date +%s) - t0 > 1200 )); then
      echo "[w3-dp2]   server failed to start in 1200s"
      return 1
    fi
    sleep 10
  done
}

flush_cache() {
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null 2>&1 \
    && echo "[w3-dp2]   /flush_cache ok" || echo "[w3-dp2]   /flush_cache failed"
  sleep 2
}

# --- bench ---------------------------------------------------------------

run_bench() {
  local policy="$1" seed="$2"
  local outdir="$RESULTS_DIR/seed_${seed}/cell_${CELL}/rate_${RATE_LABEL}"
  mkdir -p "$outdir"
  local out="$outdir/${policy}.json"
  local blog="$outdir/_run_${policy}.log"
  if [[ -f "$out" ]]; then
    echo "[w3-dp2]   skip: $out already exists"
    return 0
  fi
  echo "[w3-dp2]   benching $policy seed=$seed cell=$CELL G=$NGROUPS prefix=$PREFIX_TOKENS concurrency=$CONCURRENCY decode_mix='${DECODE_MIX:-fixed=128}' -> $out"
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
    --label "${policy}_cell${CELL}_seed${seed}_${RATE_LABEL}_dp${DP}" \
    --output "$out" \
    > "$blog" 2>&1
  echo "[w3-dp2]   $policy seed=$seed done"
}

# --- main ----------------------------------------------------------------

echo "[w3-dp2] === START === $(date)"
echo "[w3-dp2] tp=$TP dp=$DP router=$ROUTER_POLICY"
echo "[w3-dp2] cell=$CELL G=$NGROUPS prefix=$PREFIX_TOKENS N=$N warmup=$WARMUP concurrency=$CONCURRENCY"
echo "[w3-dp2] policies=$POLICIES  seeds=$SEEDS"
echo "[w3-dp2] results -> $RESULTS_DIR"

for policy in $POLICIES; do
  echo
  echo "[w3-dp2] ##### policy: $policy #####"
  kill_server
  launch_server "$policy" "$RESULTS_DIR/_server_${policy}.log" || exit 1
  for seed in $SEEDS; do
    flush_cache
    run_bench "$policy" "$seed"
  done
done

kill_server
echo
echo "[w3-dp2] === DONE === $(date)"
