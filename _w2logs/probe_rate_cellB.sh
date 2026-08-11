#!/bin/bash
# Cell-B rate calibration probe (W2, sglang, stock lpm_lru baseline).
#
# Why: the matrix's cell_rate() uses B=0.40/0.45 req/s, but measured capacity
# is ~0.33 req/s and throughput COLLAPSES past it (0.45 offered -> 0.257
# achieved, hit-rate 72%->46%). Paper Table 15 reports cell-B throughput of
# 0.114 (moderate) / 0.143-0.151 (heavy) req/s, i.e. the paper's operating
# point is ~3x below what the driver currently offers.
#
# This sweeps three candidate rates on the baseline policy and reports, per
# rate, whether the system is stable (achieved ~= offered, bounded TTFT) or
# backlogged. Pick the highest rate that stays stable, then run the
# head-to-head lpm_lru vs clpm_gm_dl_pe at that rate.
#
# One server boot, three benches, N=300 (warmup 50) to keep each point short.
set -uo pipefail

REPO_ROOT=/workspace/peek
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MEM_FRAC="${MEM_FRAC:-0.88}"
PORT="${PORT:-30000}"
PY="${PY:-/root/venvs/peek-sglang/bin/python}"
BENCH="$REPO_ROOT/scripts/bench/bench_shared_prompts.py"
SITECUSTOMIZE_DIR="$REPO_ROOT/scripts/peek_sitecustomize"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME"

OUT_DIR="$REPO_ROOT/benchmarks/w2/results_probe"
RATES="${RATES:-0.15 0.22 0.28}"
N="${N:-300}"
WARMUP="${WARMUP:-50}"
SEED="${SEED:-42}"
# Cell B shape, matching run_w2_sglang.sh
G=40
PREFIX=8192
DECODE_MIX="10:128, 25:512, 30:1024, 25:2048, 10:4096"
CONCURRENCY=256

mkdir -p "$OUT_DIR"
SLOG="$OUT_DIR/_server_lpm_lru.log"

echo "[probe] rates=$RATES  N=$N  warmup=$WARMUP  seed=$SEED  -> $OUT_DIR"

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 5

# Stock SGLang LPM + LRU: no PEEK_* env, exactly as the matrix runs lpm_lru.
env PYTHONPATH="$SITECUSTOMIZE_DIR:${PYTHONPATH:-}" \
  "$PY" -m sglang.launch_server \
    --model "$MODEL" \
    --mem-fraction-static "$MEM_FRAC" \
    --schedule-policy lpm \
    --enable-cache-report --enable-metrics \
    --host 127.0.0.1 --port "$PORT" \
    --log-level warning \
    > "$SLOG" 2>&1 &
SPID=$!

echo "[probe] waiting for server (pid $SPID)..."
for i in $(seq 1 600); do
  sleep 3
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { echo "[probe] ready after $((i*3))s"; break; }
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "[probe] server died; tail:"; tail -20 "$SLOG"; exit 1
  fi
done

for rate in $RATES; do
  out="$OUT_DIR/rate_${rate}.json"
  if [[ -f "$out" ]]; then echo "[probe] skip $out (exists)"; continue; fi
  echo
  echo "----- probe rate=$rate (cell B, lpm_lru, N=$N seed=$SEED) -----"
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 2
  "$PY" "$BENCH" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --n "$N" --groups "$G" --prefix-tokens "$PREFIX" \
    --decode-mix "$DECODE_MIX" \
    --rate "$rate" --concurrency "$CONCURRENCY" --seed "$SEED" \
    --warmup-reqs "$WARMUP" \
    --ttft-slo-ms 2000 --tpot-slo-ms 100 --e2e-slo-ms 180000 \
    --dataset repobench --distribution zipf --zipf-alpha 1.0 \
    --label "probe_rate_${rate}" \
    --output "$out" --save-per-request \
    > "$OUT_DIR/_run_rate_${rate}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[probe] BENCH FAILED rc=$rc, tail:"; tail -30 "$OUT_DIR/_run_rate_${rate}.log"
  else
    echo "[probe] wrote $out"
  fi
done

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
echo
echo "[probe] done."
