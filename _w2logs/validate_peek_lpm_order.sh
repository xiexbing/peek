#!/bin/bash
# Ground-truth LPM sort-order parity under REAL W2 traffic.
#
# The unit tests prove peek_lpm matches sglang's LPM sort on mock queues.
# This proves it on the live scheduler: PEEK_ONLINE_VALIDATE_LPM_ORDER=1
# re-runs sglang's own _sort_by_longest_prefix on the same (queue, deprio)
# every tick and appends a JSONL record for any tick whose order differs.
#
# No mismatch file (and no ORDER MISMATCH warning) == byte-exact parity.
#
# Cell-B shape, short run (N default 150 at 0.22 req/s ~ 11 min of arrivals)
# -- enough ticks at real queue depth to be meaningful.
# Needs an idle GPU.
set -uo pipefail

REPO_ROOT=/workspace/peek
PY=/root/venvs/peek-sglang/bin/python
PORT=30001
OUT=/workspace/peek/_w2logs/lpm_order_validation
SLOG="$OUT/server.log"
N="${N:-150}"
RATE="${RATE:-0.22}"
SEED="${SEED:-42}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME"

mkdir -p "$OUT"
rm -f /tmp/peek_lpm_order_diffs_*.json

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 6

echo "[val] launching peek_lpm with order validation (N=$N rate=$RATE)"
env PYTHONPATH="$REPO_ROOT/scripts/peek_sitecustomize:${PYTHONPATH:-}" \
  PEEK_ONLINE_LPM=1 PEEK_ONLINE_RANK_BY_SIZE=0 PEEK_ONLINE_VALIDATE_LPM_ORDER=1 \
  "$PY" -m sglang.launch_server \
    --model Qwen/Qwen2.5-32B-Instruct --mem-fraction-static 0.88 \
    --schedule-policy lpm --enable-cache-report --enable-metrics \
    --host 127.0.0.1 --port "$PORT" --log-level warning \
    > "$SLOG" 2>&1 &
spid=$!

for i in $(seq 1 400); do
  sleep 3
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { echo "[val] ready after $((i*3))s"; break; }
  kill -0 "$spid" 2>/dev/null || { echo "[val] FAIL: server died"; tail -20 "$SLOG"; exit 1; }
done

"$PY" "$REPO_ROOT/scripts/bench/bench_shared_prompts.py" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --model Qwen/Qwen2.5-32B-Instruct \
  --n "$N" --groups 40 --prefix-tokens 8192 \
  --decode-mix "10:128, 25:512, 30:1024, 25:2048, 10:4096" \
  --rate "$RATE" --concurrency 256 --seed "$SEED" --warmup-reqs 25 \
  --ttft-slo-ms 2000 --tpot-slo-ms 100 --e2e-slo-ms 180000 \
  --dataset repobench --distribution zipf --zipf-alpha 1.0 \
  --label peek_lpm_validation --output "$OUT/bench.json" \
  > "$OUT/bench.log" 2>&1
rc=$?
echo "[val] bench rc=$rc"

sleep 3
pkill -9 -f "sglang.launch_server" 2>/dev/null || true

echo
echo "================ LPM ORDER PARITY ================"
diffs=$(ls /tmp/peek_lpm_order_diffs_*.json 2>/dev/null)
warns=$(grep -c "ORDER MISMATCH" "$SLOG" 2>/dev/null || echo 0)
if [[ -z "$diffs" && "$warns" == "0" ]]; then
  echo "  PASS -- zero sort-order mismatches vs sglang's own LPM sort"
else
  echo "  FAIL -- mismatches detected (warnings in server log: $warns)"
  for f in $diffs; do
    echo "  --- $f ($(wc -l < "$f") mismatching ticks) ---"
    head -3 "$f"
    cp "$f" "$OUT/"
  done
fi
echo
echo "  validation-hook errors:"
grep -c "peek_lpm validation failed" "$SLOG" 2>/dev/null || echo "  0"
