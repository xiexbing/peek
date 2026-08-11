#!/bin/bash
# Smoke-test that clpm_gm_dl_pe actually installs PEEK's hooks.
#
# patch_hook.py catches integration errors and "continues as vanilla" (line
# ~880), so a broken hook yields a server that runs fine and produces
# baseline-identical numbers. That failure mode is invisible in the result
# JSON. This checks the install markers in the server log directly.
set -uo pipefail

REPO_ROOT=/workspace/peek
PY=/root/venvs/peek-sglang/bin/python
PORT=30001
SLOG=/workspace/peek/_w2logs/smoke_peek_server.log
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME"

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 5

echo "[smoke] launching clpm_gm_dl_pe on port $PORT"
env PYTHONPATH="$REPO_ROOT/scripts/peek_sitecustomize:${PYTHONPATH:-}" \
  PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 \
  PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster \
  "$PY" -m sglang.launch_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --mem-fraction-static 0.88 \
    --schedule-policy lpm \
    --enable-cache-report --enable-metrics \
    --host 127.0.0.1 --port "$PORT" \
    --log-level warning \
    > "$SLOG" 2>&1 &
SPID=$!

for i in $(seq 1 400); do
  sleep 3
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { echo "[smoke] ready after $((i*3))s"; break; }
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "[smoke] FAIL: server died"; tail -30 "$SLOG"; exit 1
  fi
done

# Drive a few requests so the scheduler hook actually executes.
echo "[smoke] sending 3 requests..."
for i in 1 2 3; do
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-32B-Instruct","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24}' \
    -o /dev/null -w "  req '"$i"' http=%{http_code}\n"
done
sleep 3

echo
echo "===== PEEK install markers ====="
fail=0
for marker in "installed hooks on RadixCache" "installed scheduler" "PeekDemandStrategy"; do
  if grep -q "$marker" "$SLOG"; then
    echo "  OK    : $marker"
  else
    echo "  MISSING: $marker"
    fail=1
  fi
done
echo
echo "===== integration failures ====="
if grep -qE "integration failed|falling back|validation failed" "$SLOG"; then
  grep -nE "integration failed|falling back|validation failed" "$SLOG" | head -10
  fail=1
else
  echo "  none"
fi
echo
echo "===== peek log lines ====="
grep -n "peek" "$SLOG" | head -20

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
echo
if [[ $fail == 0 ]]; then echo "[smoke] PASS -- PEEK hooks active"; else echo "[smoke] FAIL -- see above"; fi
exit $fail
