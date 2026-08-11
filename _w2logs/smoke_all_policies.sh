#!/bin/bash
# Boot-smoke every PEEK policy in the W2 lattice.
#
# patch_hook.py swallows integration errors and continues as vanilla sglang,
# so a broken policy produces a healthy server with silently-inert hooks.
# This boots each policy, drives real requests, and asserts the expected
# install markers are present AND that markers which should NOT be there are
# absent (e.g. lpm_pe must install eviction but NOT the scheduler -- a policy
# that quietly turns on the scheduler is a config bug that would corrupt the
# ablation ladder).
#
# Needs an idle GPU: each boot allocates ~74GB. Run only when no bench is up.
set -uo pipefail

REPO_ROOT=/workspace/peek
PY=/root/venvs/peek-sglang/bin/python
PORT=30001
LOGDIR=/workspace/peek/_w2logs/smoke_policies
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$LOGDIR"

RADIX="installed hooks on RadixCache"
SCHED="installed scheduler"
EVICT="PeekDemandStrategy"

# Keep in sync with policy_env() in benchmarks/w2/run_w2_sglang.sh.
# Format: policy|env|expected-present(comma)|expected-absent(comma)
POLICIES=(
"lpm_pe|PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster|RADIX,EVICT|SCHED"
"peek_lpm|PEEK_ONLINE_LPM=1 PEEK_ONLINE_RANK_BY_SIZE=0|RADIX,SCHED|EVICT"
"clpm|PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1|RADIX,SCHED|EVICT"
"clpm_gm|PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1|RADIX,SCHED|EVICT"
"clpm_gm_dl|PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1|RADIX,SCHED|EVICT"
"clpm_gm_pe|PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster|RADIX,SCHED,EVICT|"
"clpm_gm_dl_pe|PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster|RADIX,SCHED,EVICT|"
)

marker_of() { case "$1" in RADIX) echo "$RADIX";; SCHED) echo "$SCHED";; EVICT) echo "$EVICT";; esac; }

overall=0
declare -a SUMMARY

for spec in "${POLICIES[@]}"; do
  IFS='|' read -r pol penv want notwant <<< "$spec"
  slog="$LOGDIR/${pol}.log"
  echo
  echo "=============== $pol ==============="
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  sleep 6

  env PYTHONPATH="$REPO_ROOT/scripts/peek_sitecustomize:${PYTHONPATH:-}" $penv \
    "$PY" -m sglang.launch_server \
      --model Qwen/Qwen2.5-32B-Instruct --mem-fraction-static 0.88 \
      --schedule-policy lpm --enable-cache-report --enable-metrics \
      --host 127.0.0.1 --port "$PORT" --log-level warning \
      > "$slog" 2>&1 &
  spid=$!

  ready=0
  for i in $(seq 1 400); do
    sleep 3
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    [[ "$code" == "200" ]] && { ready=1; echo "  ready after $((i*3))s"; break; }
    kill -0 "$spid" 2>/dev/null || { echo "  FAIL: server died"; tail -20 "$slog"; break; }
  done
  if [[ $ready == 0 ]]; then SUMMARY+=("$pol  FAIL(boot)"); overall=1; continue; fi

  for i in 1 2 3; do
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen2.5-32B-Instruct","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24}' \
      -o /dev/null -w "  req $i http=%{http_code}\n"
  done
  sleep 3

  fail=0
  IFS=',' read -ra W <<< "$want"
  for m in "${W[@]}"; do
    [[ -z "$m" ]] && continue
    if grep -q "$(marker_of "$m")" "$slog"; then echo "  OK     present: $m"
    else echo "  MISSING present: $m"; fail=1; fi
  done
  IFS=',' read -ra NW <<< "$notwant"
  for m in "${NW[@]}"; do
    [[ -z "$m" ]] && continue
    if grep -q "$(marker_of "$m")" "$slog"; then echo "  UNEXPECTED: $m should be absent"; fail=1
    else echo "  OK     absent : $m"; fi
  done
  if grep -qE "integration failed|falling back|validation failed" "$slog"; then
    echo "  FAIL: integration errors:"; grep -nE "integration failed|falling back|validation failed" "$slog" | head -5
    fail=1
  fi

  [[ $fail == 0 ]] && SUMMARY+=("$pol  PASS") || { SUMMARY+=("$pol  FAIL"); overall=1; }
done

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
echo
echo "================ SUMMARY ================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
[[ $overall == 0 ]] && echo "ALL POLICIES PASS" || echo "SOME POLICIES FAILED"
exit $overall
