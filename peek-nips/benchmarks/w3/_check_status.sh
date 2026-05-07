#!/bin/bash
# Quick status of the autonomous W3 reproduction chain.
echo "=== chain process alive? ==="
ps -ef | grep -E "_run_chain_seed42|run_w3_(sglang|vllm)|sglang.launch|vllm.entrypoints" | grep -v grep | head -5
echo
echo "=== chain log tail ==="
tail -20 /tmp/w3_chain.log
echo
echo "=== JSON results so far ($(find /workspace/peek/peek-nips/benchmarks/w3/results_* -name '*.json' 2>/dev/null | wc -l)/16) ==="
find /workspace/peek/peek-nips/benchmarks/w3/results_* -name "*.json" 2>/dev/null | sort
echo
echo "=== if compare ran, last 30 lines ==="
[ -s /tmp/w3_compare.log ] && tail -30 /tmp/w3_compare.log || echo "(compare hasn't run yet)"
echo
echo "=== gpu state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
