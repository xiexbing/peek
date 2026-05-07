# W4 data

The W4 driver expects two files in this directory; the supplemental
archive ships the original Apache-2.0 prompt but **not** the Mooncake
trace itself (it is third-party data; reviewers fetch and filter it).

| File                              | Source                                                                      | Status        |
| --------------------------------- | --------------------------------------------------------------------------- | ------------- |
| `shared_system_prompt.txt`        | Original to PEEK (Apache 2.0). 1402-token agentic system prompt.            | **bundled**   |
| `conversation_trace_le6k.jsonl`   | Mooncake `conversation_trace.jsonl` filtered to sessions with cumulative tokens <=6k. | **fetch+filter** |

## Fetch + filter `conversation_trace_le6k.jsonl`

The full trace (~5x larger than the filtered subset, no PEEK-vs-baseline
ratio change) is the `conversation_trace.jsonl` from the FAST'25
Mooncake release: `https://github.com/kvcache-ai/Mooncake`. Drop it
into this directory and filter:

```bash
cd benchmarks/w4/data

# 1. Fetch the upstream trace (path may vary across Mooncake releases;
#    check the upstream README for the current file location)
curl -L -O https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/mooncake_trace/conversation_trace.jsonl

# 2. Filter to sessions whose cumulative input+output token count is <=6k
python3 - <<'PY'
import json
src = "conversation_trace.jsonl"
dst = "conversation_trace_le6k.jsonl"
kept = total = 0
with open(src) as f_in, open(dst, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        total += 1
        # Mooncake records are session-level with per-turn input/output token counts.
        # Sum input+output across all turns in the session.
        cum = 0
        for turn in rec.get("turns", []):
            cum += int(turn.get("input_length", 0)) + int(turn.get("output_length", 0))
        if cum <= 6000:
            f_out.write(line + "\n")
            kept += 1
print(f"kept {kept}/{total} sessions (cumulative tokens <=6000)")
PY
```

The W4 drivers (`run_w4_sglang.sh`, `run_w4_vllm.sh`) hardcode the path
`benchmarks/w4/data/conversation_trace_le6k.jsonl`. Override with
`MOONCAKE_PATH=...` if you keep the file elsewhere.

## License

Mooncake `conversation_trace` is governed by the Mooncake repo's
license (see upstream README). `shared_system_prompt.txt` is original
to PEEK and is licensed under Apache 2.0 with the rest of this repo.
