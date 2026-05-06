# W4 — Agentic conversation bursts (Mooncake `conversation_trace`)

> Paper §4.4 (`agentic_only` and `agentic_shared`). Environment:
> sglang 0.5.9 / vllm 0.19.1, torch 2.9.1, Python 3.12, 1×H100 80GB (bf16),
> `mistralai/Mistral-Small-24B-Instruct-2501`.

## Purpose

W4 establishes PEEK's **no-regression** claim on already-prefix-coherent
traffic. Each "session" is a 4-round burst with a tight inter-turn gap
(LogNormal median 50 ms, p99 ≤ 200 ms), modelling Cursor / Copilot /
Claude Code-style tool chains where adjacent calls naturally share prefix
by arrival order.

Two workload variants:

1. **`agentic_only`** — Mooncake `conversation_trace` (FAST'25) with
   within-session prefix accumulation; cross-session sharing comes only
   from hash_id overlap in the trace.
2. **`agentic_shared`** — same trace plus a 1402-token shared system
   prompt prepended to every session (modelling tool-definitions /
   agent-instructions blocks that real agents reuse across sessions).

## Cells

| Cell      | Sessions | Mean queue / peak                                 |
| --------- | -------- | ------------------------------------------------- |
| moderate  | 30       | ~38 / ~70  (below SGLang LPM-128 fallback)        |
| heavy     | 60       | ~80 / ~150 (above LPM-128 fallback; FCFS regime)  |

## Policies

Paper-canonical labels (Table 2):

| Label                    | Scheduling                                           | Eviction          |
| ------------------------ | ---------------------------------------------------- | ----------------- |
| `lpm_lru` (sglang)       | stock SGLang LPM                                     | LRU               |
| `fcfs_lru` (sglang)      | stock SGLang FCFS                                    | LRU               |
| `fcfs_apc_lru` (vllm)    | stock vLLM FCFS + APC                                | LRU               |
| `clpm_gm_dl` (peek-only) | cLPM + group-major + dynamic-lane                    | LRU               |
| `clpm_gm_dl_pe` (peek)   | cLPM + group-major + dynamic-lane                    | queue-aware (cluster) |

## Drivers

```bash
bash benchmarks/w4/run_w4_sglang.sh           # full matrix, 3 seeds
bash benchmarks/w4/run_w4_vllm.sh             # full matrix, 3 seeds
```

Subsets: `POLICIES=lpm_lru SEEDS=42 CELLS=moderate bash run_w4_sglang.sh`.

## Data

`run_w4_sglang.sh` expects the Mooncake conversation trace at
`benchmarks/w4/data/conversation_trace_le6k.jsonl` and the optional
shared system prompt at `benchmarks/w4/data/shared_system_prompt.txt`.
See `benchmarks/w4/data/README.md` (or the Mooncake repo) for fetch
instructions.

## Seeds

42, 142, 242. Override via `SEEDS=…`.
