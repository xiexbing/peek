# W2 — Long-document RAG (decode-dominant co-design)

> Paper §4.2. Environment: sglang 0.5.9 / vllm 0.19.1, torch 2.9.1,
> Python 3.12, 1×H100 80GB (bf16), `Qwen/Qwen2.5-32B-Instruct`.

## Purpose

Test PEEK's **scheduling + eviction co-design** under the hardest
production regime: long prefixes (documents) + long variable decodes
+ KV arena structurally overflowed. This is where co-design's
super-additivity is strongest.

## Workload shape

```
G        = 40 documents × 8192-token prefix
Queries  : 256-token chat-over-doc, 1 query per request
Sampling : Zipf α=1.0 (hot documents dominate)
Decode   : DECODE_MIX = "10:128, 25:512, 30:1024, 25:2048, 10:4096"
           (mean ≈ 1460 tokens — covers structured short responses,
            normal RAG answers, long-form RAG/code, synthesis,
            reasoning-adjacent)
mem_frac : 0.88 (production-realistic)
KV footprint : 328K prefix tokens ≈ 7× arena (eviction always firing);
               active decode KV grows linearly with decode length
N        : 500
```

## Cells

Per paper Tables 6, 13–14 — moderate and heavy load on the same workload
shape, sampled to hit two different sustained queue depths.

## Policies

Paper-canonical labels (Table 2):

| Label                  | Scheduling                            | Eviction       |
| ---------------------- | ------------------------------------- | -------------- |
| `lpm_lru` (sglang)     | stock SGLang LPM                      | LRU            |
| `fcfs_lru` (sglang)    | stock SGLang FCFS                     | LRU            |
| `fcfs_apc_lru` (vllm)  | stock vLLM FCFS + APC                 | LRU            |
| `lpm_pe`               | LPM                                   | queue-aware    |
| `clpm`                 | cLPM (cluster-aware sort) + LRU       | LRU            |
| `clpm_gm`              | cLPM + group-major                    | LRU            |
| `clpm_gm_dl`           | cLPM + GM + dynamic-lane              | LRU            |
| `clpm_gm_pe`           | cLPM + GM                             | queue-aware    |
| `clpm_gm_dl_pe`        | cLPM + GM + DL                        | queue-aware    |

The paper's primary configuration is `clpm_gm_dl_pe`.

## Drivers

```bash
bash benchmarks/w2/run_w2_sglang.sh           # full matrix, 3 seeds
# (vllm side: see paper §4.2 — uses the same client + a vllm server;
#  reachable via the W1 vllm driver with W2 cell parameters, or write
#  a thin wrapper if you need a dedicated entry point.)
```

Subsets: `POLICIES=lpm_lru CELLS=B SEEDS=42 RATES=heavy bash run_w2_sglang.sh`.

## Seeds

42, 142, 242. Override via `SEEDS=…`.
