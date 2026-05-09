# PEEK reproduction kit (W1-W5)

Per-workload drivers for the experiments reported in PEEK-Online §4
(NeurIPS 2026). Each `wN/` directory has its own README with cell
parameters, expected runtimes, and worked examples.

| W   | §  | Workload                              | Model                                   | Hardware           |
| --- | -- | ------------------------------------- | --------------------------------------- | ------------------ |
| W1  | 4.1 | Shared-prompt chat (LooGLE)          | `Qwen/Qwen2.5-32B-Instruct`             | 1xH100 80GB        |
| W2  | 4.2 | Long-document RAG (LooGLE)           | `Qwen/Qwen2.5-32B-Instruct`             | 1xH100 80GB        |
| W3  | 4.3 | Multi-GPU 70B (DP=1, DP=2)           | `meta-llama/Llama-3.1-70B-Instruct` TP=2 | 2x / 4xH100 80GB   |
| W4  | 4.4 | Agentic Mooncake conversation_trace  | `mistralai/Mistral-Small-24B-Instruct-2501` | 1xH100 80GB    |
| W5  | 4.4 | Singleton chat (LMSYS-Chat-1M)       | `google/gemma-2-27b-it`                 | 1xH100 80GB        |

## Policy labels

Paper §4 Table 2. The driver scripts and READMEs use the same
filesystem-safe IDs throughout.

| Paper label             | Filesystem ID         | Scheduling                                          | Eviction              | Role                                  |
| ----------------------- | --------------------- | --------------------------------------------------- | --------------------- | ------------------------------------- |
| FCFS+LRU                | `fcfs_lru`            | stock SGLang FCFS                                   | LRU                   | naïve baseline                        |
| LPM+LRU                 | `lpm_lru`             | stock SGLang LPM                                    | LRU                   | **SGLang baseline**                   |
| FCFS(APC)+LRU           | `fcfs_apc_lru`        | stock vLLM FCFS + APC                               | LRU                   | **vLLM baseline**                     |
| LPM+PE                  | `lpm_pe`              | stock SGLang LPM                                    | queue-aware (cluster) | eviction-only ablation                |
| FCFS(APC)+PE            | `fcfs_apc_pe`         | stock vLLM FCFS + APC                               | queue-aware (cluster) | eviction-only ablation                |
| cLPM                    | `clpm`                | cLPM (cluster-aware sort)                           | LRU                   | sort-key ablation                     |
| cLPM+GM                 | `clpm_gm`             | cLPM + group-major                                  | LRU                   | + sibling batching                    |
| cLPM+GM+DL              | `clpm_gm_dl`          | cLPM + GM + dynamic-lane                            | LRU                   | + fairness                            |
| cLPM+GM+PE              | `clpm_gm_pe`          | cLPM + GM                                           | queue-aware (cluster) | primary co-design                     |
| **cLPM+GM+DL+PE**       | **`clpm_gm_dl_pe`**   | cLPM + GM + DL                                      | queue-aware (cluster) | **paper-primary, full PEEK**          |

## Environment-flag mapping

The filesystem-safe IDs map to `PEEK_ONLINE_*` flags as follows:

| Filesystem ID    | Env flags                                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `fcfs_lru`       | _none_ (server: `--schedule-policy fcfs` on SGLang, no APC on vLLM)                                                          |
| `lpm_lru`        | _none_ (server: `--schedule-policy lpm` on SGLang)                                                                           |
| `fcfs_apc_lru`   | _none_ (server: `--enable-prefix-caching` on vLLM)                                                                           |
| `lpm_pe`         | `PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster`                                                                   |
| `fcfs_apc_pe`    | `PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster` (vLLM)                                                            |
| `clpm`           | `PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1`                                                                                 |
| `clpm_gm`        | + `PEEK_ONLINE_CLPM_GROUP_MAJOR=1`                                                                                           |
| `clpm_gm_dl`     | + `PEEK_ONLINE_CLPM_DYNAMIC_LANE=1`                                                                                          |
| `clpm_gm_pe`     | `PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster` |
| `clpm_gm_dl_pe`  | + `PEEK_ONLINE_CLPM_DYNAMIC_LANE=1`                                                                                          |

The drivers do this mapping automatically -- you set `POLICIES=...` and they
emit the right env vars per policy.

## Bench clients

Shared by multiple workloads, parked under `scripts/bench/`:

| Client                            | Used by    | What it generates                                                |
| --------------------------------- | ---------- | ---------------------------------------------------------------- |
| `bench_shared_prompts.py`         | W1, W2, W3 | Poisson-arrival shared-system-prompt traffic with Zipf-weighted groups |
| `bench_lmsys_singleton.py`        | W5         | Singleton chat from LMSYS-Chat-1M (English-only, deduped)        |
| `bench_mooncake_vllm.py` (in `w4/`) | W4 (vLLM) | Mooncake `conversation_trace` replay with optional shared prompt |
| `sglang.bench_serving` (patched)  | W4 (SGLang) | Same trace via the public `patches/sglang/bench_serving.patch`   |

## Quick start

```bash
# 1. Build PEEK and install the engine you want to test (sglang OR vllm --
#    incompatible torch pins, see top-level README).
bash scripts/install_peek_sglang.sh
# (or scripts/install_peek_vllm.sh)

# 2. Run a smoke test -- W1 cell C, primary cell, one seed, two policies.
CELLS=C SEEDS=42 RATES=heavy \
POLICIES_FULL="lpm_lru clpm_gm_dl_pe" \
bash benchmarks/w1/run_w1_sglang.sh

# 3. Aggregate
python benchmarks/w1/aggregate.py
```

## Common knobs (any driver)

| Variable                | Default                                  | Purpose                                            |
| ----------------------- | ---------------------------------------- | -------------------------------------------------- |
| `MODEL`                 | per-W default (table above)              | HuggingFace model id                               |
| `MEM_FRAC` (sglang)     | 0.88                                     | `--mem-fraction-static`                            |
| `GPU_UTIL` (vllm)       | 0.9                                      | `--gpu-memory-utilization`                         |
| `TP`                    | 1 (W1/W2/W4/W5), 2 (W3)                  | tensor-parallel size                               |
| `SEEDS`                 | `42 142 242`                             | per paper §4                                       |
| `CELLS`                 | per-W default                            | which cell(s) to run                               |
| `RATES`                 | `moderate heavy`                         | load levels                                        |
| `POLICIES`              | per-W default                            | comma-or-space list of filesystem IDs              |
| `RESULTS_DIR`           | `<W>/results/`                           | where JSON results land                            |
| `HF_HOME`               | `$HOME/.cache/huggingface`               | HuggingFace cache (override for shared mounts)     |
| `SERVER_READY_TIMEOUT_S`| 1800                                     | 30 min model load timeout                          |

## Seeds, warmup, statistical reporting

All paper numbers are **means across the 3 seeds (42, 142, 242)** after
the first ~10-20% of requests are warmup-excluded. Per-seed JSON is in
`<W>/results/seed_<N>/cell_<X>/rate_<Y>/<policy>.json`. The workload-
specific `aggregate.py` consolidates seeds and emits CSV summaries +
per-policy delta vs the engine's stock baseline.

## What's not shipped

- **W3 SGLang DP=1**: reachable via `MODEL=meta-llama/Llama-3.1-70B-Instruct
  TP=2 bash benchmarks/w1/run_w1_sglang.sh` (chat, paper W3 cell C) or
  similar with `run_w2_sglang.sh` (RAG, cell B). No standalone driver
  because the workload is structurally identical to W1/W2 with two
  parameters changed.
- **W2 vLLM dedicated driver**: paper W2 vLLM results were obtained by
  running the W1 vLLM driver with W2's cell parameters. Cell-parameter
  table is in `w2/README.md`; tweak as needed when invoking
  `run_w1_vllm.sh`.
- Calibration / probe scripts (`probe_*.sh`, `calibrate.sh`) used during
  paper development -- these are noise for reviewers and were dropped.
  The cell rates baked into the run scripts are the calibrated values.
