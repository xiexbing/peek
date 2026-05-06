# W3 — Multi-GPU Llama-3.1-70B (DP=1, DP=2)

> Paper §4.3. Scales W1 (cell C, chat-like / admission-bound) and W2
> (cell B, RAG-like / decode-bound) to `meta-llama/Llama-3.1-70B-Instruct`
> at TP=2, with two data-parallel topologies:
>
> - **DP=1** — single TP=2 replica (2×H100)
> - **DP=2** — two TP=2 replicas behind sglang's prefix-aware router (4×H100)

## Cells

| Cell | Source workload    | Prompt structure                          | Decode      |
| ---- | ------------------ | ----------------------------------------- | ----------- |
| **B** | W2 (long-doc RAG) | G=14, prefix=4096, Zipf-α=1.0             | mix(128, 512, 1024, 2048, 4096) |
| **C** | W1 (chat)         | G=88, prefix=1500, Zipf-α=1.0             | fixed 128   |

Cells inherit per-prompt parameters from W1/W2; the only differences in
W3 are model size and parallelism.

## Drivers

| Topology       | SGLang                                                  | vLLM                          |
| -------------- | ------------------------------------------------------- | ----------------------------- |
| **DP=1** (2 GPUs) | `MODEL=meta-llama/Llama-3.1-70B-Instruct TP=2 bash benchmarks/w1/run_w1_sglang.sh` (cell C) <br>or `… benchmarks/w2/run_w2_sglang.sh` (cell B) | `bash run_w3_vllm.sh` |
| **DP=2** (4 GPUs) | `bash run_w3_sglang_dp2.sh`                             | `bash run_w3_vllm_dp2.sh`     |

> The DP=1 SGLang case is reachable directly through the W1/W2 drivers
> by overriding `MODEL` and `TP` — both scripts honor those env vars.
> No dedicated `run_w3_sglang.sh` is shipped because the workload is
> structurally identical to W1/W2 with two parameters changed.

## Policies

Same paper-canonical labels as W1 (Table 2). Defaults run the strongest
stock baseline + the full PEEK config:

- SGLang: `lpm_lru` (baseline) and `clpm_gm_dl_pe` (PEEK)
- vLLM: `fcfs_apc_lru` (baseline) and `clpm_gm_dl_pe` (PEEK)

## Hardware

- DP=1: 2×H100 80GB
- DP=2: 4×H100 80GB

Set `mem_fraction_static=0.88` (SGLang) / `gpu_memory_utilization=0.9` (vLLM).

## Seeds

42, 142, 242 (paper default). Override via `SEEDS=…`.
