# Reproduction environment (W1-W5)

The exact hardware/software configuration the paper numbers were collected
on. Per-workload READMEs reference this file for the full spec and the
rate-calibration procedure.

## Hardware

| Workload | GPUs                                      | Notes                          |
| -------- | ----------------------------------------- | ------------------------------ |
| W1       | 1xH100 80GB SXM (bf16)                    |                                |
| W2       | 1xH100 80GB SXM (bf16)                    |                                |
| W3       | 2xH100 80GB (TP=2, DP=1) or 4xH100 (DP=2) | NVLink within node             |
| W4       | 1xH100 80GB SXM (bf16)                    |                                |
| W5       | 1xH100 80GB SXM (bf16)                    |                                |

CPU / RAM / disk are not load-bearing for any of the metrics reported in
the paper -- any modern server-class host with >=128 GB RAM and an NVMe
scratch disk for the HuggingFace cache is sufficient. CUDA 12.x driver.

## Software pins

| component        | version           | source                                |
| ---------------- | ----------------- | ------------------------------------- |
| Python           | 3.12.13           | `>=3.9` works for the runtime; 3.12 is what we tested |
| PyTorch          | 2.9.1             | pulled in by SGLang/vLLM pins         |
| SGLang           | 0.5.9             | `pip install "sglang[all]==0.5.9"`    |
| vLLM             | 0.19.1            | `pip install vllm==0.19.1`            |
| CUDA             | 12.x runtime      | matches the torch 2.9.1 build         |
| Rust toolchain   | stable (1.78+)    | for building the PEEK extension       |
| maturin          | latest stable     | `pip install maturin`                 |
| OS               | Linux x86_64      | tested on Ubuntu 22.04 / 24.04        |

> SGLang and vLLM pin **incompatible torch versions** and cannot share one
> Python environment. Use `scripts/install_peek_sglang.sh` and
> `scripts/install_peek_vllm.sh` in separate envs.

## Engine launch parameters

| component                | value                                                          |
| ------------------------ | -------------------------------------------------------------- |
| SGLang memory            | `--mem-fraction-static 0.88`                                   |
| vLLM memory              | `--gpu-memory-utilization 0.9`                                 |
| CUDA graphs              | **on** (production-realistic)                                  |
| Seeds                    | 42, 142, 242 (per paper §4)                                    |
| Warmup                   | First ~10-20 % of requests excluded from metric aggregation    |

## Rate calibration (moderate / heavy per cell)

Every cell defines two load points -- *moderate* and *heavy* -- relative
to a per-(engine, cell) **saturation rate** `r_sat`.

**Definition of `r_sat`**: the highest sustained arrival rate (req/s) at
which the stock engine baseline (`lpm_lru` for SGLang, `fcfs_apc_lru` for
vLLM) produces:

- `errored == 0` (no dropped requests in the run)
- `slo_attainment_pct >= 90%` against the workload's SLO target

**Derived load levels**:

- `moderate = 0.4 x r_sat`
- `heavy    = 0.8 x r_sat`

**How to recalibrate** for your hardware (your `r_sat` will differ if you
are not on a 1xH100 80GB):

1. Pick the cell you care about (e.g. W1 cell C).
2. Run the stock baseline at increasing rates (binary search works fine):
   ```bash
   CELLS=C SEEDS=42 RATES=heavy POLICIES_FULL="lpm_lru" \
   bash benchmarks/w1/run_w1_sglang.sh
   ```
3. Inspect the per-run JSON for `errored` and `slo_attainment_pct`. The
   highest rate that satisfies both constraints is `r_sat` for that cell.
4. Substitute `0.4 x r_sat` and `0.8 x r_sat` into the per-cell rate
   table in the workload's README, or override at the command line via
   the rate knobs documented there.

The rates baked into the run scripts are the calibrated values for
**1xH100 80GB SXM, bf16, mem_fraction=0.88, CUDA graphs on**. If your
hardware differs (different SKU, different memory budget, A100, etc.),
recalibrate before claiming reproduction.

### Why this protocol

A fixed rate-per-second is not portable across engines or hardware; what
*is* portable is "loaded to ~80% of where the stock baseline starts to
miss its SLO." Anchoring both *moderate* and *heavy* to per-cell `r_sat`
keeps the comparison fair when peek is evaluated against a different
stock baseline (SGLang LPM+LRU vs vLLM FCFS+APC) on the same workload.
