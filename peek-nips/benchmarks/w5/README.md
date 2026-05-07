# W5 -- Singleton chat (no-regress safety test)

> Paper §4.4 (Tables 19, 20). Environment: sglang 0.5.9 / vllm 0.19.1,
> torch 2.9.1, Python 3.12, 1xH100 80GB (bf16), `google/gemma-2-27b-it`.
> Dataset: LMSYS-Chat-1M (English-only, deduped); the singleton driver
> samples conversations directly from the HF dataset.

## Purpose

Prove PEEK **does not regress** on LPM-unfriendly workloads: chat traffic
where almost every request is a singleton (no prefix sharing). Every
reviewer's first skeptical question is "what's the downside?" -- W5 is
the answer.

The dynamic-lane controller (DL) and the `has_sharing` guard exist
specifically to protect singletons under low-overlap traffic. W5 tests
whether they hold up.

## Workload shape

LMSYS-Chat-1M conversations replayed as a Poisson stream. Two cells
that span the chat prompt-length distribution:

```
C_short : prompt = lognormal-truncated to 32-1024 tokens     (typical chat)
C_long  : prompt = lognormal-truncated to 512-4096 tokens    (long-form / instruction-heavy)
Decode  : 64-256 tokens (variable, sampled per request)
N       : 1500 prompts per cell
warmup  : 100 prompts excluded from metrics
Arrival : Poisson; per-cell rate set by the W5 calibration probe
```

Sharing is structurally near-zero: each LMSYS conversation is unique.
Any cache hit comes from byte-recurring tokens that the engine's hash
matcher coincidentally reuses (vLLM's APC reports this; SGLang's
RadixAttention does not, hence the much lower SGLang baseline cache hit
in Table 19).

## Production analog

- ChatGPT default (most queries have no custom GPT)
- Claude / Gemini default chat
- Open-source hosted chat (HuggingChat, LMSYS Arena)
- General-purpose inference API where tenants have diverse prompts

## Policies

The 8-policy lattice from W1/W2 is overkill here -- what matters is
headline stock-vs-PEEK parity.

| Filesystem ID    | Scheduling      | Eviction              | Role                             |
| ---------------- | --------------- | --------------------- | -------------------------------- |
| `lpm_lru`        | stock SGLang LPM | LRU                  | **SGLang baseline**              |
| `fcfs_lru`       | stock SGLang FCFS | LRU                 | scheduling-axis baseline         |
| `fcfs_apc_lru`   | stock vLLM FCFS + APC | LRU             | **vLLM baseline**                |
| `clpm_gm_dl_pe`  | cLPM + GM + DL  | queue-aware (cluster) | **paper-primary, full PEEK**     |

Two claims:

1. **`clpm_gm_dl_pe` ≈ baseline on `C_short`** (every metric within
   noise; almost no opportunity to either improve or hurt).
2. **`clpm_gm_dl_pe` ≥ baseline on `C_long`** (cluster-aware sort
   detects weak overlap that LPM's exact-prefix tiebreak misses, so
   PEEK gets a small but consistent win without regressing TPOT).

## Drivers

```bash
bash benchmarks/w5/run_w5_sglang.sh           # full matrix, 3 seeds
bash benchmarks/w5/run_w5_vllm.sh             # full matrix, 3 seeds
```

Subsets: `POLICIES=lpm_lru CELLS=C_short SEEDS=42 RATES=moderate bash run_w5_sglang.sh`.

## Cells and rates

| cell      | prompt range (tok) | decode range (tok) | moderate (req/s) | heavy (req/s) |
| --------- | ------------------ | ------------------ | ---------------- | ------------- |
| `C_short` | 32 - 1024          | 64 - 256           | 102              | 120           |
| `C_long`  | 512 - 4096         | 64 - 256           | 15               | 25            |

Rates are calibrated from the W5 probe (`r_sat` with `errored=0` and
`slo_attainment_pct ≥ 90%`; `moderate = 0.85 x r_sat`, `heavy ≈ r_sat`).
Override via `RATES=...`.

## Seeds

`42, 142, 242`. Override via `SEEDS=...`.

> **Note (paper Table 19 caption):** the published SGLang `C_long` row
> averages **seed 242 only** because the other two seeds saturated below
> the calibrated rate; `compare_to_paper.py` honours that seed selection
> when verifying SGLang `C_long`. The paper's vLLM `C_long` row and both
> `C_short` rows use 3-seed means.

## Aggregation

```bash
python3 benchmarks/w5/aggregate.py            # summary.csv + aggregated.csv + delta.csv
python3 benchmarks/w5/compare_to_paper.py     # vs Tables 19 (SGLang) and 20 (vLLM)
```

`compare_to_paper.py` returns exit 0 if every observed metric is within
the configurable tolerance (default ±20%) of the paper headline value.

## Pre-registered predictions (heavy load)

| metric                     | C_short                     | C_long                      |
| -------------------------- | --------------------------- | --------------------------- |
| Aggregate throughput       | within ±3%                  | within ±3% (or PEEK higher) |
| Mean TTFT                  | within ±5%                  | PEEK 5-15% lower            |
| Mean TPOT                  | within ±2%                  | within ±2%                  |
| Cache hit (SGLang)         | within ±2 pp                | PEEK ≥ baseline             |

## Falsification

- If **`clpm_gm_dl_pe` regresses beyond ±3% on aggregate throughput
  or mean TPOT in either cell**, the dynamic-lane controller is not
  protecting singletons under low-overlap traffic -- the paper's
  "general-purpose safe-default" claim has to be narrowed to
  shared-sharing-heavy deployments.
- If **`clpm_gm_dl_pe` ≈ baseline on both cells with no improvement
  at all on `C_long`**, cluster-aware sort fails to detect the weak
  overlap that the paper claims it captures -- W5 still passes the
  safety bar but loses its positive sub-claim.

## Layout

```
benchmarks/w5/
├── README.md             (this file)
├── run_w5_sglang.sh      driver: gemma-2-27b-it on SGLang
├── run_w5_vllm.sh        driver: gemma-2-27b-it on vLLM (FCFS+APC)
├── aggregate.py          summary CSVs across seeds
├── compare_to_paper.py   verify against paper Tables 19, 20
└── results/              populated by drivers (gitignored)
    └── seed_<seed>/
        └── cell_<cell>/
            └── rate_<rate>/
                ├── <policy>.json        bench output
                ├── _run_<policy>.log    bench stdout/err
                ├── _server_<policy>.log engine stdout/err
                └── _metrics_{pre,post}_<policy>.prom (SGLang)
```
