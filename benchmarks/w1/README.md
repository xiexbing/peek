# W1 — Shared-prompt co-design across oversubscription

> Environment: sglang 0.5.9 / vllm 0.19.1, torch 2.9.1, Python 3.12,
> 1×H100 80GB (bf16). Full spec at `benchmarks/ENVIRONMENT.md`.
> sglang driver: this README. vllm driver: `README_vllm.md`.
> Rate calibration: per-engine, per-cell. r_sat = highest rate with
> `errored=0` AND `slo_attainment_pct >= 90%`; `moderate = 0.4 × r_sat`, `heavy = 0.8 × r_sat`. See
> `benchmarks/ENVIRONMENT.md#rate-calibration-moderate--heavy-per-cell`.


## Purpose

One workload; two experimental jobs:

1. **Primary performance comparison** — peek co-design (clpm_gm_pe) vs baselines (lpm_lru stock
   SGLang, fcfs_lru SGLang-FCFS, fcfs_apc_lru vLLM [external]).
2. **Component ablation** — decompose peek's win into eviction (lpm_lru→lpm_pe),
   scheduling stages (lpm_lru→clpm→clpm_gm→clpm_gm_dl), and co-design (clpm_gm→clpm_gm_pe, clpm_gm_dl→clpm_gm_dl_pe).

Plus an **oversubscription-sensitivity trend** across four cells (2×, 4×, 8×, 16×)
that shows how peek's advantage scales with KV pressure.

## Matrix

### Cells (KV oversubscription relative to arena ~50K tokens on Qwen-32B / H100 / mem_frac=0.88)

| cell | G   | prefix | N    | warmup | KV footprint | oversub |
| ---- | --- | ------ | ---- | ------ | ------------ | ------- |
| A    | 100 | 1024   | 1000 | 200    | 102 K        | 2×      |
| B    | 200 | 1024   | 2000 | 400    | 205 K        | 4×      |
| C    | 100 | 4096   | 1000 | 200    | 410 K        | 8× (PRIMARY) |
| D    | 200 | 4096   | 2000 | 400    | 820 K        | 16× (extreme) |

### Rates per cell (req/s; planning estimates — recalibrate from r_sat probe)

| cell | moderate | heavy |
| ---- | -------- | ----- |
| A    | 8        | 16    |
| B    | 6        | 12    |
| C    | 3        | 6     |
| D    | 2        | 4     |

Arrival is **Poisson** (inter-arrival = `Exp(rate)`). Group assignment is
**Zipf α=1.0** (canonical). Decode is **fixed `max_tokens=128`**.

**CUDA graphs are ENABLED** (production-realistic configuration) for all W1
cells. To reproduce a graphs-off run for debugging, set
`DISABLE_CUDA_GRAPH=1` in the environment. Graphs-off should only be used
to rule out peek–graph-capture interaction issues; main-table results must
use graphs-on.

### Policies

| Label | scheduler                    | eviction          | tests                                  |
| -- | ---------------------------- | ----------------- | -------------------------------------- |
| lpm_lru | SGLang LPM                   | LRU               | baseline                               |
| fcfs_lru | SGLang FCFS                  | LRU               | scheduling-axis baseline               |
| lpm_pe | SGLang LPM                   | peek queue-aware (cluster mode) | eviction-only ablation               |
| clpm | cLPM (peek scheduler) (per-req, 0.7)     | LRU               | scheduling stage 1                     |
| clpm_gm | cLPM + GM (peek scheduler)                 | LRU               | scheduling stage 2 (+ grouping)        |
| clpm_gm_dl | cLPM + GM + DL (peek scheduler)             | LRU               | scheduling stage 3 (+ dynamic lane)    |
| clpm_gm_pe | cLPM + GM (peek scheduler)                 | peek queue-aware (cluster mode) | **co-design (primary claim)**        |
| clpm_gm_dl_pe | cLPM + GM + DL (peek scheduler)             | peek queue-aware (cluster mode) | co-design + fairness                 |

**fcfs_apc_lru (vLLM)** and **superkv (SuperKV)** require separate harnesses and are not
handled by `run_w1_sglang.sh`. See "External baselines" below.

### Per-cell policy set

- **Cell C (PRIMARY)**: full 8-policy ablation (lpm_lru fcfs_lru lpm_pe clpm clpm_gm clpm_gm_dl clpm_gm_pe clpm_gm_dl_pe).
  Carries both the headline comparison and the full decomposition.
- **Cells A, B, D**: 6-policy subset (lpm_lru, lpm_pe, clpm_gm, clpm_gm_dl, clpm_gm_pe, clpm_gm_dl_pe).
  Used only for the oversubscription-sensitivity trend.

### Seeds

`42, 142, 242` (arbitrary fixed values; same seed applied per-policy so paired
comparisons are valid).

### Totals

- Cell C: 8 policies × 2 rates × 3 seeds = **48 runs**
- Cells A, B, D: 6 × 2 × 3 = **36 runs each = 108 runs**
- **W1 total: 156 runs** (cells C + A + B + D with their respective policy sets).

### Budget estimate

| cell | avg run time | runs | GPU-hrs |
| ---- | ------------ | ---- | ------- |
| A    | ~7 min       | 36   | ~4.2    |
| B    | ~12 min      | 36   | ~7.2    |
| C    | ~14 min      | 48   | ~11.2   |
| D    | ~20 min      | 36   | ~12.0   |
| **Total** |         | 156  | **~35 GPU-hours** |

## Running

### Full matrix (default)

```bash
bash benchmarks/w1/run_w1_sglang.sh
```

### Single cell

```bash
CELLS="C" bash benchmarks/w1/run_w1_sglang.sh
```

### Smoke test

```bash
CELLS="A" POLICIES_CORE="lpm_lru clpm_gm_pe" SEEDS="42" RATES="moderate" \
  bash benchmarks/w1/run_w1_sglang.sh
```

### Other drivers (not yet written)

- `run_w1_vllm.sh` — for the fcfs_apc_lru vLLM external baseline

### Resume after interruption

`run_w1_sglang.sh` skips any `<policy>.json` that already exists, so re-running the
same command picks up where it left off.

### Restart mode

By default the driver uses **policy-major** looping: one sglang launch per unique
policy, with `/flush_cache` between benches of the same policy. This gives
8 launches total for the full W1 matrix (one per policy in cell C's ladder).
Saves ~12 GPU-hrs vs launching per-run.

To force cold-start equivalence between every bench (e.g., for paper-ironclad
reproducibility), set `FULL_RESTART=1`:

```bash
FULL_RESTART=1 bash benchmarks/w1/run_w1_sglang.sh
```

That relaunches sglang for each of the 156 runs (adds ~12 GPU-hrs). Default
is policy-major.

## Aggregation

```bash
python3 benchmarks/w1/aggregate.py
```

Produces:

- `summary.csv` — one row per run (raw per-seed metrics)
- `aggregated.csv` — median/mean/stdev across seeds per (cell × rate × policy)
- `delta.csv` — each non-baseline policy's % improvement vs lpm_lru (positive = better)

## Expected outputs

### Primary headline (cell C, heavy rate, median-of-3-seeds)

Row: **clpm_gm_pe vs lpm_lru** → report TPOT p99 reduction, goodput increase, cache-hit
improvement. This is the paper's headline table.

### Ablation ladder (cell C, heavy rate)

Graded improvement lpm_lru → lpm_pe → clpm → clpm_gm → clpm_gm_dl → clpm_gm_pe → clpm_gm_dl_pe, showing each component's
contribution.

### Oversubscription plot

X = oversub {2, 4, 8, 16}, Y = clpm_gm_pe vs lpm_lru relative improvement (%) on primary
metrics (TPOT p99, goodput, cache hit). Expect the curve to grow with oversub,
possibly plateau or fall at 16× (pure-thrash regime).

### Super-additivity test

Per (cell × rate): does `clpm_gm_pe − lpm_lru` > `(lpm_pe − lpm_lru) + (clpm_gm − lpm_lru)`?
If yes at 4× and 8×, the co-design claim lands.

## Pre-registered predictions (TPOT p99 reduction vs lpm_lru)

| oversub | sched-only (clpm_gm) | evict-only (lpm_pe) | co-design (clpm_gm_pe) | super-additivity |
| ------- | --------------- | --------------- | -------------- | ---------------- |
| 2×      | 10–20%          | <5%             | 10–25%         | 1–5 pp           |
| 4×      | 15–30%          | 5–15%           | 20–40%         | 5–10 pp          |
| 8×      | 20–40%          | 20–40%          | 35–60%         | 10–20 pp         |
| 16×     | 20–30%          | 40–60%          | 40–70%         | 15–25 pp         |

## Falsification

- **clpm_gm ≈ clpm** → group-major adds nothing → drop grouping, simplify paper
- **clpm_gm_dl ≈ clpm_gm** → dynamic lane adds nothing → drop dyn from the claim
- **clpm_gm_pe ≈ max(lpm_pe, clpm_gm)** → no super-additivity → "two separate improvements",
  not "co-design"
- **seed stdev > mean effect** in any cell → underpowered, add seeds

## External baselines (not in `run_w1_sglang.sh`)

- **fcfs_apc_lru vLLM with `--enable-prefix-caching`**: requires separate bench harness.
  Setup pending (~1 day).

## Layout

```
benchmarks/w1/
├── README.md                   (this file)
├── run_w1_sglang.sh            driver for sglang-based policies (lpm_lru, fcfs_lru, lpm_pe–clpm_gm_dl_pe)
├── aggregate.py                post-hoc summary
└── results/                    populated by drivers (gitignored)
    └── seed_<seed>/
        └── cell_<cell>/
            └── rate_<rate>/
                ├── <policy>.json         bench output
                ├── _run_<policy>.log     bench stdout/err
                └── _server_<policy>.log  sglang server stdout/err
```
