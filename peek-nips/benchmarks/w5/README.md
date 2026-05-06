# W5 — Heterogeneous chat (no-regress safety test)

> Environment: sglang 0.5.9 / vllm 0.19.1, torch 2.9.1, Python 3.12,
> 1×H100 80GB (bf16). Full spec at `benchmarks/ENVIRONMENT.md`.
> Rate calibration: per-engine, per-cell. r_sat = highest rate with
> `errored=0` AND `slo_attainment_pct >= 90%`; `moderate = 0.4 × r_sat`, `heavy = 0.8 × r_sat`. See
> `benchmarks/ENVIRONMENT.md#rate-calibration-moderate--heavy-per-cell`.


**Status: DRAFT.**

## Purpose

Prove peek **does not regress** on LPM-unfriendly workloads: mixed traffic
where most requests are singletons (no prefix sharing). Every reviewer's
first skeptical question will be "what's the downside?" W5 is the answer.

The dynamic-lane controller exists specifically to protect singletons
under mixed traffic. W5 tests whether it works.

## Workload shape

```
Mix:
  40% of requests  : one of G=10 shared system prompts (500 tokens each)
  60% of requests  : unique prompts, length lognormal(mean=512, sigma≈2)
                     spanning 32–4096 tokens

Decode: 64–256 tokens (variable)
Distribution (within the 40% shared segment): Zipf α=1.0
Total N: 1500
Arrival: Poisson
```

The 40% / 60% split reflects general-purpose chat: some traffic uses a
system-prompt template; much is ad-hoc. Variation: 20/80 and 60/40 as
sensitivity.

## Production analog

- ChatGPT default (most queries have no custom GPT; some do)
- Claude / Gemini default chat
- Open-source hosted chat (HuggingChat, LMSYS Arena)
- General-purpose inference API (where tenants have diverse workloads)
- Any chat platform without enforced prompt templating

## Key question this answers

**Does peek stay within ±3% of stock LPM on every primary metric when
sharing structure is weak?**

If peek regresses here, it's not production-deployable as a general
scheduler — it would only be usable for known-sharing-heavy workloads,
which is a much narrower claim.

## Policy matrix

Simpler than W1/W3/W4. The 8-policy lattice is overkill here; what
matters is headline stock-vs-peek parity.

| Label | scheduler | eviction | role |
| -- | --------- | --------- | ---- |
| lpm_lru | SGLang LPM | LRU | baseline |
| clpm_gm | cLPM + GM | LRU | does scheduling regress? |
| clpm_gm_dl | cLPM + GM + DL | LRU | **does dynamic-lane protect singletons?** — primary W5 claim |
| clpm_gm_pe | cLPM + GM + peek queue-aware | peek queue-aware (cluster mode) | full co-design — should be neutral |

Four rows. Two claims:

1. **clpm_gm_dl (with dynamic lane) ≈ lpm_lru** on all metrics → dynamic lane works
2. **clpm_gm_pe ≈ lpm_lru** on all metrics → full peek in production is safe-default

Explicit prediction: **clpm_gm may regress here** because group-major without
dynamic lane can starve singletons. clpm_gm_dl fixes it.

## Cells

Draft — W5 axes are the **sharing ratio**:

| cell | shared % | singleton % | purpose |
| ---- | -------- | ----------- | ------- |
| A    | 60       | 40          | moderate sharing — typical SaaS chat |
| **B** | **40**  | **60**      | **PRIMARY** — canonical heterogeneous |
| C    | 20       | 80          | singleton-dominated — worst case for LPM-style schedulers |

## Rates

Moderate only (this is a safety test, not a scaling claim):

| cell | moderate |
| ---- | -------- |
| A (60/40) | 5 |
| B (40/60) | 5 |
| C (20/80) | 5 |

## Seeds

Same as W1: 42, 142, 242.

## Metrics that matter here

Standard set. Focus on **per-class metrics** (shared-segment vs
singleton-segment) as well as aggregate:

- Singleton TTFT p50 / p99 — must not regress
- Singleton goodput — must not regress
- Shared-segment metrics — may improve (peek's normal advantage)
- Aggregate metrics — must be ≈ stock (within ±3%)

## What we need to build

- **Mixed-distribution workload generator** — partial existing
  infrastructure (`bench_shared_prompts.py` does shared-only; needs a
  knob to inject a singleton fraction with lognormal prompt length)
- **Per-class metric aggregation** — split by "which class was this req"
  at report time

## Pre-registered predictions

| metric | clpm_gm vs lpm_lru | **clpm_gm_dl vs lpm_lru** | clpm_gm_pe vs lpm_lru |
| --- | --- | --- | --- |
| Aggregate goodput | −3% to +3% | **within ±2%** | within ±3% |
| Singleton TTFT p99 | possibly worse (+5 to +15%) | **within ±3%** | within ±3% |
| Shared-segment TPOT p99 | same as W1 | same as W1 | same as W1 |
| SLO attainment | possibly lower | **within ±2 pp** | within ±2 pp |

## Falsification

- If **clpm_gm_dl regresses beyond ±3% on any primary metric in W5 cell B**, dynamic
  lane doesn't do its job. The paper needs to either:
  - Tune dyn's parameters (SLO budget, EMA alpha, floor/ceiling)
  - Drop the "general-purpose safe" claim and limit peek to shared-sharing
    deployments
- If **clpm_gm and clpm_gm_dl look the same (both regress OR both don't)**, dynamic lane
  is either always-needed-and-works or always-unnecessary. Either is a
  valid finding but changes the narrative.

## Status

Draft. Workload generator needs a small extension. No urgent timeline —
W5 runs last (after W1/W3/W4 confirm peek's positive claims; W5 seals
the "doesn't hurt" story).
