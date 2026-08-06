#!/usr/bin/env python3
"""W4 results aggregator.

Walks results/<policy>/<dataset>_<cell>/seed_<seed>/result.jsonl, extracts
summary metrics (TTFT/TPOT/ITL/E2E p50/p99 + throughput + concurrency),
and prints a per-cell comparison table grouped by (policy x cell x dataset).

Each result.jsonl has a single summary record (sglang's bench_serving
writes one per run when --output-file is given).

When --output-details was used, the same record additionally carries
per-request `ttfts`, `itls`, `output_lens`, `errors` arrays -- useful for
turn-level analysis (turn-1 vs turn-2+ TTFT split is approximated by
ranking ttfts by per-session order in subsequent W4-specific scripts).

Across-seed aggregation defaults to the MEDIAN, not the mean. W4 runs 3
seeds and the tail metrics carry a seed CV of 11-14%, so one outlier seed
moves a 3-sample mean substantially (measured: p99_ttft mean-of-3 8522 ms
vs median-of-3 9002 ms for the same runs). The `spread%` column reports
2*stdev/sqrt(n) as a percentage of the aggregate -- read it as "an effect
smaller than this is not distinguishable from seed noise", which is the
README's falsification criterion made visible.

Usage:
  python aggregate.py                           # all policies, all cells
  python aggregate.py --metric mean_ttft_ms     # rank by a specific metric
  python aggregate.py --agg mean                # legacy mean-of-seeds
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

W4 = Path(__file__).resolve().parent
RESULTS = W4 / "results"

HEADLINE_METRICS = [
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_e2e_latency_ms", "p99_e2e_latency_ms",
    "request_throughput", "output_throughput",
    "concurrency", "max_concurrent_requests",
    "completed",
]


def load_run(path: Path) -> dict | None:
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # The summary record has top-level metric keys (mean_ttft_ms etc).
                if "mean_ttft_ms" in rec:
                    return rec
    except Exception as e:
        print(f"  WARN: failed to read {path}: {e}")
    return None


def parse_prom(path: Path) -> dict[str, float]:
    """Parse a Prometheus text-format snapshot, sum across labels per metric.

    `sglang:cached_tokens_total` is exported with a `cache_source` label whose
    value is currently always "total" (an already-summed series). If sglang
    ever adds per-source breakdowns (device/host), blindly summing every label
    set would double-count, so restrict that metric to the "total" series.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        name, _, labels = head.partition("{")
        if name == "sglang:cached_tokens_total" and "cache_source=" in labels:
            if 'cache_source="total"' not in labels:
                continue
        out[name] = out.get(name, 0.0) + v
    return out


def cell_cache_stats(seed_dir: Path) -> dict[str, float]:
    pre = parse_prom(seed_dir / "_metrics_pre.prom")
    post = parse_prom(seed_dir / "_metrics_post.prom")
    if not pre or not post:
        return {}
    def d(k):
        return post.get(k, 0.0) - pre.get(k, 0.0)
    prompt = d("sglang:prompt_tokens_total")
    cached = d("sglang:cached_tokens_total")
    finished = d("sglang:num_retractions_count")
    retract_events = d("sglang:num_retractions_sum")
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_rate": cached / prompt if prompt > 0 else float("nan"),
        "retracted_reqs": retract_events,
        "finished_reqs": finished,
        "gen_tokens": d("sglang:generation_tokens_total"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="median_ttft_ms",
                    help="primary metric to display in comparison table")
    ap.add_argument("--agg", choices=("median", "mean"), default="median",
                    help="how to combine seeds (default: median -- robust to "
                         "the single outlier seed that skews a 3-sample mean)")
    ap.add_argument("--baseline", default="lpm_lru",
                    help="policy to compute deltas against")
    args = ap.parse_args()
    combine = statistics.median if args.agg == "median" else statistics.mean

    # Walk: RESULTS/<policy>/<dataset>_<cell>/seed_<seed>/result.jsonl
    runs: dict[tuple, list[dict]] = defaultdict(list)
    for policy_dir in sorted(RESULTS.iterdir()):
        if not policy_dir.is_dir():
            continue
        policy = policy_dir.name
        if policy.startswith("_"):
            continue
        for cell_dir in sorted(policy_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            # cell_dir name format: <dataset>_<cell>
            parts = cell_dir.name.split("_", 1)
            if len(parts) != 2:
                continue
            dataset, cell = parts
            for seed_dir in sorted(cell_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = seed_dir.name.replace("seed_", "")
                rec = load_run(seed_dir / "result.jsonl")
                if rec is None:
                    continue
                cache = cell_cache_stats(seed_dir)
                runs[(policy, dataset, cell)].append({"seed": seed, **rec, "_cache": cache})

    print(f"\n=== W4 results -- {args.agg}-of-seeds, primary metric: {args.metric} ===\n")
    print(f"{'policy':<14} {'dataset':<10} {'cell':<10} {'seeds':<6} "
          f"{'mean_ttft':<11} {'p99_ttft':<11} {'p99_tpot':<10} "
          f"{'p99_itl':<10} {'p99_e2e':<10} {'thpt':<8} {'hit%':<7} {'retr':<6} {'errs':<5}")
    print("-" * 134)

    for (policy, dataset, cell), recs in sorted(runs.items()):
        n_seeds = len(recs)

        def agg(field):
            vals = [r.get(field) for r in recs if r.get(field) is not None]
            return combine(vals) if vals else float("nan")

        def cagg(field):
            vals = [r["_cache"].get(field) for r in recs
                    if r.get("_cache") and r["_cache"].get(field) is not None]
            return combine(vals) if vals else float("nan")

        errs = sum(
            sum(1 for e in (r.get("errors") or []) if e)
            for r in recs
        )
        hit = cagg("hit_rate")
        retr = cagg("retracted_reqs")
        hit_s = f"{hit*100:.1f}" if hit == hit else "--"
        retr_s = f"{retr:.0f}" if retr == retr else "--"

        print(f"{policy:<14} {dataset:<10} {cell:<10} {n_seeds:<6} "
              f"{agg('mean_ttft_ms'):<11.0f} {agg('p99_ttft_ms'):<11.0f} {agg('p99_tpot_ms'):<10.1f} "
              f"{agg('p99_itl_ms'):<10.1f} {agg('p99_e2e_latency_ms'):<10.0f} "
              f"{agg('request_throughput'):<8.2f} "
              f"{hit_s:<7} {retr_s:<6} {errs:<5}")

    # Headline comparison: peek vs baseline deltas, per (dataset, cell).
    # Columns are derived from the data, not hardcoded -- a hardcoded list
    # silently omits any policy the matrix gains and prints "--" for ones it
    # never had.
    base_pol = args.baseline
    print(f"\n=== {base_pol} -> peek deltas ({args.metric}, {args.agg}-of-seeds; "
          f"positive = peek faster) ===\n")

    by_cell: dict[tuple, dict[str, list[float]]] = defaultdict(dict)
    for (policy, dataset, cell), recs in runs.items():
        vals = [r.get(args.metric) for r in recs if r.get(args.metric) is not None]
        if vals:
            by_cell[(dataset, cell)][policy] = vals

    policies = sorted({p for per in by_cell.values() for p in per if p != base_pol})
    if not policies:
        print(f"  (no policies to compare against {base_pol})")
        return

    def noise_pct(vals: list[float]) -> float:
        """2*stdev/sqrt(n) as % -- effects below this are seed noise."""
        if len(vals) < 2:
            return float("nan")
        mu = statistics.mean(vals)
        return 2 * statistics.stdev(vals) / (len(vals) ** 0.5) / mu * 100 if mu else float("nan")

    hdr = f"{'dataset':<10} {'cell':<10} {base_pol:<12} {'noise':<8}"
    for p in policies:
        hdr += f" {p:<20}"
    print(hdr)
    print("-" * len(hdr))
    for (dataset, cell), per_pol in sorted(by_cell.items()):
        bvals = per_pol.get(base_pol)
        if not bvals:
            continue
        b1 = combine(bvals)
        thr = noise_pct(bvals)
        thr_s = f"{thr:.1f}%" if thr == thr else "n/a"
        row = f"{dataset:<10} {cell:<10} {b1:<12.0f} {thr_s:<8}"
        for p in policies:
            vals = per_pol.get(p)
            if not vals:
                row += f" {'--':<20}"
                continue
            v = combine(vals)
            delta = (b1 - v) / b1 * 100  # positive = peek faster
            # Mark results that cannot be separated from seed noise.
            tag = "" if (thr != thr or abs(delta) >= thr) else " ns"
            row += f" {f'{v:.0f} ({delta:+.1f}%){tag}':<20}"
        print(row)
    print("\n  ns = within seed noise (|delta| < 2*stdev/sqrt(n) of the baseline)")


if __name__ == "__main__":
    main()
