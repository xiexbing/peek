#!/usr/bin/env python3
"""W2 results aggregator.

Walks results/<policy>/<dataset>_<cell>/seed_<seed>/result.jsonl, extracts
summary metrics (TTFT/TPOT/ITL/E2E p50/p99 + throughput + concurrency),
and prints a per-cell comparison table grouped by (policy × cell × dataset).

Each result.jsonl has a single summary record (sglang's bench_serving
writes one per run when --output-file is given).

When --output-details was used, the same record additionally carries
per-request `ttfts`, `itls`, `output_lens`, `errors` arrays — useful for
turn-level analysis (turn-1 vs turn-2+ TTFT split is approximated by
ranking ttfts by per-session order in subsequent W2-specific scripts).

Usage:
  python aggregate.py                           # all policies, all cells
  python aggregate.py --metric mean_ttft_ms     # rank by a specific metric
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

W2 = Path(__file__).resolve().parent
RESULTS = W2 / "results"

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
    """Parse a Prometheus text-format snapshot, sum across labels per metric."""
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
        name = head.split("{", 1)[0]
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
    args = ap.parse_args()

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

    print(f"\n=== W2 results — primary metric: {args.metric} ===\n")
    print(f"{'policy':<10} {'dataset':<10} {'cell':<10} {'seeds':<6} "
          f"{'mean_ttft':<11} {'p99_ttft':<11} {'p99_tpot':<10} "
          f"{'p99_itl':<10} {'p99_e2e':<10} {'thpt':<8} {'hit%':<7} {'retr':<6} {'errs':<5}")
    print("-" * 130)

    seen_keys: set = set()
    for (policy, dataset, cell), recs in sorted(runs.items()):
        n_seeds = len(recs)
        seen_keys.add((policy, dataset, cell))

        def avg(field):
            vals = [r.get(field) for r in recs if r.get(field) is not None]
            return statistics.mean(vals) if vals else float("nan")

        def cavg(field):
            vals = [r["_cache"].get(field) for r in recs if r.get("_cache") and r["_cache"].get(field) is not None]
            return statistics.mean(vals) if vals else float("nan")

        errs = sum(
            sum(1 for e in (r.get("errors") or []) if e)
            for r in recs
        )
        hit = cavg("hit_rate")
        retr = cavg("retracted_reqs")
        hit_s = f"{hit*100:.1f}" if hit == hit else "—"
        retr_s = f"{retr:.0f}" if retr == retr else "—"

        print(f"{policy:<10} {dataset:<10} {cell:<10} {n_seeds:<6} "
              f"{avg('mean_ttft_ms'):<11.0f} {avg('p99_ttft_ms'):<11.0f} {avg('p99_tpot_ms'):<10.1f} "
              f"{avg('p99_itl_ms'):<10.1f} {avg('p99_e2e_latency_ms'):<10.0f} "
              f"{avg('request_throughput'):<8.2f} "
              f"{hit_s:<7} {retr_s:<6} {errs:<5}")

    # Headline comparison: peek vs lpm_lru deltas, per (dataset, cell)
    print(f"\n=== baseline → peek deltas ({args.metric}, lower is better) ===\n")
    by_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    for (policy, dataset, cell), recs in runs.items():
        vals = [r.get(args.metric) for r in recs if r.get(args.metric) is not None]
        if vals:
            by_cell[(dataset, cell)][policy] = statistics.mean(vals)

    print(f"{'dataset':<10} {'cell':<10} {'lpm_lru':<10} {'clpm_gm_dl_pe_decay':<14} {'clpm_gm_dl':<14} {'clpm_gm_dl_pe':<14}")
    print("-" * 80)
    for (dataset, cell), per_pol in sorted(by_cell.items()):
        b1 = per_pol.get("lpm_lru")
        row = [f"{dataset:<10}", f"{cell:<10}", f"{b1:<10.0f}" if b1 else "-"]
        for p in ("clpm_gm_dl_pe_decay", "clpm_gm_dl", "clpm_gm_dl_pe"):
            v = per_pol.get(p)
            if v is None or b1 is None:
                row.append(f"{'—':<14}")
            else:
                delta_pct = (b1 - v) / b1 * 100  # positive = peek is faster
                row.append(f"{v:.0f} ({delta_pct:+.1f}%)".ljust(14))
        print(" ".join(row))


if __name__ == "__main__":
    main()
