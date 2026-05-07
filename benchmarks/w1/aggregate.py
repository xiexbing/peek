#!/usr/bin/env python3
"""W1 results aggregator.

Walks results/seed_<S>/cell_<X>/rate_<R>/<policy>.json and produces:
  - summary.csv     : one row per completed run (seed x cell x rate x policy)
  - aggregated.csv  : median / mean / stdev across seeds per (cell x rate x policy)
  - delta.csv       : relative improvement (%) of each non-baseline policy vs lpm_lru
                     per (cell x rate), using median across seeds
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict


METRIC_SPECS = [
    ("throughput_req_s",  lambda s: s["throughput"]["request_per_s"]),
    ("goodput_req_s",     lambda s: s["throughput"]["goodput_req_per_s"]),
    ("slo_pct",           lambda s: s["throughput"]["slo_attainment_pct"]),
    ("cache_hit_pct",     lambda s: s["cache"]["hit_rate_pct"]),
    ("ttft_p50",          lambda s: s["ttft_ms"]["p50"]),
    ("ttft_p95",          lambda s: s["ttft_ms"]["p95"]),
    ("ttft_p99",          lambda s: s["ttft_ms"]["p99"]),
    ("itl_p95",           lambda s: s["itl_ms"]["p95"]),
    ("tpot_p50",          lambda s: s["tpot_ms"]["p50"]),
    ("tpot_p95",          lambda s: s["tpot_ms"]["p95"]),
    ("tpot_p99",          lambda s: s["tpot_ms"]["p99"]),
    ("e2e_p50",           lambda s: s["e2e_ms"]["p50"]),
    ("e2e_p99",           lambda s: s["e2e_ms"]["p99"]),
    ("max_in_flight",     lambda s: s["concurrency"]["max_in_flight"]),
    ("errored",           lambda s: s["counts"]["errored"]),
]

# For delta analysis: higher-better vs lower-better direction.
LOWER_IS_BETTER = {
    "ttft_p50", "ttft_p95", "ttft_p99", "itl_p95",
    "tpot_p50", "tpot_p95", "tpot_p99", "e2e_p50", "e2e_p99", "errored",
}


def load_row(path: str) -> dict:
    with open(path) as f:
        s = json.load(f)
    row = {}
    for name, fn in METRIC_SPECS:
        try:
            row[name] = fn(s)
        except (KeyError, TypeError):
            row[name] = None
    return row


def parse_path(path: str) -> tuple:
    # .../seed_42/cell_C/rate_heavy/clpm_gm_pe.json
    parts = path.split(os.sep)
    seed = int(next(x for x in parts if x.startswith("seed_")).split("_", 1)[1])
    cell = next(x for x in parts if x.startswith("cell_")).split("_", 1)[1]
    rate = next(x for x in parts if x.startswith("rate_")).split("_", 1)[1]
    policy = os.path.splitext(os.path.basename(path))[0]
    return seed, cell, rate, policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir",
                    default="/workspace/peek/benchmarks/w1/results")
    ap.add_argument("--out-dir",
                    default="/workspace/peek/benchmarks/w1")
    ap.add_argument("--baseline", default="lpm_lru",
                    help="policy name used as reference in delta.csv (default lpm_lru)")
    args = ap.parse_args()

    files = sorted(glob.glob(
        os.path.join(args.results_dir, "seed_*/cell_*/rate_*/*.json")))
    if not files:
        print(f"no results found under {args.results_dir}")
        return

    # --- summary.csv : one row per run ---
    rows = []
    for p in files:
        seed, cell, rate, policy = parse_path(p)
        m = load_row(p)
        rows.append({"seed": seed, "cell": cell, "rate": rate,
                     "policy": policy, **m})

    metric_keys = [name for name, _ in METRIC_SPECS]
    out_raw = os.path.join(args.out_dir, "summary.csv")
    with open(out_raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "cell", "rate", "policy",
                                          *metric_keys])
        w.writeheader()
        w.writerows(rows)

    # --- aggregated.csv : median/mean/stdev across seeds ---
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["rate"], r["policy"])].append(r)

    agg_fields = ["cell", "rate", "policy", "n_seeds"]
    for k in metric_keys:
        agg_fields.extend([f"{k}_median", f"{k}_mean", f"{k}_stdev"])

    agg_rows = []
    for (cell, rate, policy), grs in sorted(groups.items()):
        rec = {"cell": cell, "rate": rate, "policy": policy,
               "n_seeds": len(grs)}
        for k in metric_keys:
            vals = [g[k] for g in grs if g[k] is not None]
            if not vals:
                rec[f"{k}_median"] = rec[f"{k}_mean"] = rec[f"{k}_stdev"] = None
                continue
            rec[f"{k}_median"] = statistics.median(vals)
            rec[f"{k}_mean"] = statistics.mean(vals)
            rec[f"{k}_stdev"] = (statistics.stdev(vals)
                                 if len(vals) > 1 else 0.0)
        agg_rows.append(rec)

    out_agg = os.path.join(args.out_dir, "aggregated.csv")
    with open(out_agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)

    # --- delta.csv : each policy vs baseline, median ---
    median_of: dict[tuple, dict] = {}
    for (cell, rate, policy), grs in groups.items():
        median_of[(cell, rate, policy)] = {
            k: statistics.median([g[k] for g in grs if g[k] is not None])
               if any(g[k] is not None for g in grs) else None
            for k in metric_keys
        }

    delta_rows = []
    cells_rates = sorted({(c, r) for (c, r, _) in median_of.keys()})
    for cell, rate in cells_rates:
        base = median_of.get((cell, rate, args.baseline))
        if base is None:
            continue
        policies_here = [p for (c, r, p) in median_of.keys()
                         if c == cell and r == rate and p != args.baseline]
        for policy in sorted(policies_here):
            cur = median_of[(cell, rate, policy)]
            rec = {"cell": cell, "rate": rate,
                   "policy": policy, "vs": args.baseline}
            for k in metric_keys:
                b, v = base.get(k), cur.get(k)
                if b is None or v is None or b == 0:
                    rec[f"{k}_delta_pct"] = None
                    continue
                pct = 100.0 * (v - b) / b
                if k in LOWER_IS_BETTER:
                    pct = -pct  # flip so positive = improvement
                rec[f"{k}_delta_pct"] = round(pct, 2)
            delta_rows.append(rec)

    delta_fields = ["cell", "rate", "policy", "vs"] + [
        f"{k}_delta_pct" for k in metric_keys
    ]
    out_delta = os.path.join(args.out_dir, "delta.csv")
    with open(out_delta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=delta_fields)
        w.writeheader()
        w.writerows(delta_rows)

    print(f"wrote {len(rows)} raw rows       -> {out_raw}")
    print(f"wrote {len(agg_rows)} agg rows   -> {out_agg}")
    print(f"wrote {len(delta_rows)} deltas   -> {out_delta}")


if __name__ == "__main__":
    main()
