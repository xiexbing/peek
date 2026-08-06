#!/usr/bin/env python3
"""Compare a W4 vLLM run against the PEEK paper Table 19 (W4 heavy, vLLM).

Reads results_vllm/<scenario>/seed_<seed>/cell_<cell>/<policy>.json, averages
each metric across seeds (paper reports means across 3 seeds), and prints a
side-by-side table with % delta vs the paper. Also emits the run's own
peek-vs-baseline no-regression check (the falsification criterion).

Usage:
  python compare_to_paper.py [--results DIR] [--scenario agentic_shared]
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

# Paper Table 19: W4 heavy load on vLLM, means across 3 seeds.
# metric -> {scenario -> {policy_label -> value}}
PAPER = {
    "agentic_only": {
        "fcfs_apc_lru":  dict(tput=3.57, hit=85.6, ttft=178, tpot=22.8, e2e=5.40),
        "clpm_gm_dl_pe": dict(tput=3.58, hit=87.2, ttft=167, tpot=22.5, e2e=5.31),
    },
    "agentic_shared": {
        "fcfs_apc_lru":  dict(tput=3.54, hit=91.5, ttft=171, tpot=22.8, e2e=4.78),
        "clpm_gm_dl_pe": dict(tput=3.54, hit=91.5, ttft=173, tpot=22.9, e2e=4.79),
    },
}
METRICS = [("tput", "req/s"), ("hit", "%"), ("ttft", "ms"), ("tpot", "ms"), ("e2e", "s")]


def extract(j: dict) -> dict:
    return dict(
        tput=j["throughput"]["request_per_s"],
        hit=j["cache"]["hit_rate_pct"],
        ttft=j["ttft_ms"]["mean"],
        tpot=j["tpot_ms"]["mean"],
        e2e=j["e2e_ms"]["mean"] / 1000.0,
    )


def load_cell(results: Path, scenario: str, cell: str, policy: str):
    """Return per-seed metric dicts for a (cell, policy)."""
    root = results / scenario
    out = []
    for seed_dir in sorted(root.glob("seed_*")):
        f = seed_dir / f"cell_{cell}" / f"{policy}.json"
        if f.exists():
            out.append(extract(json.loads(f.read_text())))
    return out


def agg(rows, key):
    vals = [r[key] for r in rows]
    return statistics.mean(vals) if vals else float("nan"), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(v, unit):
    if unit in ("req/s",): return f"{v:.2f}"
    if unit in ("%",):     return f"{v:.1f}"
    if unit in ("ms",):    return f"{v:.0f}"
    if unit in ("s",):     return f"{v:.2f}"
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results_vllm"))
    ap.add_argument("--scenario", default="agentic_shared")
    a = ap.parse_args()
    results = Path(a.results)
    scenario = a.scenario

    print(f"\n### W4 vLLM run vs PEEK paper Table 19  ({scenario})\n")
    print(f"results: {results/scenario}")

    for cell in ("heavy", "moderate"):
        have = {p: load_cell(results, scenario, cell, p)
                for p in ("fcfs_apc_lru", "clpm_gm_dl", "clpm_gm_dl_pe")}
        n = {p: len(v) for p, v in have.items()}
        if not any(n.values()):
            print(f"\n[{cell}] no results yet")
            continue
        paper_cell = PAPER.get(scenario, {}) if cell == "heavy" else {}
        tag = "PAPER-COMPARABLE (Table 19 is heavy-only)" if cell == "heavy" else "no paper table (figures only)"
        print(f"\n[{cell}]  seeds found: " + ", ".join(f"{p}={n[p]}" for p in n) + f"   [{tag}]")
        header = f"  {'metric':<7} " + "".join(f"{p:>16}" for p in have) + f"{'paper.base':>12}{'paper.peek':>12}"
        print(header)
        for key, unit in METRICS:
            line = f"  {key:<7} "
            for p in have:
                if have[p]:
                    m, sd = agg(have[p], key)
                    line += f"{fmt(m, unit)+'±'+fmt(sd, unit):>16}"
                else:
                    line += f"{'-':>16}"
            pb = paper_cell.get("fcfs_apc_lru", {}).get(key)
            pp = paper_cell.get("clpm_gm_dl_pe", {}).get(key)
            line += f"{(fmt(pb,unit) if pb is not None else '-'):>12}{(fmt(pp,unit) if pp is not None else '-'):>12}"
            print(line)

        # deltas vs paper (heavy only)
        if cell == "heavy" and paper_cell:
            print("\n  Δ vs paper (run_mean / paper - 1):")
            for run_pol, paper_pol in (("fcfs_apc_lru", "fcfs_apc_lru"), ("clpm_gm_dl_pe", "clpm_gm_dl_pe")):
                if not have[run_pol]:
                    continue
                parts = []
                for key, unit in METRICS:
                    m, _ = agg(have[run_pol], key)
                    pv = paper_cell[paper_pol][key]
                    d = 100.0 * (m / pv - 1.0) if pv else float("nan")
                    parts.append(f"{key} {d:+.1f}%")
                print(f"    {run_pol:<16} " + "  ".join(parts))

        # in-run no-regression: peek vs baseline
        if have["fcfs_apc_lru"] and have["clpm_gm_dl_pe"]:
            print("\n  no-regress (clpm_gm_dl_pe vs fcfs_apc_lru, this run):")
            parts = []
            for key, unit in METRICS:
                b, _ = agg(have["fcfs_apc_lru"], key)
                pk, _ = agg(have["clpm_gm_dl_pe"], key)
                d = 100.0 * (pk / b - 1.0) if b else float("nan")
                parts.append(f"{key} {d:+.1f}%")
            print("    " + "  ".join(parts))
    print()


if __name__ == "__main__":
    main()
