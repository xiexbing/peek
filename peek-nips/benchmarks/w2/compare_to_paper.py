#!/usr/bin/env python3
"""Compare W2 reproduction outputs against paper Tables 13 & 14.

W2 is the long-document RAG workload on cell B (7x KV pressure). The
paper reports both moderate and heavy load on this cell, for SGLang
(Table 13) and vLLM with APC (Table 14).

Reads
  <RESULTS_BASE_SGLANG>/seed_<S>/cell_B/rate_<R>/<policy>.json   (Table 13)
  <RESULTS_BASE_VLLM>/seed_<S>/cell_B/rate_<R>/<policy>.json     (Table 14)

and prints a side-by-side table with relative deltas. Means are computed
across all seeds present (paper config = 3 seeds: 42, 142, 242).

Env vars:
  RESULTS_BASE_SGLANG   default: <this dir>/results
  RESULTS_BASE_VLLM     default: <this dir>/results_vllm
  TOL_PCT               relative-delta tolerance, percent (default: 20)

Exit 0 if every metric is within tolerance, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

W2 = Path(__file__).resolve().parent

# Table 13: W2 SGLang, cell B (7x KV pressure), 3 seeds
PAPER_SGLANG = {
    ("B", "moderate", "lpm_lru"):       {"tput": 0.289, "cache": 62.4, "ttft_s": 391.1, "tpot_ms": 32.6, "e2e_s": 419.2},
    ("B", "moderate", "clpm_gm_dl_pe"): {"tput": 0.330, "cache": 79.4, "ttft_s": 178.8, "tpot_ms": 31.8, "e2e_s": 206.8},
    ("B", "heavy",    "lpm_lru"):       {"tput": 0.283, "cache": 60.0, "ttft_s": 543.4, "tpot_ms": 31.9, "e2e_s": 571.5},
    ("B", "heavy",    "clpm_gm_dl_pe"): {"tput": 0.357, "cache": 83.2, "ttft_s": 212.2, "tpot_ms": 33.4, "e2e_s": 241.6},
}

# Table 14: W2 vLLM (APC enabled), cell B, 3 seeds
PAPER_VLLM = {
    ("B", "moderate", "fcfs_apc_lru"):  {"tput": 0.114, "cache": 29.4, "ttft_s":  28.4, "tpot_ms": 27.4, "e2e_s":  51.4},
    ("B", "moderate", "clpm_gm_dl_pe"): {"tput": 0.114, "cache": 33.0, "ttft_s":   8.3, "tpot_ms": 27.8, "e2e_s":  31.6},
    ("B", "heavy",    "fcfs_apc_lru"):  {"tput": 0.143, "cache": 29.5, "ttft_s":  90.7, "tpot_ms": 28.8, "e2e_s": 114.6},
    ("B", "heavy",    "clpm_gm_dl_pe"): {"tput": 0.151, "cache": 40.1, "ttft_s":  24.7, "tpot_ms": 31.2, "e2e_s":  50.5},
}

TOL_PCT = float(os.environ.get("TOL_PCT", "20"))


def load_observed(base: Path, cell: str, rate: str, policy: str):
    matches = sorted(base.glob(f"seed_*/cell_{cell}/rate_{rate}/{policy}.json"))
    seeds = []
    samples = {"tput": [], "cache": [], "ttft_s": [], "tpot_ms": [], "e2e_s": []}
    for p in matches:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        try:
            seed = int(p.parent.parent.parent.name.split("_", 1)[1])
        except (ValueError, IndexError):
            seed = -1
        seeds.append(seed)
        samples["tput"].append(d["throughput"]["request_per_s"])
        samples["cache"].append(d["cache"]["hit_rate_pct"])
        samples["ttft_s"].append(d["ttft_ms"]["mean"] / 1000.0)
        samples["tpot_ms"].append(d["tpot_ms"]["mean"])
        samples["e2e_s"].append(d["e2e_ms"]["mean"] / 1000.0)
    if not seeds:
        return None, []
    return {k: statistics.mean(v) for k, v in samples.items()}, seeds


def fmt_delta(obs, paper):
    if obs is None or paper == 0:
        return "--", False
    delta = (obs - paper) / paper * 100.0
    ok = abs(delta) <= TOL_PCT
    return f"{delta:+6.1f}% {'OK' if ok else '!!'}", ok


def report(engine: str, base: Path, paper: dict) -> bool:
    print(f"\n=== W2 {engine} cell B -- vs paper (means across present seeds; tol ±{TOL_PCT:.0f}%) ===")
    print(f"  results_base = {base}")
    if not base.exists():
        print(f"  (results_base does not exist -- skipping {engine})")
        return True
    cols = ["tput", "cache", "ttft_s", "tpot_ms", "e2e_s"]
    print(f"\n{'cell':4} {'rate':9} {'policy':16}  {'metric':8} {'paper':>9}  {'observed':>10}  {'delta':>16}  seeds")
    print("-" * 96)
    all_ok = True
    for (cell, rate, policy), ref in paper.items():
        obs, seeds = load_observed(base, cell, rate, policy)
        if obs is None:
            print(f"{cell:4} {rate:9} {policy:16}  MISSING  (no seed_*/cell_{cell}/rate_{rate}/{policy}.json)")
            all_ok = False
            continue
        for c in cols:
            d, ok = fmt_delta(obs.get(c), ref[c])
            all_ok &= ok
            print(f"{cell:4} {rate:9} {policy:16}  {c:8} {ref[c]:>9.2f}  {obs[c]:>10.2f}  {d:>16}  "
                  f"{','.join(str(s) for s in sorted(seeds))}")
        print()
    return all_ok


def main() -> int:
    sglang_base = Path(os.environ.get("RESULTS_BASE_SGLANG", str(W2 / "results")))
    vllm_base = Path(os.environ.get("RESULTS_BASE_VLLM", str(W2 / "results_vllm")))
    s_ok = report("SGLang (Table 13)", sglang_base, PAPER_SGLANG)
    v_ok = report("vLLM (Table 14)", vllm_base, PAPER_VLLM)
    return 0 if (s_ok and v_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
