#!/usr/bin/env python3
"""Compare W3 reproduction JSON outputs against the paper's Tables 15 & 16.

Reads <results_base>/results_{sglang,sglang_dp2,vllm,vllm_dp2}/seed_<seed>/
cell_{B,C}/rate_heavy/{policy}.json and prints a side-by-side table with
relative deltas. Exit 0 if all metrics within tolerance, 1 otherwise.

Env vars:
  RESULTS_BASE   directory containing results_{sglang,sglang_dp2,vllm,vllm_dp2}/
                 (default: directory of this script)
  SEED           seed subdir to read (default: 42)
  TOL_PCT        tolerance on relative delta in percent (default: 20)
"""
import json
import os
import sys
from pathlib import Path

# Paper Tables 15 (SGLang) and 16 (vLLM), W3 heavy load, mean across 3 seeds.
PAPER = {
    ("sglang", 1, "B", "lpm_lru"):       {"tput": 0.464,  "cache": 55.1, "ttft_s": 324.2, "tpot_ms": 32.4, "e2e_s": 330.1},
    ("sglang", 1, "B", "clpm_gm_dl_pe"): {"tput": 1.237,  "cache": 98.2, "ttft_s":  85.3, "tpot_ms": 32.8, "e2e_s":  91.3},
    ("sglang", 1, "C", "lpm_lru"):       {"tput": 2.896,  "cache": 44.6, "ttft_s":  46.6, "tpot_ms": 47.6, "e2e_s":  50.0},
    ("sglang", 1, "C", "clpm_gm_dl_pe"): {"tput": 6.022,  "cache": 84.2, "ttft_s":  15.7, "tpot_ms": 44.3, "e2e_s":  18.9},
    ("sglang", 2, "B", "lpm_lru"):       {"tput": 1.043,  "cache": 87.8, "ttft_s": 180.7, "tpot_ms": 31.9, "e2e_s": 192.6},
    ("sglang", 2, "B", "clpm_gm_dl_pe"): {"tput": 1.689,  "cache": 96.9, "ttft_s":  91.4, "tpot_ms": 32.8, "e2e_s": 103.6},
    ("sglang", 2, "C", "lpm_lru"):       {"tput": 7.757,  "cache": 72.9, "ttft_s":  28.7, "tpot_ms": 37.0, "e2e_s":  33.4},
    ("sglang", 2, "C", "clpm_gm_dl_pe"): {"tput":13.634,  "cache": 91.5, "ttft_s":  10.2, "tpot_ms": 36.7, "e2e_s":  14.9},
    ("vllm",   1, "B", "fcfs_apc_lru"):  {"tput": 0.241,  "cache": 37.7, "ttft_s": 559.6, "tpot_ms": 36.0, "e2e_s": 572.7},
    ("vllm",   1, "B", "clpm_gm_dl_pe"): {"tput": 1.072,  "cache": 97.1, "ttft_s":  83.2, "tpot_ms": 65.9, "e2e_s": 103.4},
    ("vllm",   1, "C", "fcfs_apc_lru"):  {"tput": 1.798,  "cache": 37.2, "ttft_s":  75.8, "tpot_ms": 32.1, "e2e_s":  79.9},
    ("vllm",   1, "C", "clpm_gm_dl_pe"): {"tput": 3.728,  "cache": 76.5, "ttft_s":  24.9, "tpot_ms": 48.1, "e2e_s":  31.0},
    ("vllm",   2, "B", "fcfs_apc_lru"):  {"tput": 0.478,  "cache": 55.4, "ttft_s": 384.4, "tpot_ms": 42.1, "e2e_s": 399.6},
    ("vllm",   2, "B", "clpm_gm_dl_pe"): {"tput": 2.152,  "cache": 97.3, "ttft_s":  54.3, "tpot_ms": 93.9, "e2e_s":  83.8},
    ("vllm",   2, "C", "fcfs_apc_lru"):  {"tput": 4.040,  "cache": 51.4, "ttft_s":  60.5, "tpot_ms": 32.8, "e2e_s":  64.7},
    ("vllm",   2, "C", "clpm_gm_dl_pe"): {"tput":11.440,  "cache": 92.7, "ttft_s":  12.3, "tpot_ms": 43.7, "e2e_s":  17.9},
}

DIRMAP = {
    ("sglang", 1): "results_sglang",
    ("sglang", 2): "results_sglang_dp2",
    ("vllm",   1): "results_vllm",
    ("vllm",   2): "results_vllm_dp2",
}

# tolerance on relative delta — single-seed runs can wobble vs 3-seed paper means
TOL_PCT = float(os.environ.get("TOL_PCT", "20"))  # green if within tolerance


def load_observed(root: Path, engine: str, dp: int, cell: str, policy: str, seed: int):
    rel = DIRMAP[(engine, dp)]
    p = root / rel / f"seed_{seed}" / f"cell_{cell}" / "rate_heavy" / f"{policy}.json"
    if not p.exists():
        return None, str(p)
    with open(p) as f:
        d = json.load(f)
    return {
        "tput":   d["throughput"]["request_per_s"],
        "cache":  d["cache"]["hit_rate_pct"],
        "ttft_s": d["ttft_ms"]["mean"] / 1000.0,
        "tpot_ms":d["tpot_ms"]["mean"],
        "e2e_s":  d["e2e_ms"]["mean"] / 1000.0,
    }, str(p)


def fmt_delta(obs, pap):
    if obs is None or pap == 0:
        return "—"
    delta = (obs - pap) / pap * 100
    flag = "OK" if abs(delta) <= TOL_PCT else "!!"
    return f"{delta:+6.1f}% {flag}"


def main():
    default_base = Path(__file__).resolve().parent
    root = Path(os.environ.get("RESULTS_BASE", str(default_base)))
    seed = int(os.environ.get("SEED", "42"))
    print(f"W3 reproduction vs paper")
    print(f"  results_base = {root}")
    print(f"  seed         = {seed}")
    print(f"  tolerance    = ±{TOL_PCT:.0f}%\n")
    cols = ["tput", "cache", "ttft_s", "tpot_ms", "e2e_s"]
    fail = False
    print(f"{'engine':6} {'dp':>2} {'cell':4} {'policy':16}  {'metric':8} {'paper':>8}  {'observed':>10}  delta")
    print("-" * 78)
    for (engine, dp, cell, policy), pap in PAPER.items():
        obs, path = load_observed(root, engine, dp, cell, policy, seed)
        if obs is None:
            print(f"{engine:6} {dp:2d} {cell:4} {policy:16}  MISSING  ({path})")
            fail = True
            continue
        for c in cols:
            d = fmt_delta(obs[c], pap[c])
            print(f"{engine:6} {dp:2d} {cell:4} {policy:16}  {c:8} {pap[c]:>8.2f}  {obs[c]:>10.2f}  {d}")
            if "!!" in d:
                fail = True
        print()
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
