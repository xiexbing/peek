#!/usr/bin/env python3
"""Compare W1 reproduction outputs against paper Tables 10, 11, 12.

Reads
  <RESULTS_BASE_SGLANG>/seed_<S>/cell_<C>/rate_heavy/<policy>.json   (Table 10, 12-SGLang)
  <RESULTS_BASE_VLLM>/seed_<S>/cell_<C>/rate_heavy/<policy>.json     (Table 11, 12-vLLM)

and prints a side-by-side table of every (cell x policy x metric) cell
in the paper's heavy-load tables, with the relative delta and a
green "OK" / red "!!" tag.

Means are computed across all seeds present (paper config = 3 seeds:
42, 142, 242). Single-seed runs are accepted with a warning.

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

W1 = Path(__file__).resolve().parent

# --- Paper Tables 10 (SGLang) and 11 (vLLM): W1 heavy load, means across 3 seeds ---
# Cells: A (2x), B (4x), C (8x, primary), D (16x).
# Policies are the paper-canonical labels mapped to the filesystem IDs.

# Table 10: SGLang heavy load
PAPER_SGLANG = {
    # cell -> policy -> {tput, cache, ttft_s, e2e_s}
    "A": {
        "lpm_lru":         {"tput":  7.43, "cache": 65.8, "ttft_s":  42.6, "e2e_s":  47.8},
        "lpm_pe":          {"tput":  7.36, "cache": 65.4, "ttft_s":  43.1, "e2e_s":  48.3},
        "clpm":            {"tput": 14.46, "cache": 93.7, "ttft_s":  11.0, "e2e_s":  16.9},
        "clpm_gm":         {"tput": 14.80, "cache": 93.6, "ttft_s":  10.8, "e2e_s":  16.4},
        "clpm_gm_dl":      {"tput": 14.98, "cache": 93.5, "ttft_s":  10.4, "e2e_s":  16.0},
        "clpm_gm_pe":      {"tput": 14.89, "cache": 93.5, "ttft_s":  10.5, "e2e_s":  16.2},
        "clpm_gm_dl_pe":   {"tput": 15.03, "cache": 93.6, "ttft_s":  10.3, "e2e_s":  15.9},
    },
    "B": {
        "lpm_lru":         {"tput":  6.73, "cache": 59.2, "ttft_s":  55.0, "e2e_s":  60.3},
        "lpm_pe":          {"tput":  6.70, "cache": 58.8, "ttft_s":  55.3, "e2e_s":  60.8},
        "clpm":            {"tput": 13.74, "cache": 91.9, "ttft_s":  16.5, "e2e_s":  22.5},
        "clpm_gm":         {"tput": 14.34, "cache": 93.3, "ttft_s":  15.4, "e2e_s":  21.1},
        "clpm_gm_dl":      {"tput": 14.49, "cache": 93.3, "ttft_s":  14.9, "e2e_s":  20.8},
        "clpm_gm_pe":      {"tput": 14.66, "cache": 93.3, "ttft_s":  14.9, "e2e_s":  20.8},
        "clpm_gm_dl_pe":   {"tput": 14.55, "cache": 93.2, "ttft_s":  15.0, "e2e_s":  20.8},
    },
    "C": {
        "lpm_lru":         {"tput":  1.42, "cache": 38.3, "ttft_s": 233.2, "e2e_s": 238.2},
        "lpm_pe":          {"tput":  1.41, "cache": 38.3, "ttft_s": 234.0, "e2e_s": 239.0},
        "clpm":            {"tput":  3.72, "cache": 88.3, "ttft_s":  36.9, "e2e_s":  42.6},
        "clpm_gm":         {"tput":  3.75, "cache": 89.4, "ttft_s":  32.7, "e2e_s":  38.3},
        "clpm_gm_dl":      {"tput":  3.71, "cache": 89.2, "ttft_s":  31.5, "e2e_s":  37.2},
        "clpm_gm_pe":      {"tput":  3.75, "cache": 89.2, "ttft_s":  32.6, "e2e_s":  38.3},
        "clpm_gm_dl_pe":   {"tput":  3.73, "cache": 88.8, "ttft_s":  29.7, "e2e_s":  35.3},
    },
    "D": {
        "lpm_lru":         {"tput":  1.22, "cache": 30.8, "ttft_s": 335.5, "e2e_s": 340.6},
        "lpm_pe":          {"tput":  1.20, "cache": 30.9, "ttft_s": 340.2, "e2e_s": 345.4},
        "clpm":            {"tput":  4.28, "cache": 91.0, "ttft_s":  62.5, "e2e_s":  69.1},
        "clpm_gm":         {"tput":  4.42, "cache": 92.2, "ttft_s":  59.6, "e2e_s":  66.1},
        "clpm_gm_dl":      {"tput":  4.36, "cache": 92.2, "ttft_s":  60.3, "e2e_s":  66.7},
        "clpm_gm_pe":      {"tput":  4.44, "cache": 92.4, "ttft_s":  59.9, "e2e_s":  66.3},
        "clpm_gm_dl_pe":   {"tput":  4.41, "cache": 92.1, "ttft_s":  59.7, "e2e_s":  66.2},
    },
}

# Table 11: vLLM heavy load (APC enabled)
PAPER_VLLM = {
    "A": {
        "fcfs_apc_lru":    {"tput":  8.50, "cache": 66.3, "ttft_s":  14.0, "e2e_s":  19.3},
        "fcfs_apc_pe":     {"tput":  8.34, "cache": 66.1, "ttft_s":  14.4, "e2e_s":  19.9},
        "clpm":            {"tput": 11.58, "cache": 87.5, "ttft_s":   6.0, "e2e_s":  11.1},
        "clpm_gm":         {"tput": 13.14, "cache": 86.4, "ttft_s":   4.2, "e2e_s":  10.4},
        "clpm_gm_dl":      {"tput": 11.41, "cache": 86.7, "ttft_s":   6.1, "e2e_s":  11.2},
        "clpm_gm_pe":      {"tput": 12.96, "cache": 86.8, "ttft_s":   4.3, "e2e_s":  10.7},
        "clpm_gm_dl_pe":   {"tput": 13.04, "cache": 86.7, "ttft_s":   4.2, "e2e_s":  10.4},
    },
    "B": {
        "fcfs_apc_lru":    {"tput":  7.52, "cache": 58.1, "ttft_s":  17.1, "e2e_s":  22.4},
        "fcfs_apc_pe":     {"tput":  7.31, "cache": 57.9, "ttft_s":  17.7, "e2e_s":  23.1},
        "clpm":            {"tput": 10.83, "cache": 83.7, "ttft_s":   8.0, "e2e_s":  13.2},
        "clpm_gm":         {"tput": 11.76, "cache": 81.1, "ttft_s":   6.5, "e2e_s":  12.6},
        "clpm_gm_dl":      {"tput": 10.65, "cache": 83.5, "ttft_s":   7.8, "e2e_s":  13.1},
        "clpm_gm_pe":      {"tput": 11.56, "cache": 81.7, "ttft_s":   6.5, "e2e_s":  12.8},
        "clpm_gm_dl_pe":   {"tput": 11.57, "cache": 81.7, "ttft_s":   6.4, "e2e_s":  12.5},
    },
    "C": {
        "fcfs_apc_lru":    {"tput":  1.68, "cache": 37.7, "ttft_s":  97.8, "e2e_s": 102.6},
        "fcfs_apc_pe":     {"tput":  1.82, "cache": 38.0, "ttft_s":  86.6, "e2e_s":  91.5},
        "clpm":            {"tput":  4.90, "cache": 86.5, "ttft_s":  16.9, "e2e_s":  22.4},
        "clpm_gm":         {"tput":  4.36, "cache": 86.2, "ttft_s":  21.8, "e2e_s":  27.2},
        "clpm_gm_dl":      {"tput":  4.80, "cache": 86.2, "ttft_s":  18.5, "e2e_s":  23.8},
        "clpm_gm_pe":      {"tput":  4.24, "cache": 85.8, "ttft_s":  22.6, "e2e_s":  28.7},
        "clpm_gm_dl_pe":   {"tput":  4.25, "cache": 86.3, "ttft_s":  22.4, "e2e_s":  28.5},
    },
    "D": {
        "fcfs_apc_lru":    {"tput":  1.54, "cache": 32.8, "ttft_s": 106.3, "e2e_s": 111.3},
        "fcfs_apc_pe":     {"tput":  1.46, "cache": 31.1, "ttft_s": 111.1, "e2e_s": 116.3},
        "clpm":            {"tput":  3.04, "cache": 81.8, "ttft_s":  39.4, "e2e_s":  44.4},
        "clpm_gm":         {"tput":  3.60, "cache": 78.9, "ttft_s":  35.0, "e2e_s":  40.4},
        "clpm_gm_dl":      {"tput":  2.96, "cache": 80.9, "ttft_s":  40.5, "e2e_s":  45.6},
        "clpm_gm_pe":      {"tput":  3.57, "cache": 79.2, "ttft_s":  34.4, "e2e_s":  40.4},
        "clpm_gm_dl_pe":   {"tput":  3.52, "cache": 79.0, "ttft_s":  33.2, "e2e_s":  39.0},
    },
}

TOL_PCT = float(os.environ.get("TOL_PCT", "20"))


def load_observed(base: Path, cell: str, policy: str) -> tuple[dict | None, list[int]]:
    """Mean across all seed_*/cell_<cell>/rate_heavy/<policy>.json files present."""
    matches = sorted(base.glob(f"seed_*/cell_{cell}/rate_heavy/{policy}.json"))
    seeds = []
    samples = {"tput": [], "cache": [], "ttft_s": [], "e2e_s": []}
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
        samples["e2e_s"].append(d["e2e_ms"]["mean"] / 1000.0)
    if not seeds:
        return None, []
    return {k: statistics.mean(v) for k, v in samples.items()}, seeds


def fmt_delta(obs: float | None, paper: float) -> tuple[str, bool]:
    if obs is None or paper == 0:
        return "--", False
    delta = (obs - paper) / paper * 100.0
    ok = abs(delta) <= TOL_PCT
    flag = "OK" if ok else "!!"
    return f"{delta:+6.1f}% {flag}", ok


def report(engine: str, base: Path, paper: dict) -> bool:
    print(f"\n=== W1 {engine} heavy load -- vs paper (means across present seeds; tol ±{TOL_PCT:.0f}%) ===")
    print(f"  results_base = {base}")
    if not base.exists():
        print(f"  (results_base does not exist -- skipping {engine})")
        return True
    cols = ["tput", "cache", "ttft_s", "e2e_s"]
    print(f"\n{'cell':4} {'policy':16}  {'metric':8} {'paper':>8}  {'observed':>10}  {'delta':>16}  seeds")
    print("-" * 86)
    all_ok = True
    for cell, policies in paper.items():
        for policy, ref in policies.items():
            obs, seeds = load_observed(base, cell, policy)
            if obs is None:
                print(f"{cell:4} {policy:16}  MISSING  (no seed_*/cell_{cell}/rate_heavy/{policy}.json)")
                all_ok = False
                continue
            for c in cols:
                d, ok = fmt_delta(obs.get(c), ref[c])
                all_ok &= ok
                print(f"{cell:4} {policy:16}  {c:8} {ref[c]:>8.2f}  {obs[c]:>10.2f}  {d:>16}  "
                      f"{','.join(str(s) for s in sorted(seeds))}")
            print()
    return all_ok


def main() -> int:
    sglang_base = Path(os.environ.get("RESULTS_BASE_SGLANG", str(W1 / "results")))
    vllm_base = Path(os.environ.get("RESULTS_BASE_VLLM", str(W1 / "results_vllm")))
    s_ok = report("SGLang (Table 10)", sglang_base, PAPER_SGLANG)
    v_ok = report("vLLM (Table 11)", vllm_base, PAPER_VLLM)
    return 0 if (s_ok and v_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
