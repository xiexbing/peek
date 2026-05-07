#!/usr/bin/env python3
"""Compare W5 reproduction outputs against paper Tables 19 & 20.

W5 is the no-regression test on singleton chat (LMSYS-Chat-1M). Two
cells -- `C_short` (32-1024 tok prompts) and `C_long` (512-4096 tok
prompts). Heavy load only.

Reads (SGLang -- with rate sub-directory):
  <RESULTS_BASE_SGLANG>/seed_<S>/cell_<C>/rate_heavy/<policy>.json

Reads (vLLM -- no rate sub-directory; W5 vLLM is single-rate):
  <RESULTS_BASE_VLLM>/seed_<S>/cell_<C>/<policy>.json

Note (paper Table 19 caption): the published SGLang C_long row averages
**seed 242 only** because the other two seeds saturated below the
calibrated rate. By default this script honours that and averages only
seed 242 for SGLang C_long; pass --no-seed-restriction to use all
present seeds.

Env vars:
  RESULTS_BASE_SGLANG   default: <this dir>/results
  RESULTS_BASE_VLLM     default: <this dir>/results_vllm
  TOL_PCT               default: 20

Exit 0 if every metric is within tolerance, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

W5 = Path(__file__).resolve().parent

# Table 19: W5 SGLang heavy load.
# C_short: 3-seed mean. C_long: seed 242 only (per paper caption).
PAPER_SGLANG = {
    ("C_short", "lpm_lru"):       {"tput": 21.89, "cache":  3.7, "ttft_s":  3.6, "tpot_ms": 54.6, "e2e_s": 10.05},
    ("C_short", "clpm_gm_dl_pe"): {"tput": 21.68, "cache":  3.7, "ttft_s":  3.6, "tpot_ms": 54.9, "e2e_s": 10.14},
    ("C_long",  "lpm_lru"):       {"tput":  7.83, "cache":  0.7, "ttft_s": 21.7, "tpot_ms": 52.2, "e2e_s": 28.30},
    ("C_long",  "clpm_gm_dl_pe"): {"tput":  8.20, "cache": 10.2, "ttft_s": 19.4, "tpot_ms": 52.7, "e2e_s": 26.00},
}

# Per paper Table 19 caption, SGLang C_long uses only seed 242.
SGLANG_CELL_SEED_RESTRICTION = {"C_long": {242}}

# Table 20: W5 vLLM heavy load (APC enabled), 3 seeds.
PAPER_VLLM = {
    ("C_short", "fcfs_apc_lru"):  {"tput": 39.63, "cache": 96.7, "ttft_s": 0.39, "tpot_ms": 43.9, "e2e_s":  5.69},
    ("C_short", "clpm_gm_dl_pe"): {"tput": 39.70, "cache": 96.7, "ttft_s": 0.35, "tpot_ms": 44.0, "e2e_s":  5.66},
    ("C_long",  "fcfs_apc_lru"):  {"tput": 18.33, "cache": 97.5, "ttft_s": 4.25, "tpot_ms": 73.6, "e2e_s": 12.45},
    ("C_long",  "clpm_gm_dl_pe"): {"tput": 22.27, "cache": 98.8, "ttft_s": 1.61, "tpot_ms": 74.3, "e2e_s":  9.98},
}

TOL_PCT = float(os.environ.get("TOL_PCT", "20"))


def _seed_from_path(p: Path) -> int:
    # SGLang : .../seed_<S>/cell_<C>/rate_heavy/<policy>.json -> seed dir is .parent.parent.parent
    # vLLM   : .../seed_<S>/cell_<C>/<policy>.json            -> seed dir is .parent.parent
    for ancestor in (p.parent.parent.parent, p.parent.parent):
        if ancestor.name.startswith("seed_"):
            try:
                return int(ancestor.name.split("_", 1)[1])
            except ValueError:
                pass
    return -1


def load_observed(base: Path, cell: str, policy: str, seed_filter: set[int] | None = None):
    """Means across present seeds (optionally filtered).

    Walks both the SGLang `rate_heavy` layout and the vLLM rate-less layout.
    """
    matches = sorted(
        list(base.glob(f"seed_*/cell_{cell}/rate_heavy/{policy}.json"))
        + list(base.glob(f"seed_*/cell_{cell}/{policy}.json"))
    )
    matches = [p for p in matches if not p.name.startswith("_")]
    seeds = []
    samples = {"tput": [], "cache": [], "ttft_s": [], "tpot_ms": [], "e2e_s": []}
    for p in matches:
        seed = _seed_from_path(p)
        if seed_filter is not None and seed not in seed_filter:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        seeds.append(seed)
        samples["tput"].append(d.get("throughput", {}).get("request_per_s") or 0.0)
        samples["cache"].append(d.get("cache", {}).get("hit_rate_pct") or float("nan"))
        samples["ttft_s"].append((d.get("ttft_ms", {}).get("mean") or 0.0) / 1000.0)
        samples["tpot_ms"].append(d.get("tpot_ms", {}).get("mean") or 0.0)
        samples["e2e_s"].append((d.get("e2e_ms", {}).get("mean") or 0.0) / 1000.0)
    if not seeds:
        return None, []
    out = {}
    for k, v in samples.items():
        clean = [x for x in v if x == x]
        out[k] = statistics.mean(clean) if clean else float("nan")
    return out, seeds


def fmt_delta(obs, paper):
    if obs is None or obs != obs or paper == 0:
        return "--", False if obs is None else True
    delta = (obs - paper) / paper * 100.0
    ok = abs(delta) <= TOL_PCT
    return f"{delta:+6.1f}% {'OK' if ok else '!!'}", ok


def report(engine: str, table_id: str, base: Path, paper: dict,
           seed_restrictions: dict[str, set[int]] | None) -> bool:
    print(f"\n=== W5 {engine} {table_id} -- vs paper (tol ±{TOL_PCT:.0f}%) ===")
    print(f"  results_base = {base}")
    if not base.exists():
        print(f"  (results_base does not exist -- skipping)")
        return True
    cols = ["tput", "cache", "ttft_s", "tpot_ms", "e2e_s"]
    print(f"\n{'cell':9} {'policy':16}  {'metric':8} {'paper':>9}  {'observed':>10}  {'delta':>16}  seeds")
    print("-" * 92)
    all_ok = True
    for (cell, policy), ref in paper.items():
        sf = (seed_restrictions or {}).get(cell)
        obs, seeds = load_observed(base, cell, policy, sf)
        if obs is None:
            sf_note = f" (filter={sorted(sf)})" if sf else ""
            print(f"{cell:9} {policy:16}  MISSING{sf_note}")
            all_ok = False
            continue
        for c in cols:
            ref_v = ref[c]
            obs_v = obs.get(c)
            d, ok = fmt_delta(obs_v, ref_v)
            if obs_v is None or obs_v != obs_v:
                obs_str = "    --"
            else:
                obs_str = f"{obs_v:>10.2f}"
                all_ok &= ok
            print(f"{cell:9} {policy:16}  {c:8} {ref_v:>9.2f}  {obs_str}  {d:>16}  "
                  f"{','.join(str(s) for s in sorted(seeds))}")
        print()
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seed-restriction", action="store_true",
                    help="Average over all present seeds even where the paper "
                         "table reports a single-seed value (SGLang C_long).")
    args = ap.parse_args()

    sglang_base = Path(os.environ.get("RESULTS_BASE_SGLANG", str(W5 / "results")))
    vllm_base = Path(os.environ.get("RESULTS_BASE_VLLM", str(W5 / "results_vllm")))

    sglang_restrict = None if args.no_seed_restriction else SGLANG_CELL_SEED_RESTRICTION
    s_ok = report("SGLang", "(Table 19)", sglang_base, PAPER_SGLANG, sglang_restrict)
    v_ok = report("vLLM", "(Table 20)", vllm_base, PAPER_VLLM, None)
    return 0 if (s_ok and v_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
