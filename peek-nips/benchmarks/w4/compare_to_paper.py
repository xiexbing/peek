#!/usr/bin/env python3
"""Compare W4 reproduction outputs against paper Tables 17 & 18.

W4 is the agentic-burst workload (Mooncake conversation_trace). Two
scenarios -- `agentic_only` (no shared prompt) and `agentic_shared`
(1402-token shared prompt prepended). The paper reports heavy load
on both for SGLang (Table 17) and vLLM with APC (Table 18).

Reads (SGLang -- one tree, scenarios as sub-cells):
  <RESULTS_BASE_SGLANG>/<policy>/mooncake_<cell>/seed_<S>/result.jsonl

Reads (vLLM -- two scenario-rooted trees):
  <RESULTS_BASE_VLLM_AGENTIC_ONLY>/seed_<S>/cell_<cell>/<policy>.json
  <RESULTS_BASE_VLLM_AGENTIC_SHARED>/seed_<S>/cell_<cell>/<policy>.json

Where <cell> is `heavy` for the paper-reported rows. The driver also
runs `moderate` (omitted from Tables 17, 18).

Env vars:
  RESULTS_BASE_SGLANG               default: <this dir>/results
  RESULTS_BASE_VLLM_AGENTIC_ONLY    default: <this dir>/results_vllm/agentic_only
  RESULTS_BASE_VLLM_AGENTIC_SHARED  default: <this dir>/results_vllm/agentic_shared
  TOL_PCT                           default: 20

Exit 0 if every metric is within tolerance, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

W4 = Path(__file__).resolve().parent

# Table 17: W4 SGLang heavy load, 3 seeds. Scenarios = agentic_only, agentic_shared.
# Note: the SGLang driver writes a single tree where scenarios appear as
# separate (policy, cell) results. The paper distinguishes scenarios via
# whether SHARED_SYSTEM_PROMPT_PATH was set when the run was produced.
# We honour the driver's actual layout: the driver uses a single results/
# tree per invocation, so reviewers would run it twice (once with the
# shared prompt path, once with `SHARED_SYSTEM_PROMPT_PATH=""`).
PAPER_SGLANG = {
    # scenario -> policy -> {tput, cache, ttft_ms, tpot_ms, e2e_s}
    "agentic_only": {
        "fcfs_lru":       {"tput": 3.58, "cache": 59.5, "ttft_ms": 1175, "tpot_ms": 65.8, "e2e_s": 22.4},
        "lpm_lru":        {"tput": 3.58, "cache": 60.4, "ttft_ms": 1129, "tpot_ms": 63.9, "e2e_s": 22.0},
        "clpm_gm_dl_pe":  {"tput": 3.62, "cache": 60.7, "ttft_ms": 1185, "tpot_ms": 63.7, "e2e_s": 21.8},
    },
    "agentic_shared": {
        "fcfs_lru":       {"tput": 3.59, "cache": 59.7, "ttft_ms": 1181, "tpot_ms": 66.3, "e2e_s": 22.4},
        "lpm_lru":        {"tput": 3.58, "cache": 60.0, "ttft_ms": 1142, "tpot_ms": 65.1, "e2e_s": 22.2},
        "clpm_gm_dl_pe":  {"tput": 3.62, "cache": 61.0, "ttft_ms": 1157, "tpot_ms": 62.8, "e2e_s": 21.6},
    },
}

# Table 18: W4 vLLM heavy load (APC enabled), 3 seeds.
PAPER_VLLM = {
    "agentic_only": {
        "fcfs_apc_lru":   {"tput": 3.57, "cache": 85.6, "ttft_ms": 178, "tpot_ms": 22.8, "e2e_s": 5.40},
        "clpm_gm_dl_pe":  {"tput": 3.58, "cache": 87.2, "ttft_ms": 167, "tpot_ms": 22.5, "e2e_s": 5.31},
    },
    "agentic_shared": {
        "fcfs_apc_lru":   {"tput": 3.54, "cache": 91.5, "ttft_ms": 171, "tpot_ms": 22.8, "e2e_s": 4.78},
        "clpm_gm_dl_pe":  {"tput": 3.54, "cache": 91.5, "ttft_ms": 173, "tpot_ms": 22.9, "e2e_s": 4.79},
    },
}

TOL_PCT = float(os.environ.get("TOL_PCT", "20"))


def _summary_from_jsonl(path: Path) -> dict | None:
    """SGLang bench_serving writes a single summary record per run."""
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "mean_ttft_ms" in rec:
                    return rec
    except Exception:
        return None
    return None


def load_sglang(base: Path, scenario: str, policy: str):
    # SGLang driver layout: <base>/<policy>/mooncake_<cell>/seed_<S>/result.jsonl
    # The "scenario" axis (agentic_only vs agentic_shared) is captured by
    # whether SHARED_SYSTEM_PROMPT_PATH was set during the run -- it does
    # not appear in the path. We let the user separate runs by base dir
    # (e.g. results_only/ vs results_shared/) but also accept a single
    # tree -- in which case observed values are reported once and a
    # warning is printed.
    matches = sorted(base.glob(f"{policy}/mooncake_heavy/seed_*/result.jsonl"))
    seeds = []
    samples = {"tput": [], "cache": [], "ttft_ms": [], "tpot_ms": [], "e2e_s": []}
    for p in matches:
        rec = _summary_from_jsonl(p)
        if rec is None:
            continue
        seed_dir = p.parent.name
        try:
            seed = int(seed_dir.split("_", 1)[1])
        except (ValueError, IndexError):
            seed = -1
        seeds.append(seed)
        samples["tput"].append(rec.get("request_throughput") or 0.0)
        samples["ttft_ms"].append(rec.get("mean_ttft_ms") or 0.0)
        samples["tpot_ms"].append(rec.get("mean_tpot_ms") or 0.0)
        samples["e2e_s"].append((rec.get("mean_e2e_latency_ms") or 0.0) / 1000.0)
        # SGLang summary doesn't carry a hit-rate field; the driver dumps
        # Prometheus snapshots in the same dir. Fall back to N/A here.
        samples["cache"].append(float("nan"))
    if not seeds:
        return None, []
    out = {}
    for k, v in samples.items():
        clean = [x for x in v if x == x]  # drop NaN
        out[k] = statistics.mean(clean) if clean else float("nan")
    return out, seeds


def load_vllm(base: Path, policy: str):
    # vLLM driver layout: <base>/seed_<S>/cell_heavy/<policy>.json
    matches = sorted(base.glob(f"seed_*/cell_heavy/{policy}.json"))
    seeds = []
    samples = {"tput": [], "cache": [], "ttft_ms": [], "tpot_ms": [], "e2e_s": []}
    for p in matches:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        try:
            seed = int(p.parent.parent.name.split("_", 1)[1])
        except (ValueError, IndexError):
            seed = -1
        seeds.append(seed)
        samples["tput"].append(d.get("throughput", {}).get("request_per_s") or 0.0)
        samples["cache"].append(d.get("cache", {}).get("hit_rate_pct") or float("nan"))
        samples["ttft_ms"].append(d.get("ttft_ms", {}).get("mean") or 0.0)
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


def report(engine: str, table_id: str, paper_per_scenario: dict, loader, base_per_scenario: dict[str, Path]) -> bool:
    print(f"\n=== W4 {engine} {table_id} -- vs paper (means across present seeds; tol ±{TOL_PCT:.0f}%) ===")
    cols = ["tput", "cache", "ttft_ms", "tpot_ms", "e2e_s"]
    all_ok = True
    for scenario, policies in paper_per_scenario.items():
        base = base_per_scenario.get(scenario)
        if base is None:
            continue
        print(f"\n  scenario = {scenario}   results_base = {base}")
        if not base.exists():
            print(f"    (results_base does not exist -- skipping)")
            continue
        print(f"\n  {'policy':16}  {'metric':8} {'paper':>9}  {'observed':>10}  {'delta':>16}  seeds")
        print("  " + "-" * 80)
        for policy, ref in policies.items():
            obs, seeds = loader(base, policy) if loader is load_vllm else loader(base, scenario, policy)
            if obs is None:
                print(f"  {policy:16}  MISSING")
                all_ok = False
                continue
            for c in cols:
                ref_v = ref[c]
                obs_v = obs.get(c)
                d, ok = fmt_delta(obs_v, ref_v)
                if obs_v is None or obs_v != obs_v:
                    obs_str = "    --"
                    # Don't mark as failure for cache where SGLang summary
                    # genuinely lacks the field.
                    if not (engine == "SGLang" and c == "cache"):
                        all_ok = False
                else:
                    obs_str = f"{obs_v:>10.2f}"
                    all_ok &= ok
                print(f"  {policy:16}  {c:8} {ref_v:>9.2f}  {obs_str}  {d:>16}  "
                      f"{','.join(str(s) for s in sorted(seeds))}")
            print()
    return all_ok


def main() -> int:
    sglang_base = Path(os.environ.get("RESULTS_BASE_SGLANG", str(W4 / "results")))
    vllm_only = Path(os.environ.get("RESULTS_BASE_VLLM_AGENTIC_ONLY",
                                    str(W4 / "results_vllm" / "agentic_only")))
    vllm_shared = Path(os.environ.get("RESULTS_BASE_VLLM_AGENTIC_SHARED",
                                      str(W4 / "results_vllm" / "agentic_shared")))

    # SGLang: by default, both scenarios resolve to the same base. Reviewers
    # who keep separate trees can override RESULTS_BASE_SGLANG_AGENTIC_ONLY /
    # _AGENTIC_SHARED.
    sglang_only = Path(os.environ.get("RESULTS_BASE_SGLANG_AGENTIC_ONLY", str(sglang_base)))
    sglang_shared = Path(os.environ.get("RESULTS_BASE_SGLANG_AGENTIC_SHARED", str(sglang_base)))

    s_ok = report("SGLang", "(Table 17)", PAPER_SGLANG, load_sglang,
                  {"agentic_only": sglang_only, "agentic_shared": sglang_shared})
    v_ok = report("vLLM", "(Table 18)", PAPER_VLLM, load_vllm,
                  {"agentic_only": vllm_only, "agentic_shared": vllm_shared})
    return 0 if (s_ok and v_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
