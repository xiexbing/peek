#!/usr/bin/env python3
"""Cell-B head-to-head: stock lpm_lru vs clpm_gm_dl_pe (PEEK co-design).

Reports the paper's metrics plus the co-design ratio. Latency metrics are
base/peek (higher = PEEK better); throughput and hit-rate are peek/base.
"""
import json, sys, os

d = sys.argv[1] if len(sys.argv) > 1 else "/workspace/peek/benchmarks/w2/results"
BASE, PEEK = "lpm_lru", "clpm_gm_dl_pe"
SEED = os.environ.get("SEED", "42")

def load(rate, policy):
    p = os.path.join(d, f"seed_{SEED}", "cell_B", f"rate_{rate}", f"{policy}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def row(j):
    a = j["args"]
    ceiling = (a["n"] - a["warmup_reqs"]) / a["n"]
    return dict(
        tput=j["throughput"]["request_per_s"],
        eff=(j["throughput"]["request_per_s"] / a["rate"]) / ceiling,
        hit=j["cache"]["hit_rate_pct"],
        ttft=j["ttft_ms"]["mean"] / 1000,
        ttft_p95=j["ttft_ms"]["p95"] / 1000,
        tpot=j["tpot_ms"]["mean"],
        e2e=j["e2e_ms"]["mean"] / 1000,
        slo=j["throughput"]["slo_attainment_pct"],
        goodput=j["throughput"]["goodput_req_per_s"],
        err=j["counts"]["errored"],
        matched=j.get("phases", {}).get("matched_rids", 0),
    )

# (key, unit, higher_is_better)
METRICS = [
    ("tput", "req/s", True), ("eff", "", True), ("hit", "%", True),
    ("ttft", "s", False), ("ttft_p95", "s", False), ("tpot", "ms", False),
    ("e2e", "s", False), ("slo", "%", True), ("goodput", "req/s", True),
    ("err", "", False), ("matched", "", True),
]

print(f"\n### W2 cell B head-to-head (sglang, seed {SEED})   {d}")
for rate in ("moderate", "heavy"):
    jb, jp = load(rate, BASE), load(rate, PEEK)
    if not jb and not jp:
        print(f"\n[{rate}] no results yet"); continue
    b = row(jb) if jb else None
    p = row(jp) if jp else None
    off = (jb or jp)["args"]["rate"]
    print(f"\n[{rate}]  offered={off} req/s  N={(jb or jp)['args']['n']}")
    print(f"  {'metric':<10}{'lpm_lru':>12}{'clpm_gm_dl_pe':>16}{'ratio':>10}")
    for k, u, hib in METRICS:
        bv = f"{b[k]:.3f}" if b else "--"
        pv = f"{p[k]:.3f}" if p else "--"
        r = ""
        if b and p and b[k] and p[k]:
            r = (p[k] / b[k]) if hib else (b[k] / p[k])
            r = f"{r:.2f}x"
        print(f"  {k+('('+u+')' if u else ''):<10}{bv:>12}{pv:>16}{r:>10}")
print()
