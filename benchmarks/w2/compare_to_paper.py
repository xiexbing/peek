#!/usr/bin/env python3
"""Compare a W2 vLLM run against PEEK paper Table 15 (W2 on vLLM, cell B).

Layout: results_vllm/seed_<seed>/cell_B/rate_<moderate|heavy>/<policy>.json
Averages each metric across seeds (paper reports means across 3 seeds).
Paper reports TTFT and E2E in SECONDS, TPOT in ms.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

# Paper Table 15: W2 on vLLM (cell B, 7x KV pressure). Means across 3 seeds, errored=0.
PAPER = {
    "moderate": {
        "fcfs_apc_lru":  dict(tput=0.114, hit=29.4, ttft=28.4, tpot=27.4, e2e=51.4),
        "clpm_gm_dl_pe": dict(tput=0.114, hit=33.0, ttft=8.3,  tpot=27.8, e2e=31.6),
    },
    "heavy": {
        "fcfs_apc_lru":  dict(tput=0.143, hit=29.5, ttft=90.7, tpot=28.8, e2e=114.6),
        "clpm_gm_dl_pe": dict(tput=0.151, hit=40.1, ttft=24.7, tpot=31.2, e2e=50.5),
    },
}
METRICS = [("tput", "req/s"), ("hit", "%"), ("ttft", "s"), ("tpot", "ms"), ("e2e", "s")]


def extract(j):
    return dict(
        tput=j["throughput"]["request_per_s"],
        hit=j["cache"]["hit_rate_pct"],
        ttft=j["ttft_ms"]["mean"] / 1000.0,   # -> seconds
        tpot=j["tpot_ms"]["mean"],            # ms
        e2e=j["e2e_ms"]["mean"] / 1000.0,     # -> seconds
    )


def load(results, rate, policy):
    out = []
    for sd in sorted(results.glob("seed_*")):
        f = sd / "cell_B" / f"rate_{rate}" / f"{policy}.json"
        if f.exists():
            out.append(extract(json.loads(f.read_text())))
    return out


def agg(rows, k):
    v = [r[k] for r in rows]
    return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


def fmt(v, u):
    return {"req/s": f"{v:.3f}", "%": f"{v:.1f}", "s": f"{v:.1f}", "ms": f"{v:.1f}"}.get(u, f"{v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results_vllm"))
    a = ap.parse_args()
    results = Path(a.results)
    print(f"\n### W2 vLLM run vs PEEK paper Table 15  (cell B)\nresults: {results}")

    for rate in ("moderate", "heavy"):
        have = {p: load(results, rate, p) for p in ("fcfs_apc_lru", "clpm_gm_dl_pe")}
        n = {p: len(v) for p, v in have.items()}
        if not any(n.values()):
            print(f"\n[{rate}] no results yet"); continue
        print(f"\n[{rate}]  seeds: " + ", ".join(f"{p}={n[p]}" for p in n))
        print(f"  {'metric':<7}{'run base':>14}{'run peek':>14}{'paper base':>12}{'paper peek':>12}"
              f"{'run x':>8}{'paper x':>9}")
        for k, u in METRICS:
            rb, _ = agg(have["fcfs_apc_lru"], k)
            rp, _ = agg(have["clpm_gm_dl_pe"], k)
            pb = PAPER[rate]["fcfs_apc_lru"][k]
            pp = PAPER[rate]["clpm_gm_dl_pe"][k]
            # speedup ratio (base/peek for latency, peek/base for tput/hit)
            def ratio(base, peek):
                if any(x != x for x in (base, peek)) or peek == 0: return float("nan")
                return (base / peek) if k in ("ttft", "e2e", "tpot") else (peek / base)
            rx, px = ratio(rb, rp), ratio(pb, pp)
            print(f"  {k:<7}{fmt(rb,u):>14}{fmt(rp,u):>14}{fmt(pb,u):>12}{fmt(pp,u):>12}"
                  f"{rx:>7.2f}x{px:>8.2f}x")
    print()


if __name__ == "__main__":
    main()
