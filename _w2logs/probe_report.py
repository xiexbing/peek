#!/usr/bin/env python3
"""Summarize the cell-B rate calibration probe.

NOTE on the achieved/offered ratio: throughput is (post-warmup ok) / (FULL
wall clock), but wall clock includes the warmup arrivals. So a perfectly
stable run tops out at (n - warmup)/n, NOT 1.0 -- 0.833 for N=300/warmup=50,
0.90 for the matrix's cell B (N=1000/warmup=100), 0.80 for N=500/warmup=100.
'eff' below is ratio / that ceiling, which is the number to read.

Verdict: STABLE = eff >= 0.95 and no timeouts and bounded ttft_p95.
"""
import json, sys, glob, os

d = sys.argv[1] if len(sys.argv) > 1 else "/workspace/peek/benchmarks/w2/results_probe"
files = sorted(glob.glob(os.path.join(d, "rate_*.json")),
               key=lambda p: float(os.path.basename(p)[5:-5]))
if not files:
    print(f"no probe results yet in {d}")
    sys.exit(0)

print(f"\n### cell-B rate probe (stock lpm_lru, sglang)   {d}\n")
hdr = (f"{'offered':>8}{'achieved':>9}{'eff':>6}{'hit%':>7}{'ttft_mean':>11}{'ttft_p95':>10}"
       f"{'e2e_mean':>10}{'slo%':>7}{'err':>5}  verdict")
print(hdr)
print("-" * len(hdr))
for f in files:
    j = json.load(open(f))
    a = j["args"]
    off = a["rate"]
    ach = j["throughput"]["request_per_s"]
    ceiling = (a["n"] - a["warmup_reqs"]) / a["n"]
    eff = (ach / off) / ceiling
    ttft_mean = j["ttft_ms"]["mean"] / 1000
    ttft_p95 = j["ttft_ms"]["p95"] / 1000
    e2e = j["e2e_ms"]["mean"] / 1000
    err = j["counts"]["errored"]
    slo = j["throughput"]["slo_attainment_pct"]
    hit = j["cache"]["hit_rate_pct"]
    stable = eff >= 0.95 and err == 0 and ttft_p95 < 60
    verdict = "STABLE" if stable else ("KNEE" if eff >= 0.90 and err == 0 else "BACKLOGGED")
    print(f"{off:>8}{ach:>9.3f}{eff:>6.2f}{hit:>7.1f}{ttft_mean:>10.1f}s{ttft_p95:>9.1f}s"
          f"{e2e:>9.1f}s{slo:>7.1f}{err:>5}  {verdict}")

print("\npaper Table 15 (vLLM cell B): tput 0.114 mod / 0.143-0.151 heavy; "
      "hit 29.4-29.5% base; ttft 28.4s mod / 90.7s heavy (base)\n"
      "Pick the highest STABLE rate as 'moderate'; ~1.3x that as 'heavy'.\n")
