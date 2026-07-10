<!--
SGLang design-issue post. Open a GitHub issue on sgl-project/sglang (or a
Discussion) with the title below and the body that follows. Add labels if the
project uses them (e.g. "enhancement", "scheduler").
-->

# Title

`[RFC] Queue-aware scheduling (cluster-LPM) and demand-aware radix eviction`

---

## Summary

Add small, opt-in, default-unchanged mechanisms that let the scheduler and the
radix-cache evictor use the **waiting queue's prefix-sharing structure** — a
predictive signal for near-future reuse that neither consumes today. Design and
evaluation are from the PEEK paper (Xie et al., 2026). We have a working
reference implementation and want to align on approach before opening PRs.

## Problem

Under memory pressure with prefix-sharing arrivals (shared system prompts / tool
definitions, multi-turn sessions, agentic tool-chains):

- **`--radix-eviction-policy lru` is past-only** — it can evict a shared prefix
  moments before a burst of queued requests that share it is scheduled, forcing
  redundant prefill.
- **`--schedule-policy lpm`** orders by existing cache hit and breaks ties by
  arrival. When K queued requests share a *cold* prefix it has no cheap way to
  warm it once pioneer-first; it also rebuilds a throwaway auxiliary radix tree
  on every `calc_priority` call for its in-batch dedup, and falls back to FCFS
  past 128 queued.

## Proposal

Three opt-in mechanisms, staged as separate PRs:

1. **`--schedule-policy peek-lpm`** — LPM ordering (same key, same in-batch
   dedup), but over an **incrementally-maintained pending radix tree** instead of
   the per-call aux-tree rebuild. Behavior-preserving foundation that establishes
   the tree lifecycle (diff-synced against `waiting_queue` in `calc_priority`).
2. **`--schedule-policy peek-clpm`** — cluster-LPM: warm / cold-pioneer /
   cold-sibling sections + a cache/cluster-preferring lane interleaved with an
   oldest-first fairness lane, so shared prefixes warm once and dense clusters
   admit first without starving small ones. Degenerates to LPM on no-sharing
   queues.
3. **`--radix-eviction-policy peek`** — a `PeekDemandStrategy` protects nodes
   whose prefix queued requests will reuse (token-weighted `pending_demand` over
   the node's ancestors), evicting undemanded nodes first, LRU within a tier. It
   reads the same pending tree the peek schedule policy maintains.

Touches `managers/schedule_policy.py`, `mem_cache/radix_cache.py`,
`mem_cache/evict_policy.py`, `server_args.py`. Default-off; no existing policy or
sort/match path changes when disabled.

## One open question for maintainers

The pending radix tree + cluster-LPM ordering can be hosted as **(A)** native
pure-Python in-tree, **(B)** an optional dependency, or **(C)** a vendored
compiled core. We recommend **(A)** — self-contained and reviewable, and
consistent with SGLang already building an aux radix tree per call — but would
like your preference before finalizing, since it determines how the PRs are
built. Tradeoffs are in the full RFC.

**Full RFC (design + tradeoffs + vLLM parity):**
https://github.com/xiexbing/peek/blob/main/docs/rfc/0001-queue-informed-kv-cache-management.md

**Reference implementation:** working, DCO-signed branches with CPU-only tests in
`test/registered/` (peek-lpm → peek-clpm → peek-eviction); happy to share.

Benchmarks (PEEK paper §4, up to 4×H100, over `lpm`+LRU on shared-prefix
workloads): up to **3.0× cache-hit, 7.9× TTFT, 6.7× end-to-end latency, 3.6×
throughput**; within noise on workloads with no exploitable prefix structure.
These are from PEEK's reference implementation; per-branch numbers to follow.

## Notes

- Suggest landing stage 1 first (behavior-preserving), then 2 and 3.
- If keeping eviction fully decoupled from scheduling is preferred, a
  self-contained per-`TreeNode` demand-count variant (no pending tree, no
  dependency) is also viable — noted in the RFC.
