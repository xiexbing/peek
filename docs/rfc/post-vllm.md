<!--
vLLM RFC issue post. Open at https://github.com/vllm-project/vllm/issues/new
using the "🎇 RFC" template and paste the sections below into the matching
fields. Title goes in the issue title box.
-->

# Title

`[RFC]: Queue-informed KV cache management — cache-aware scheduling (cluster-LPM) and demand-aware eviction`

---

## Motivation

Under memory pressure with prefix-sharing arrivals — shared system prompts / tool
definitions, multi-turn sessions, agentic tool-chains — vLLM v1 leaves reuse on
the table in two places:

- **Eviction is pure LRU** over the free-block list. It can evict a prefix moments
  before a burst of queued requests that share it is scheduled, forcing redundant
  prefill.
- **Scheduling is `fcfs`/`priority`** with no cache-aware reorder of the waiting
  queue, and no notion that N queued requests share a not-yet-cached prefix that
  should be warmed once, pioneer-first.

Both gaps share a cause: the **waiting queue already predicts near-future reuse**,
and neither subsystem consumes that signal. This is the subject of the PEEK paper
(Xie et al., 2026); we'd like to upstream a small, opt-in version.

## Proposed Change

Three opt-in, default-unchanged mechanisms, staged as separate PRs:

1. **`--scheduling-policy peek-lpm`** — reorder the FCFS-backed waiting queue by
   longest cached-prefix length each step, backed by an incrementally-maintained
   pending radix tree (built once, updated on arrival/schedule) instead of a
   per-step recompute. Establishes the tree lifecycle.
2. **`--scheduling-policy peek-clpm`** — cluster-LPM: split requests into
   warm / cold-pioneer / cold-sibling sections and interleave a
   cache/cluster-preferring lane with an oldest-first fairness lane (stride
   scheduler), so shared prefixes warm once and dense clusters admit first
   without starving small ones.
3. **`--enable-peek-eviction`** — each `KVCacheBlock` gets a `peek_demand`
   refreshed each step from the waiting queue's block hashes;
   `FreeKVCacheBlockQueue` evicts undemanded blocks first, spilling to
   lowest-demand only when needed, and falls back to the fast `popleft_n` path
   when nothing is demanded. Block-hash based, so it works with any scheduler.

Touches `v1/core/sched/scheduler.py`, `request_queue.py`, `block_pool.py`,
`kv_cache_utils.py`, `config/cache.py`, `engine/arg_utils.py`. All default-off;
existing paths unchanged when disabled.

**One open question we want guidance on:** the pending radix tree (and the
cluster-LPM ordering) can be hosted as (A) native pure-Python in-tree, (B) an
optional third-party dependency, or (C) a vendored compiled core. We recommend
**A** for reviewability and to avoid a core-path dependency, and would like the
maintainers' preference before we finalize the PRs. Full tradeoffs and design in
the RFC doc below.

**Full RFC (design + per-engine detail + tradeoffs):**
https://github.com/xiexbing/peek/blob/main/docs/rfc/0001-queue-informed-kv-cache-management.md

**Reference implementation:** working, DCO-signed branches for both engines with
CPU unit tests (peek-lpm → peek-clpm → peek-eviction); happy to share for review.

Benchmarks (PEEK paper §4, up to 4×H100, over `fcfs`+APC+LRU on shared-prefix
workloads): up to **2.6× cache-hit, 7.1× TTFT, 5.5× end-to-end latency, 4.5×
throughput**; within noise on workloads with no exploitable prefix structure.
These are from PEEK's reference implementation; per-branch numbers to follow.

## Feedback Period

2 weeks, or until maintainers weigh in on the hosting question (§ above), which
gates how we build the PRs.

## CC List

`/vllm/v1/core` CODEOWNERS: @WoosukKwon @robertgshaw2-redhat @njhill @ywang96
@alexm-redhat @heheda12345 @ApostaC @orozery @ivanium

Most relevant to this proposal (v1 scheduling + prefix caching / KV-cache core):
@WoosukKwon, @njhill (scheduler), @heheda12345 and @ApostaC (KV cache / prefix
caching).

## Any Other Things

- We suggest landing stage 1 (behavior-preserving foundation) first, then 2 and 3.
- Eviction (stage 3) is independent of the scheduler stages and could land on its
  own.
