# RFC: Queue-informed KV cache management (cache-aware scheduling + demand eviction)

**Authors:** Bing Xie (@xiexbing), Zhipeng Wang, Masahiro Tanaka, Zhen Zheng
**Status:** Draft — request for comments
**Applies to:** SGLang and vLLM (v1 engine). One design; per-engine integration
sections below.
**Reference:** Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zhen Zheng. *"PEEK:
Predictive Queue-Informed KV Cache Management for LLM Serving,"* 2026.
https://arxiv.org/abs/2607.02525 

---

## 1. Summary

SGLang and vLLM already exploit KV-cache reuse — SGLang's LPM scheduling over
RadixAttention, vLLM's automatic prefix caching (APC). But they exploit it
**reactively**, from *past* and *current* cache state; neither uses the
**predictive** signal in the waiting queue — the token-prefix **relationships
among the pending (upcoming) requests themselves** — for scheduling or for
eviction. Under memory pressure with bursty, prefix-sharing arrivals (shared
system prompts, multi-turn sessions, agentic tool-chains), that blind spot
costs reuse:

- **Eviction ignores upcoming demand.** LRU (and LFU/SLRU) score blocks only by
  *past* access, with no view of the pending requests about to arrive at the
  batch. So LRU can evict a prefix moments before a burst of queued requests
  that share it is scheduled, forcing redundant prefill.
- **Scheduling ignores the queue's sharing structure.** Longest-prefix-match
  (LPM) orders by *existing* cache hit and breaks ties by arrival; it has no
  cheap way to see that N queued requests share a not-yet-cached prefix and
  should be warmed once, pioneer-first.

Both gaps share one root cause: **the waiting queue already tells us which
prefixes the pending requests are about to (re)use, and neither the scheduler
nor the KV-cache evictor — in SGLang or vLLM — consumes that signal.**

This RFC proposes a small, opt-in, default-unchanged set of mechanisms that do:

1. an **incrementally-maintained pending radix tree** over the waiting queue's
   token sequences (the shared primitive),
2. **cache-aware scheduling** — `peek-lpm` (LPM, tree-backed) and `peek`, the
   full PEEK cluster scheduler (cluster-LPM + group-major + dynamic-lane), and
3. **demand-aware eviction** — protect blocks a queued request will reuse.

We have a working reference implementation for both engines (links in §9) and
are seeking guidance on **one main question — how the pending tree should be
hosted** (§7) — before opening PRs.

---

## 2. Motivation

Prefix sharing across *queued* requests is common and growing:

- **Shared system prompts / tool definitions.** Agent frameworks (Cursor,
  Copilot, Claude Code) and RAG pipelines prepend a large, identical prefix to
  every request. Dozens of queued requests share thousands of prefix tokens.
- **Multi-turn sessions.** Each turn's prefix is the whole prior conversation.
- **Bursty tool-chains.** Requests arrive in correlated bursts that share
  structure.

Two concrete failures under memory pressure:

- **Eviction thrash.** LPM/LRU evicts a shared prefix; a burst of queued
  requests that need it then all re-prefill it. The prefill work is quadratic in
  the number of siblings that miss.
- **Pioneer starvation.** When K queued requests share a *cold* prefix, LPM
  admits them in arrival order; the prefix is (re)computed up to K times instead
  of once, because no request is designated to warm it first.

The waiting queue is a *predictive* signal for the very near future. Consuming
it closes both gaps.

---

## 3. Background: what each engine does today

**SGLang.** `--schedule-policy lpm` sorts the waiting queue by
`num_matched_prefix_tokens` against the radix cache, with an in-batch
deduplication pass that (re)builds a throwaway auxiliary radix tree on every
`calc_priority` call, and a fallback to FCFS when the queue exceeds 128.
Eviction is pluggable via `--radix-eviction-policy {lru,lfu,slru,priority}`,
scored per node.

**vLLM (v1).** Scheduling is `fcfs` or `priority`; there is no cache-aware
reorder of the waiting queue. Prefix-cache eviction is pure LRU over the
free-block list (`FreeKVCacheBlockQueue.popleft_n`).

Neither reads the *queue's* prefix-sharing structure for scheduling or eviction.

---

## 4. Proposal overview

Three mechanisms, staged so each is independently reviewable and each is
strictly opt-in (default behavior is byte-for-byte unchanged):

| Stage | Name | Surface | Behavior |
|---|---|---|---|
| 1 | **peek-lpm** | new schedule policy | LPM ordering, but over an incrementally-maintained pending radix tree instead of a per-call aux-tree rebuild. Establishes the tree lifecycle. |
| 2 | **peek** | new schedule policy | The full PEEK cluster scheduler (paper §3.2): cluster-LPM sort + **group-major** cluster batching + **dynamic-lane** fairness. PEEK's default, most-performant scheduling policy. |
| 3 | **peek eviction** | new eviction policy / flag | Protect blocks a queued request will reuse; evict undemanded blocks first. |

We propose landing them in that order (1 → 2 → 3). The recommended production
config is `peek` scheduling **+** peek eviction (the paper's `clpm_gm_dl_pe`,
"full PEEK").

---

## 5. Design

### 5.1 The pending radix tree (shared primitive)

A radix tree keyed by the token sequences of the requests **currently in the
waiting queue**. It is maintained **incrementally** — `insert(rid, tokens)` on
arrival, `remove(rid)` on schedule/finish — and exposes:

- `pending_demand(path) -> int` — number of waiting requests whose sequence has
  `path` as a prefix (drives eviction).
- `compute_req_scores() -> {rid: score}` — per-request dense-subtree weight,
  `Σ pending_count(v)·|edge(v)|` along the request's path (drives cLPM).
- `all_cluster_info() -> {rid: (node, depth, size) | None}` — the request's
  deepest ≥2-pending ancestor (its cluster) and that cluster's size.
- `has_sharing() -> bool` — short-circuit for no-sharing queues.

**Lifecycle (both engines).** The scheduler reconciles the tree against the live
waiting queue once per step via a **diff**: insert rids newly present, remove
rids no longer present. This tolerates any queue mutation path (append, pop,
list reassignment on retract/preempt) without hooking each mutation, and keeps
the tree exact at scheduling time with no cross-step leakage.

> This primitive is the crux of the hosting question in §7.

### 5.2 peek-lpm (stage 1)

Same scheduling *order* as LPM — primary key is the KV-cache prefix-match length,
with the same in-batch deduplication — but the pending tree is maintained
incrementally rather than rebuilt each call, and it is the substrate cLPM builds
on. Intended as a low-risk, behavior-preserving foundation.

### 5.3 peek (stage 2) — the full PEEK cluster scheduler

`--schedule-policy peek` is PEEK's default, most-performant scheduling policy.
It refines LPM with three signals LPM cannot cheaply produce, all read off the
pending tree — **cLPM + GM + DL** in the paper's notation (§3.2):

- **cLPM — cluster-LPM sort.** Each request gets a **section**: `0 = warm`
  (already cached beyond a threshold), `1 = cold pioneer` (first-seen of a
  shared cold prefix), `2 = cold sibling` (a later request sharing a pioneer's
  claim key). Siblings sort behind their pioneer so the shared prefix is warmed
  once. Within a section, requests are keyed by
  `(−main_hit, −req_score, −cluster_size, arrival)` — cache-warm and
  dense-cluster requests first.
- **GM — group-major batching.** Within a section, members of the same cluster
  are emitted **contiguously** (clusters ranked by depth × size), so an entire
  cluster admits as one batch and its shared prefix is reused across the batch
  instead of being interleaved with unrelated work.
- **DL — dynamic-lane fairness.** The cLPM+GM order is stride-interleaved with an
  oldest-first fairness lane whose share is **recomputed each step** from the
  queue's singleton fraction and oldest-singleton wait (EMA-smoothed). This
  recovers the singleton starvation that aggressive cluster batching would
  otherwise cause. (A static lane share, `big_lane_share`, is available as a
  simpler fallback.)

Degenerates to LPM ordering when the queue has no sharing structure (guarded by
`has_sharing`).

### 5.4 Demand-aware eviction (stage 3)

Each scheduling step, per-prefix (SGLang) / per-block (vLLM) **demand** is
refreshed from the waiting queue. The evictor then prefers victims **no queued
request needs** (demand 0), spilling into demanded blocks lowest-demand-first
only when necessary, LRU within a tier. When nothing is demanded it falls back
to the existing fast path, so non-sharing workloads pay only one extra list
walk. Rebuilt from scratch each step → no leakage.

---

## 6. Per-engine integration

### 6.1 SGLang

- **Scheduling.** Add `CacheAwarePolicy.PEEK_LPM` / `PEEK_CLPM`
  (`--schedule-policy peek-lpm|peek`). `SchedulePolicy.calc_priority`
  gains a branch that syncs the pending tree (diff-based) and reorders the
  waiting queue. peek-lpm reuses `_sort_by_longest_prefix`; peek calls the
  cLPM ordering.
- **Eviction.** Add `--radix-eviction-policy peek`. A `PeekDemandStrategy` reads
  per-node demand from the pending tree (max token-weighted `pending_demand`
  over the node's ancestor paths). The tree is **shared** with the schedule
  policy via `tree_cache.pending_tree`, so peek eviction requires a `peek-*`
  schedule policy to keep it in sync (without one it degrades to LRU).
- Files touched: `managers/schedule_policy.py`, `mem_cache/radix_cache.py`,
  `mem_cache/evict_policy.py`, `server_args.py` (+ tests). Diff is small; the
  substance is the tree (§7).

### 6.2 vLLM (v1)

- **Scheduling.** Add `SchedulingPolicy.PEEK_LPM` / `PEEK_CLPM`
  (`--scheduling-policy peek-lpm|peek`), backed by the FCFS queue, reordered
  in `Scheduler.schedule()` before allocation. Cache-hit length per request uses
  the coordinator's `find_longest_cache_hit` (the lookup `get_computed_blocks`
  already performs, minus its stats side effect).
- **Eviction.** Add `--enable-peek-eviction`. `KVCacheBlock` gains a
  `peek_demand` int; `FreeKVCacheBlockQueue.popleft_peek` does the two-tier
  victim pick; `Scheduler._refresh_peek_demand` rebuilds per-block demand each
  step from the waiting queue's block hashes. **Block-hash based and independent
  of the pending tree**, so it works with any scheduler.
- Files touched: `v1/core/sched/scheduler.py`, `v1/core/sched/request_queue.py`,
  `v1/core/block_pool.py`, `v1/core/kv_cache_utils.py`, `config/cache.py`,
  `engine/arg_utils.py` (+ tests).

Note the asymmetry: vLLM's eviction signal is naturally block-hash demand (no
tree needed); SGLang's is token-path demand over the pending tree. cLPM on both
engines does need the tree.

---

## 7. The open question — how should the pending tree be hosted?

This is the decision we most want maintainer input on. The reference
implementation currently factors the tree + cLPM ordering into a separate
package (a Rust core with Python bindings) that the engine imports. That keeps
the engine diff tiny, but it means the substance lives outside the PR and adds a
third-party dependency to a core path — which we recognize is a poor fit for
upstream review. Options, with tradeoffs:

| Option | Pros | Cons |
|---|---|---|
| **A. Native, in-tree (pure Python)** | Self-contained; fully reviewable in the PR; no external dep; consistent with SGLang already rebuilding an aux radix tree per call | More code in the engine; some CPU cost vs a compiled core (optimizable later; can move to the engine's existing C++/Rust if warranted) |
| **B. Optional extra dependency** | Tiny engine diff; shared, tuned implementation; compiled speed | Adds a third-party dep to core scheduling/eviction; substance not in the diff; harder to review/accept |
| **C. Vendor a compiled core into the engine** | Self-contained; compiled speed | Adds a build step (Rust/pyo3 or C++) to the engine's toolchain; heavy |

**Our recommendation is A** for upstream: reimplement the pending tree + cLPM
natively in-tree so the PR stands on its own, and keep the compiled core as the
research/reference implementation. We would like maintainers to confirm the
preference (A / B / C, or a variant) so the PRs are built to it rather than
reworked after the fact.

---

## 8. Compatibility, performance, evaluation

- **Default-unchanged.** Every mechanism is opt-in. With any existing policy the
  new code paths are never entered; the new `KVCacheBlock`/`TreeNode` fields are
  a single int initialized to 0.
- **Overhead.** Tree maintenance is O(1) amortized per arrival/schedule; the
  per-step reconcile is a set diff over the waiting queue. Eviction adds one free
  -list walk only when something is demanded. cLPM's cluster reads are amortized
  over the queue.
- **Evaluation.** Full design and evaluation are in the PEEK paper (§4): five
  workloads (shared-prompt chat, long-document RAG, multi-GPU 70B, agentic
  Mooncake, singleton chat), up to 4×H100 (DP=2 over TP=2). Over each engine's
  **strongest stock baseline** (SGLang `lpm`+LRU / vLLM `fcfs`+APC+LRU), PEEK
  delivers up to:

  | Metric (up to) | SGLang | vLLM |
  |---|---|---|
  | Cache-hit rate | **3.0×** | **2.6×** |
  | TTFT | **7.9×** | **7.1×** |
  | End-to-end latency | **6.7×** | **5.5×** |
  | Throughput | **3.6×** | **4.5×** |

  …while **matching baselines within noise** on workloads with no exploitable
  prefix structure, and wins hold as KV-cache pressure and parallelism scale.

  Per-workload, full PEEK (cLPM+GM+DL+PE) vs each engine's strongest baseline
  (ranges span load levels / cells; SGLang / vLLM):

  | Workload | Sharing | Cache hit | TTFT | E2E | Throughput |
  |---|---|---|---|---|---|
  | W1 shared-prompt chat | high | +28–61pp / +20–49pp | 3.7–7.9× / 1.9–4.4× | 2.9–6.7× / 1.2–3.6× | 1.5–3.6× / 1.2–2.5× |
  | W2 long-document RAG | high | 1.27–1.39× / 1.12–1.36× | 2.2–2.6× / 3.4–3.7× | 2.0–2.4× / 1.6–2.3× | ~1.0–1.14× |
  | W3 70B multi-GPU (DP=1) | high | +40–59pp | 3.0–6.7× | 2.6–5.5× | 2.1–4.5× |
  | W4 prefix-coherent | already warm | no-regress (±1–2pp) | ±5–7% | within 2.5% | within ~2% |
  | W5 singleton chat | none | no-regress | within ~2% | within ~2% | within ~2% |

  W4/W5 establish the **no-regress** claim: where there is no exploitable
  structure, a `has_sharing` guard short-circuits PEEK to the stock path. The
  dominant lever is cluster-aware admission (cLPM); eviction (PE) adds 0–3% on
  top and never *creates* locality the scheduler didn't establish.

  These numbers are from PEEK's reference implementation (paper §4); the
  mechanisms in this RFC are its productized, in-tree form. Per-branch engine
  numbers will accompany the PRs.

---

## 9. Reference implementation

Working, DCO-signed branches (staged peek-lpm → peek → peek-eviction), CPU
unit tests included:

- SGLang: *[link once public]*
- vLLM: *[link once public]*
- Pending tree + cLPM + demand strategy (current compiled core):
  https://github.com/xiexbing/peek

*(Reference branches are private during pre-review; happy to share on request or
publish once this RFC has a direction.)*

---

## 10. Alternatives considered

- **Score-based cache-aware scheduling.** An earlier design scored requests by a
  weighted objective; retired in favor of LPM-on-pending-tree + cLPM sections,
  which are simpler and reproduce LPM exactly in the no-sharing case.
- **Eviction via per-node reference counts, no tree (SGLang).** A self-contained
  variant refreshes a per-`TreeNode` count each step with no pending tree or
  dependency. Viable and even simpler if maintainers prefer to keep eviction
  fully decoupled from scheduling; noted as a fallback under Option A.
- **Do nothing / app-level prefix pinning.** Doesn't generalize across the
  workloads above and pushes cache policy into every application.

---

## 11. Rollout plan

1. Land **peek-lpm** (behavior-preserving foundation + tree lifecycle).
2. Land **peek** (the scheduling win; needs benchmarks).
3. Land **peek eviction** (SGLang: pairs with a peek scheduler; vLLM:
   standalone).

Each stage is a separate PR. Happy to gate stage 2/3 behind further discussion.

---

## 12. References

- Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zhen Zheng. *"PEEK: Predictive
  Queue-Informed KV Cache Management for LLM Serving,"* 2026.
  https://github.com/xiexbing/peek
