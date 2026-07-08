# [Core] Queue-aware KV cache eviction

**Author:** Bing Xie (@xiexbing)
**Target:** `vllm-project/vllm` @ v0.24.0 (rebase onto `main` before opening)

## Summary

Adds an opt-in KV-cache eviction mode, `enable_queue_aware_eviction`, that
protects blocks whose prefix is needed by a request in the **waiting queue**,
evicting blocks no queued request needs first. Off by default; when disabled,
`get_new_blocks` takes the existing `popleft_n` (LRU) path unchanged.

## Motivation

vLLM's prefix-cache eviction is pure LRU over the free-block list. Under memory
pressure with prefix-sharing arrivals (shared system prompts, multi-turn
sessions, agentic tool-chains), LRU can evict a prefix moments before a batch of
queued requests that share it is scheduled, forcing redundant prefill. The
waiting queue already knows which prefixes are about to be reused; this uses
that signal to bias eviction.

## Design

Each `KVCacheBlock` gets a `queue_ref_count`, refreshed every scheduling step in
`Scheduler._update_queue_refs()`: for each waiting request, walk its
`block_hashes`, look up the cached blocks, and increment their count. Counts are
rebuilt from scratch each step, so protection tracks the live queue with no
leakage. `FreeKVCacheBlockQueue.popleft_queue_aware` then selects victims in two
tiers (lower = evicted first):

- **Tier 0 — unprotected** (`queue_ref_count == 0`): evicted first, LRU order.
- **Tier 1 — protected** (`queue_ref_count > 0`): only spilled into when too few
  unprotected blocks exist, cheapest (lowest `queue_ref_count`) first.

When nothing is protected it falls back to the fast `popleft_n` path, so the
overhead on non-sharing workloads is one list walk.

## Changes (5 files, +130 / -1)

| File | Change |
|---|---|
| `v1/core/kv_cache_utils.py` | `KVCacheBlock.queue_ref_count` field; `FreeKVCacheBlockQueue.popleft_queue_aware` |
| `v1/core/block_pool.py` | `enable_queue_aware_eviction` flag; `reset/inc_queue_ref_count`; `get_new_blocks` branch |
| `v1/core/sched/scheduler.py` | Enable flag from config; `_update_queue_refs()` refresh in `schedule()` |
| `config/cache.py` | `CacheConfig.enable_queue_aware_eviction` field |
| `engine/arg_utils.py` | `--enable-queue-aware-eviction` CLI flag |

## Default-unchanged guarantee

`enable_queue_aware_eviction` defaults to `False`. `_update_queue_refs` and the
`popleft_queue_aware` branch return immediately when it is off. The new
`KVCacheBlock` field is one int (fits the existing `slots=True` dataclass). No
existing allocation or eviction path changes when the flag is unset.

## Design note: no coordinator plumbing

The flag is set on `BlockPool` by the scheduler after construction rather than
threaded through `get_kv_cache_coordinator` and every coordinator subclass. This
keeps the diff to 5 files and avoids touching hot constructors; happy to switch
to full config threading if maintainers prefer.

## Test plan

`tests/v1/core/test_queue_aware_eviction.py` (CPU-only, no model):

- `queue_ref_count` is a real slotted field.
- Protected blocks are skipped while unprotected blocks remain; spill picks the
  cheapest protected block first; fast path equals `popleft_n` when nothing is
  protected.
- `BlockPool` ref counting accumulates, resets with no cross-step leakage, and
  skips the null block; `get_new_blocks` honors protection when enabled.

```
pytest tests/v1/core/test_queue_aware_eviction.py -v
```

## Usage

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --enable-prefix-caching \
    --enable-queue-aware-eviction
```

## Notes

- This is the eviction half of a broader line of work on queue-informed KV cache
  management. The complementary prefix-aware *scheduling* change is left out to
  keep this PR small and independently reviewable.
- Multi-group (hybrid) caches are handled: ref counting uses all
  `kv_cache_groups`.
