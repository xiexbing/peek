# [Feature] Queue-aware radix cache eviction policy

**Authors:** Bing Xie (@xiexbing), Zhipeng Wang, Masahiro Tanaka, Zhen Zheng
**Target:** `sgl-project/sglang` @ v0.5.14 (rebase onto `main` before opening)

## Summary

Adds a new opt-in radix-cache eviction policy, `queue-aware`, selectable with
`--radix-eviction-policy queue-aware`. It protects KV-cache blocks that
requests **currently in the waiting queue** will reuse, evicting blocks no
queued request needs first. Default behavior is unchanged: with any other
policy the new code paths are never entered.

## Motivation

SGLang's eviction policies (LRU/LFU/SLRU/priority) score blocks only by their
*past* access pattern. Under memory pressure with bursty, prefix-sharing
arrivals (agentic tool-chains, shared system prompts, multi-turn sessions),
LRU can evict a prefix moments before a batch of queued requests that share it
is scheduled, forcing redundant prefill. The waiting queue already tells us
which prefixes are about to be reused; this policy uses that signal. The design
and its evaluation are described in the PEEK paper (see References).

## Design

Two-tier priority (lower value = evicted first):

- **Tier 0 — unreferenced** (`queue_ref_count == 0`): no waiting request needs
  the block. Evicted first, LRU order within the tier.
- **Tier 1 — referenced** (`queue_ref_count > 0`): a waiting request will reuse
  this prefix. Evicted last; within the tier, cheapest-to-recompute first
  (`queue_ref_count * num_tokens`), then LRU.

`queue_ref_count` is rebuilt from scratch each scheduling step in
`SchedulePolicy.calc_priority`, reusing the prefix match the scheduler already
computes for each waiting request (no extra matching pass in the common path).
Counting walks root→matched-node, so every block on a needed prefix is
protected. Rebuilding each step means protection tracks the live queue exactly,
with no cross-step leakage.

## Changes (5 files, +99 / -1)

| File | Change |
|---|---|
| `mem_cache/evict_policy.py` | New `QueueAwareStrategy` |
| `mem_cache/utils.py` | Register `"queue-aware"` in `_EVICTION_POLICY_FACTORIES` |
| `mem_cache/radix_cache.py` | `TreeNode.queue_ref_count`; `inc_queue_ref` / `reset_all_queue_refs`; `_split_node` inheritance |
| `managers/schedule_policy.py` | `_update_queue_refs` hook in `calc_priority` |
| `server_args.py` | Add `"queue-aware"` to `RADIX_EVICTION_POLICY_CHOICES` |

## Default-unchanged guarantee

`_update_queue_refs` returns immediately unless
`tree_cache.eviction_policy == "queue-aware"`. The new `TreeNode` attribute is a
single int initialized to 0. No existing policy, sort path, or matching path is
altered.

## Test plan

`test/srt/test_queue_aware_eviction.py` (CPU-only, no model):

- `queue-aware` resolves via `get_eviction_strategy`.
- Unreferenced blocks always outrank referenced blocks for eviction, even when
  far more recently accessed.
- Within the referenced tier, cheapest-to-recompute is evicted first; LRU
  breaks ties within a tier.
- `inc_queue_ref` accumulates on ancestors and never counts root;
  `reset_all_queue_refs` clears prior refs with no leakage into the next step;
  `_split_node` carries protection to the new parent.

```
python -m pytest test/srt/test_queue_aware_eviction.py -v
```

## Usage

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --schedule-policy lpm \
    --radix-eviction-policy queue-aware
```

## Notes

- This is the eviction half of a larger line of work on queue-informed KV cache
  management. The complementary cache-aware *scheduling* change is intentionally
  left out of this PR to keep it small and independently reviewable; happy to
  open a follow-up RFC for it.
- Follow-up optimization: when paired with a schedule policy that skips prefix
  matching, `_update_queue_refs` matches on demand; this could be elided by
  reusing the load snapshot.

## References

- Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zhen Zheng. "PEEK: Predictive
  Queue-Informed KV Cache Management for LLM Serving," 2026.
  https://github.com/xiexbing/peek
