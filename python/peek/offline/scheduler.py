# Copyright 2026 Bing Xie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Peek server-side scheduler hooks.

All feature-gating logic lives here -- the patches injected into vLLM and SGLang
are thin, unconditional calls into these functions.  Each hook internally decides
whether to activate (based on config flags, sharing detection, throttling) so
that non-sharing workloads pay zero overhead.

Design principles:
  1. Patches are minimal: a single function call, no if-statements.
  2. Self-detection: each hook checks whether the opportunity exists
     (prefix sharing, queue-aware eviction enabled, etc.) before doing work.
  3. Graceful skip: when conditions aren't met, the hook returns immediately.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol, Sequence

from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Sharing detection
# ---------------------------------------------------------------------------

def detect_sharing(prefix_keys: Sequence[Any]) -> bool:
    """Return True if any two items share the same prefix key.

    O(n) check via a set.  When this returns False the workload has no
    prefix sharing, so the caller can skip reorder and eviction scanning.
    """
    seen: set[Any] = set()
    for key in prefix_keys:
        if key in seen:
            return True
        seen.add(key)
    return False


# ---------------------------------------------------------------------------
# Prefix-aware request reordering (server-side, DFS-weight style)
# ---------------------------------------------------------------------------

class _HasBlockHashes(Protocol):
    """Minimal interface for a waiting request object."""
    block_hashes: Sequence[Any]


def reorder_requests_by_prefix(
    new_requests: list[Any],
    other_requests: list[Any],
) -> list[Any]:
    """Reorder *new_requests* by prefix group (DFS-weight style).

    Groups requests by their first block hash, sorts groups by size
    (largest first), and within each group sorts by sequence length
    (shortest first).  Returns the full reordered list
    (reordered new + other).
    """
    if len(new_requests) <= 1:
        return new_requests + other_requests

    groups: dict[Any, list[Any]] = defaultdict(list)
    for r in new_requests:
        groups[r.block_hashes[0]].append(r)

    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    reordered: list[Any] = []
    for grp in sorted_groups:
        grp.sort(key=lambda r: len(r.block_hashes))
        reordered.extend(grp)

    return reordered + other_requests


# ---------------------------------------------------------------------------
# Queue-aware eviction: reference counting
# ---------------------------------------------------------------------------

class _BlockPool(Protocol):
    """Minimal interface for a block pool that supports queue ref counts."""
    def reset_queue_ref_counts(self) -> None: ...
    def inc_queue_ref_count(self, blocks: list[Any]) -> None: ...


def update_queue_ref_counts(
    new_requests: list[Any],
    block_pool: Any,
    get_cached_fn: Any,
    group_ids: list[int],
) -> None:
    """Reset and recompute queue_ref_count for all blocks.

    Walks each waiting request's block hashes, looks up cached blocks,
    and increments their ref count.  Stops at the first cache miss per
    request (blocks beyond the miss boundary aren't cached).
    """
    block_pool.reset_queue_ref_counts()
    for req in new_requests:
        for bh in req.block_hashes:
            blocks = get_cached_fn(bh, group_ids)
            if blocks:
                block_pool.inc_queue_ref_count(blocks)
            else:
                break


# ---------------------------------------------------------------------------
# Direct cache state queries
# ---------------------------------------------------------------------------

def sglang_cached_prefix_length(tree_cache: Any, token_ids: list[int]) -> int:
    """Query SGLang's radix tree for contiguous cached prefix length.

    Walks the radix tree following *token_ids*, counting tokens that are
    present in the cache.  Stops at the first miss.
    """
    node = tree_cache.root_node
    cached = 0
    i = 0
    n = len(token_ids)
    while i < n:
        token = token_ids[i]
        child = node.children.get(token)
        if child is None:
            break
        # RadixCache nodes store a key (token sequence) at each edge.
        # The child's key length tells us how many tokens this edge covers.
        key_len = len(child.key) if hasattr(child, "key") and child.key is not None else 1
        # Verify the full edge matches
        if i + key_len > n:
            break
        if key_len > 1:
            edge_tokens = token_ids[i : i + key_len]
            if hasattr(child, "key") and list(child.key) != list(edge_tokens):
                break
        # Check if this node has allocated KV blocks (is actually cached)
        if hasattr(child, "value") and child.value is None:
            break
        cached += key_len
        i += key_len
        node = child
    return cached


def vllm_cached_prefix_length(
    block_pool: Any,
    block_hashes: list[Any],
    group_ids: list[int],
) -> int:
    """Query vLLM's block pool for contiguous cached prefix length.

    Walks *block_hashes*, calling ``get_cached_block`` for each.
    Stops at the first miss and returns the number of cached blocks.
    """
    count = 0
    for bh in block_hashes:
        blocks = block_pool.get_cached_block(bh, group_ids)
        if blocks:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Group-level scheduling
# ---------------------------------------------------------------------------

def _group_and_score_sglang(
    waiting_queue: list[Any],
    tree_cache: Any,
    min_prefix_len: int = 1,
) -> list[Any]:
    """Group waiting requests by shared prefix, score by cache state.

    Returns the reordered request list: groups sorted by
    ``cached_prefix_length * group_size`` descending, requests within
    each group sorted by sequence length ascending (shorter first for
    better packing).
    """
    seqs: list[list[int]] = []
    for r in waiting_queue:
        ids = getattr(r, "origin_input_ids", None)
        if ids is not None:
            seqs.append(list(ids))
        else:
            seqs.append([])

    trie = PrefixTrie()
    for idx, seq in enumerate(seqs):
        trie.insert(seq, idx)

    groups = trie.prefix_groups_with_tokens(min_prefix_len)

    # Collect ungrouped indices (singletons not in any group)
    grouped_indices: set[int] = set()
    for _, indices in groups:
        grouped_indices.update(indices)
    for idx in range(len(waiting_queue)):
        if idx not in grouped_indices:
            groups.append((seqs[idx], [idx]))

    # Score each group by cached_prefix_length * group_size
    scored: list[tuple[float, list[int], list[int]]] = []
    for prefix_tokens, indices in groups:
        if prefix_tokens:
            cached_len = sglang_cached_prefix_length(tree_cache, prefix_tokens)
        else:
            cached_len = 0
        score = cached_len * len(indices)
        scored.append((score, prefix_tokens, indices))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Build reordered list: within each group, sort by sequence length asc
    reordered: list[Any] = []
    for _, _, indices in scored:
        group_reqs = [(len(seqs[i]), i) for i in indices]
        group_reqs.sort()
        for _, idx in group_reqs:
            reordered.append(waiting_queue[idx])

    return reordered


def sglang_group_schedule(
    waiting_queue: list[Any],
    tree_cache: Any,
    reorder_threshold: int = 2,
) -> None:
    """Reorder SGLang's waiting queue using group-level cache-aware scheduling.

    Groups requests by shared prefix, scores each group by
    ``cached_prefix_length * group_size``, and schedules the
    highest-scoring groups first.

    Mutates *waiting_queue* in-place.  Skips when queue depth is below
    *reorder_threshold* (LPM handles shallow queues well).
    """
    if len(waiting_queue) < reorder_threshold:
        return

    if not detect_sharing_sglang(waiting_queue):
        return

    reordered = _group_and_score_sglang(waiting_queue, tree_cache)
    waiting_queue.clear()
    waiting_queue.extend(reordered)


def vllm_group_schedule(
    new_reqs: list[Any],
    block_pool: Any,
    group_ids: list[int],
    reorder_threshold: int = 2,
) -> list[Any]:
    """Reorder vLLM's new requests using group-level cache-aware scheduling.

    Groups requests by shared block hash prefix, scores each group by
    ``cached_blocks * group_size``, and schedules highest-scoring groups first.

    Returns the reordered list.  Skips when queue depth is below
    *reorder_threshold* or when there's no prefix sharing.
    """
    if len(new_reqs) <= 1 or len(new_reqs) < reorder_threshold:
        return new_reqs

    first_hashes = [r.block_hashes[0] for r in new_reqs
                    if len(r.block_hashes) > 0]
    if not detect_sharing(first_hashes):
        return new_reqs

    # Build trie from block hashes (treat each hash as a "token")
    trie = PrefixTrie()
    hash_seqs: list[list[int]] = []
    for idx, r in enumerate(new_reqs):
        seq = list(r.block_hashes)
        hash_seqs.append(seq)
        trie.insert(seq, idx)

    groups = trie.prefix_groups_with_tokens(min_prefix_len=1)

    # Collect ungrouped
    grouped_indices: set[int] = set()
    for _, indices in groups:
        grouped_indices.update(indices)
    for idx in range(len(new_reqs)):
        if idx not in grouped_indices:
            groups.append((hash_seqs[idx], [idx]))

    # Score each group
    scored: list[tuple[float, list[int]]] = []
    for prefix_hashes, indices in groups:
        if prefix_hashes:
            cached = vllm_cached_prefix_length(block_pool, prefix_hashes, group_ids)
        else:
            cached = 0
        score = cached * len(indices)
        scored.append((score, indices))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Flatten: within each group, sort by block_hashes length ascending
    reordered: list[Any] = []
    for _, indices in scored:
        group_reqs = [(len(hash_seqs[i]), i) for i in indices]
        group_reqs.sort()
        for _, idx in group_reqs:
            reordered.append(new_reqs[idx])

    return reordered


# ---------------------------------------------------------------------------
# vLLM top-level hook: single entry point called from patched scheduler
# ---------------------------------------------------------------------------

# Module-level throttle state keyed by scheduler id.
_vllm_counters: dict[int, int] = {}
_VLLM_THROTTLE_INTERVAL = 1

# Lazy-init VllmPeekEngine instances keyed by scheduler id.
_vllm_engines: dict[int, Any] = {}


def vllm_on_schedule(scheduler: Any) -> None:
    """Single entry point for vLLM's patched Scheduler.schedule().

    Called unconditionally from the patch.  Internally checks:
      1. Is queue-aware eviction enabled?  (env var / config flag)
      2. Is prefix caching active?
      3. Are there waiting requests?
      4. Throttle: only run full scan every N calls.
      5. Is there prefix sharing among waiting requests?

    If any check fails, returns immediately with zero overhead beyond
    a few attribute lookups.
    """
    # 1. Feature gate: queue-aware eviction must be enabled
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    if not getattr(block_pool, "enable_queue_aware_eviction", False):
        return

    # 2. Prefix caching must be active
    if not getattr(kv_mgr, "enable_caching", False):
        return

    # 3. Must have waiting requests
    waiting = getattr(scheduler, "waiting", None)
    if not waiting:
        return

    # 4. Throttle: only run every N calls (1 = every call)
    sid = id(scheduler)
    if _VLLM_THROTTLE_INTERVAL > 1:
        count = _vllm_counters.get(sid, 0) + 1
        _vllm_counters[sid] = count
        if count % _VLLM_THROTTLE_INTERVAL != 1:
            return

    # 5. Server-side scheduling: only when PEEK_OFFLINE_SERVER_REORDER=1
    #    Offline mode uses client-side reorder; online (Poisson) needs this.
    import os
    if os.environ.get("PEEK_OFFLINE_SERVER_REORDER") != "1":
        # Queue-aware eviction only: update ref counts without reordering
        new_reqs = [r for r in waiting
                    if r.num_computed_tokens == 0
                    and len(r.block_hashes) > 0]
        if len(new_reqs) > 1:
            first_hashes = [r.block_hashes[0] for r in new_reqs]
            if detect_sharing(first_hashes):
                group_ids = (
                    list(range(len(kv_mgr.managers)))
                    if hasattr(kv_mgr, "managers") else [0]
                )
                update_queue_ref_counts(
                    new_reqs, block_pool, block_pool.get_cached_block, group_ids,
                )
        return

    # 6. Full coordinated scheduling + adaptive protection via VllmPeekEngine
    sid = id(scheduler)
    engine = _vllm_engines.get(sid)
    if engine is None:
        from peek.offline.engine_vllm import VllmPeekEngine
        group_ids = (
            list(range(len(kv_mgr.managers)))
            if hasattr(kv_mgr, "managers") else [0]
        )
        engine = VllmPeekEngine(block_pool, group_ids)
        _vllm_engines[sid] = engine
    engine.run(waiting)


def vllm_reorder_waiting_queue(
    new_reqs: list[Any],
    reorder_threshold: int = 2,
) -> list[Any]:
    """Reorder vLLM's new requests using Peek's trie-based DFS.

    Uses ``block_hashes`` as the prefix key sequence (block hashes are
    sequential and requests sharing a prefix have identical hash prefixes).

    Guard condition: when queue depth >= *reorder_threshold*, client-side
    reorder has already grouped requests -- skip server-side reorder.
    When queue is shallow (< threshold), apply server-side trie DFS
    since vLLM has no native prefix-aware scheduling.
    """
    if len(new_reqs) <= 1 or len(new_reqs) >= reorder_threshold:
        return new_reqs

    from peek.offline.reorder import reorder_for_prefix_sharing

    # Use block_hashes as proxy token sequences for the trie.
    # block_hashes are already sequential (chain-hashed), so prefix
    # sharing in token space = prefix sharing in hash space.
    seqs = [list(r.block_hashes) for r in new_reqs]
    order = reorder_for_prefix_sharing(
        seqs,
        min_sharing_depth=1,     # block-level, not token-level
        min_avg_sharing_depth=1,  # single shared block is enough
    )

    # Check if reorder was skipped (identity permutation)
    if all(order[i] == i for i in range(len(order))):
        return new_reqs

    return [new_reqs[i] for i in order]


def schedule_hook_vllm(
    waiting: list[Any],
    kv_cache_manager: Any,
) -> None:
    """Legacy entry point -- delegates to VllmPeekEngine.

    Kept for backward compatibility with existing tests and callers.
    """
    from peek.offline.engine_vllm import VllmPeekEngine

    block_pool = getattr(kv_cache_manager, "block_pool", None)
    if block_pool is None:
        return
    group_ids = (
        list(range(len(kv_cache_manager.managers)))
        if hasattr(kv_cache_manager, "managers") else [0]
    )
    sid = id(kv_cache_manager)
    engine = _vllm_engines.get(sid)
    if engine is None:
        engine = VllmPeekEngine(block_pool, group_ids)
        _vllm_engines[sid] = engine
    engine.run(waiting)


# ---------------------------------------------------------------------------
# SGLang top-level hooks
# ---------------------------------------------------------------------------

def sglang_reorder_waiting_queue(
    waiting_queue: list[Any],
    tree_cache: Any = None,
    reorder_threshold: int = 2,
) -> None:
    """Reorder SGLang's waiting queue using group-level cache-aware scheduling.

    When *tree_cache* is provided, uses group-level scheduling: groups
    requests by shared prefix, scores each group by
    ``cached_prefix_length * group_size``, schedules highest-scoring
    groups first.

    When *tree_cache* is None, falls back to trie-based DFS grouping
    without cache awareness.

    Mutates *waiting_queue* in-place.
    """
    if len(waiting_queue) < reorder_threshold:
        return

    if tree_cache is not None:
        sglang_group_schedule(waiting_queue, tree_cache, reorder_threshold)
        return

    # Fallback: DFS grouping without cache awareness
    from peek.offline.reorder import reorder_for_prefix_sharing

    seqs: list[list[int]] = []
    for r in waiting_queue:
        ids = getattr(r, "origin_input_ids", None)
        if ids is not None:
            seqs.append(list(ids))
        else:
            seqs.append([])

    order = reorder_for_prefix_sharing(seqs)
    if all(order[i] == i for i in range(len(order))):
        return

    reordered = [waiting_queue[i] for i in order]
    waiting_queue.clear()
    waiting_queue.extend(reordered)


def detect_sharing_sglang(
    waiting_queue: list[Any],
    key_len: int = 64,
) -> bool:
    """Check if any two requests in the SGLang waiting queue share a prefix.

    Uses the first *key_len* tokens as the sharing key -- the same key
    used for grouping.  A single-token check is too coarse (most prompts
    start with <bos>) and causes PeekEngine to activate on workloads
    with zero real sharing.
    """
    seen: set[tuple] = set()
    for r in waiting_queue:
        ids = getattr(r, "origin_input_ids", None)
        if ids and len(ids) > 0:
            key = tuple(ids[:key_len]) if len(ids) >= key_len else tuple(ids)
            if key in seen:
                return True
            seen.add(key)
    return False


def _fast_prefix_group_reorder(
    waiting_queue: list[Any],
    key_len: int = 64,
) -> None:
    """Lightweight O(n) reorder: group requests by prefix hash.

    Groups requests sharing the same first *key_len* tokens, then
    orders groups by size (largest first).  Within each group,
    preserves arrival order (FCFS).  Mutates *waiting_queue* in-place.

    This avoids building a trie or walking the radix cache in the
    hot scheduling loop -- just a hash-based grouping.
    """
    groups: dict[tuple, list[Any]] = defaultdict(list)
    for r in waiting_queue:
        ids = getattr(r, "origin_input_ids", None)
        if ids is not None and len(ids) >= key_len:
            key = tuple(ids[:key_len])
        elif ids is not None and len(ids) > 0:
            key = tuple(ids)
        else:
            key = ()
        groups[key].append(r)

    # Sort groups by size descending (schedule largest groups first
    # to maximize prefix cache reuse before eviction)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    waiting_queue.clear()
    for grp in sorted_groups:
        waiting_queue.extend(grp)


def sglang_pre_schedule(tree_cache: Any, waiting_queue: list[Any]) -> bool:
    """Called at the top of SGLang's _compute_prefix_matches.

    Performs group-level prefix matching at O(G x D) cost instead of
    per-request O(N x D).  For each prefix group:
      1. Match ONE representative against the radix tree.
      2. Copy match results to all group members.
      3. Score by cached_length x group_size.
      4. Increment queue ref counts once per group (not per request).

    Reorders the waiting queue by score (highest first) and marks all
    requests as pre-matched so the caller can skip per-request radix
    walks.

    Returns True if queue-aware eviction is active.
    """
    import time as _time_mod
    _t0 = _time_mod.perf_counter()

    is_queue_aware = getattr(tree_cache, "eviction_policy", "") == "queue-aware"
    if not detect_sharing_sglang(waiting_queue):
        tree_cache._peek_grouped = False
        return False

    # Queue-aware eviction: reset previous ref counts
    if is_queue_aware:
        prev = getattr(tree_cache, "_peek_prev_protected", None)
        if prev is not None:
            for node in prev:
                walk = node
                while walk is not None and walk != tree_cache.root_node:
                    walk.queue_ref_count = 0
                    walk = walk.parent
        else:
            tree_cache.reset_all_queue_refs()

    # Group requests by shared prefix (first 64 tokens as key)
    _KEY_LEN = 64
    groups: dict[tuple, list[Any]] = defaultdict(list)
    for r in waiting_queue:
        ids = getattr(r, "origin_input_ids", None)
        if ids is not None and len(ids) >= _KEY_LEN:
            key = tuple(ids[:_KEY_LEN])
        elif ids is not None and len(ids) > 0:
            key = tuple(ids)
        else:
            key = ()
        groups[key].append(r)

    # Match ONE representative per group against the radix tree
    # and copy results to all members -- O(G x D) instead of O(N x D)
    import time as _time
    from sglang.srt.mem_cache.radix_cache import MatchPrefixParams, RadixKey

    group_data: list[tuple[tuple, list[Any], int, float]] = []
    # (key, members, cached_len, eviction_risk)
    total_requests = len(waiting_queue)

    # Use read-only match to avoid poisoning LRU timestamps.
    # Scoring all groups with regular match_prefix refreshes every
    # group's last_access_time, making LRU eviction essentially random
    # under memory pressure.  Only the actually-admitted requests should
    # update LRU (via init_next_round_input in the scheduler loop).
    _match_fn = getattr(tree_cache, "match_prefix_readonly", tree_cache.match_prefix)

    for key, members in groups.items():
        rep = members[0]
        prefix_ids = rep.origin_input_ids + rep.output_ids
        extra_key = rep.extra_key

        match_result = _match_fn(
            MatchPrefixParams(
                key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
            )
        )

        # Copy match results to all group members
        for r in members:
            r.prefix_indices = match_result.device_indices
            r.last_node = match_result.last_device_node
            r.last_host_node = match_result.last_host_node
            r.host_hit_length = match_result.host_hit_length
            r._peek_matched = True  # skip per-request radix walk

        cached_len = len(match_result.device_indices)

        # Eviction risk: how close is this prefix to being evicted?
        # Lower last_access_time = older = higher eviction risk.
        # Normalized: 0 = just accessed (safe), 1 = oldest (at risk).
        node = match_result.last_device_node
        if node is not None and cached_len > 0:
            last_access = getattr(node, "last_access_time", 0.0)
            eviction_risk = max(0.0, _time.monotonic() - last_access)
        else:
            eviction_risk = 0.0

        group_data.append((key, members, cached_len, eviction_risk))

    # Compute future_refs per group: count of requests in the SAME
    # group beyond the first wave.  High future_refs = defer this group
    # (keep its prefix cached for later use).
    # Estimate wave size as min(total_requests, 32) -- rough batch size.
    _BATCH_EST = 32
    scored: list[tuple[float, list[Any]]] = []
    for key, members, cached_len, eviction_risk in group_data:
        group_size = len(members)
        future_refs = max(0, group_size - _BATCH_EST)

        # Coordinated score from paper Eq. 1:
        #   cache_frac x 10^4        -> exploit cached (schedule first)
        #   - future_refs             -> defer if many future requests need it
        #   + eviction_risk x 10^2   -> urgent: schedule before eviction
        #   + count x 10^-3          -> tiebreaker: larger groups first
        prefix_len = len(getattr(members[0], "origin_input_ids", []))
        cache_frac = cached_len / max(prefix_len, 1)
        score = (
            cache_frac * 1e4
            - future_refs
            + eviction_risk * 1e2
            + group_size * 1e-3
        )
        scored.append((score, members))

    # Reorder: highest-scoring groups first
    scored.sort(key=lambda x: x[0], reverse=True)
    waiting_queue.clear()
    for _, members in scored:
        waiting_queue.extend(members)

    tree_cache._peek_grouped = True

    # Queue-aware eviction: adaptive protection.
    # Skip if using LRU eviction (online mode without queue-aware).
    # Return False so caller skips per-request inc_queue_ref.
    if not is_queue_aware:
        _elapsed = _time_mod.perf_counter() - _t0
        _call_count = getattr(tree_cache, "_peek_profile_count", 0) + 1
        _total_time = getattr(tree_cache, "_peek_profile_total", 0.0) + _elapsed
        tree_cache._peek_profile_count = _call_count
        tree_cache._peek_profile_total = _total_time
        if _call_count % 100 == 0:
            _n = len(waiting_queue)
            _g = len(groups)
            print(f"  [peek-profile] calls={_call_count} avg={_total_time/_call_count*1000:.2f}ms "
                  f"last={_elapsed*1000:.2f}ms N={_n} G={_g}", flush=True)
        return False

    # Adaptive protection: only protect groups that fit in cache.
    # When #groups <= cache capacity, protect all (no ossification).
    # When #groups > cache capacity, protect only the top-K groups
    # whose prefixes fit in cache, let LRU rotate the rest.
    #
    # Estimate cache capacity in prefix slots:
    #   total_cached_tokens / avg_prefix_len_per_group
    # Use tree_cache.total_size() for total cached tokens.
    total_cached = 0
    try:
        total_cached = tree_cache.total_size()
    except Exception:
        pass

    num_groups = len(scored)
    if num_groups > 0 and total_cached > 0:
        avg_prefix_len = max(1, sum(
            len(getattr(members[0], "origin_input_ids", []))
            for _, members in scored
        ) / num_groups)
        cache_capacity_groups = int(total_cached / avg_prefix_len)
    else:
        cache_capacity_groups = num_groups

    # Protect at most 3/4 of cache capacity -- leave 25% for LRU
    # rotation so new groups can get cached.  This balances protection
    # (avoid evicting needed prefixes) with rotation (avoid ossification).
    protect_k = min(num_groups, max(1, cache_capacity_groups * 3 // 4))

    protected_nodes: list[Any] = []
    for i, (_, members) in enumerate(scored):
        if i >= protect_k:
            break
        rep = members[0]
        node = getattr(rep, "last_node", None)
        if node is not None and len(getattr(rep, "prefix_indices", [])) > 0:
            tree_cache.inc_queue_ref(node)
            protected_nodes.append(node)

    # Track for next cycle's targeted reset -- O(K x depth) not O(T)
    tree_cache._peek_prev_protected = protected_nodes

    _elapsed = _time_mod.perf_counter() - _t0
    # Log every 100 calls
    _call_count = getattr(tree_cache, "_peek_profile_count", 0) + 1
    _total_time = getattr(tree_cache, "_peek_profile_total", 0.0) + _elapsed
    tree_cache._peek_profile_count = _call_count
    tree_cache._peek_profile_total = _total_time
    if _call_count % 100 == 0:
        _n = len(waiting_queue)
        _g = len(groups)
        print(f"  [peek-profile] calls={_call_count} avg={_total_time/_call_count*1000:.2f}ms "
              f"last={_elapsed*1000:.2f}ms N={_n} G={_g}", flush=True)

    return True


def sglang_post_match_reorder(waiting_queue: list[Any]) -> None:
    """Reorder waiting queue AFTER prefix matching using match results.

    Groups requests by their matched radix node (last_node), scores
    each group by total cached tokens, and reorders so highest-scoring
    groups are scheduled first.  This achieves cache-aware scheduling
    at O(N) cost -- no trie construction, no extra cache walk -- by
    piggybacking on the prefix matching that already ran.

    Called at the end of _compute_prefix_matches, after all requests
    have their prefix_indices and last_node populated.
    """
    if len(waiting_queue) <= 1:
        return

    # Group by last_node identity (requests matching the same prefix
    # land on the same radix tree node)
    groups: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for i, r in enumerate(waiting_queue):
        node_id = id(getattr(r, "last_node", None))
        cached_len = len(getattr(r, "prefix_indices", []))
        groups[node_id].append((cached_len, r))

    # Score each group: total cached tokens (= cached_len x group_size
    # since all members share the same prefix match)
    scored: list[tuple[int, list[Any]]] = []
    for node_id, members in groups.items():
        total_cached = sum(cl for cl, _ in members)
        reqs = [r for _, r in members]
        scored.append((total_cached, reqs))

    # Highest score first -> exploit cached prefixes
    scored.sort(key=lambda x: x[0], reverse=True)

    waiting_queue.clear()
    for _, reqs in scored:
        waiting_queue.extend(reqs)


def sglang_should_run_prefix_matching(tree_cache: Any) -> bool:
    """Check if FCFS should run prefix matching for Peek.

    In stock SGLang, FCFS skips _compute_prefix_matches entirely.
    Peek always needs prefix matching to run so that the server-side
    cache-aware reorder in sglang_pre_schedule can execute.
    Returns True when Peek patches are installed (detected by the
    presence of queue_ref_count on the root node or queue-aware policy).
    """
    if getattr(tree_cache, "eviction_policy", "") == "queue-aware":
        return True
    # Also run for LRU when Peek's reorder is active (root has queue_ref_count attr)
    root = getattr(tree_cache, "root_node", None)
    return root is not None and hasattr(root, "queue_ref_count")


# ---------------------------------------------------------------------------
# Queue-aware eviction: priority scoring (called from SGLang patch)
# ---------------------------------------------------------------------------

def queue_aware_eviction_priority(node: Any) -> tuple[int, float, float]:
    """Compute eviction priority for a RadixCache TreeNode.

    Returns ``(is_referenced, cost_score, last_access_time)`` where lower
    values are evicted first.  O(1) per node -- uses cached ``node.depth``.

    Hard partition: ``is_referenced`` (0 or 1) ensures unreferenced nodes
    are ALWAYS evicted before ANY referenced node.  This is the key
    mechanism -- it protects prefixes that pending requests need.

    Within each partition:
      - ``cost_score = log(1 + ref_count) * depth`` breaks ties
      - ``last_access_time`` as final tiebreaker (LRU)
    """
    import math
    ref_count = getattr(node, "queue_ref_count", 0)
    is_referenced = 1 if ref_count > 0 else 0
    depth = getattr(node, "depth", 0)
    cost_score = math.log1p(ref_count) * max(depth, 1)
    return (is_referenced, cost_score, node.last_access_time)
