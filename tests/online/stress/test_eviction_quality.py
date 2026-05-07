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

"""Eviction quality: peek's demand-aware strategy vs sglang's LRU.

Scenario: the main KV cache holds a mix of hot shared prefixes (system
prompts used by many pending requests) and cold leaves (unique tails from
long-finished requests). Memory pressure forces eviction. The question:
what gets evicted, and what does that cost?

* LRU: evicts whichever node was accessed longest ago. If a hot system
  prompt has been sitting untouched while cold leaves were recently touched,
  LRU evicts the hot prompt -- forcing every pending req that shared it to
  re-prefill.
* PeekDemandStrategy: evicts lowest (pending_demand, last_access_time) first.
  Zero-demand nodes always evict before nonzero. Hot prompts are protected
  as long as pending reqs depend on them.

Metric: total re-prefill tokens caused by eviction, given the pending queue
that follows.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import time
from typing import List, Set, Tuple

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.evict_policy import LRUStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek.online import PeekDemandStrategy, PendingTree

class _MockAllocator:
    """Minimal allocator satisfying RadixCache.evict's needs."""

    device = "cpu"

    def free(self, _value) -> None:  # pragma: no cover - trivial
        pass

def _make_cache() -> RadixCache:
    return RadixCache.create_simulated(mock_allocator=_MockAllocator())

def _insert(cache: RadixCache, tokens: List[int]) -> None:
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=tokens, extra_key=None),
            value=torch.empty(len(tokens), dtype=torch.bool),
        )
    )

def _match_len(cache: RadixCache, tokens: List[int]) -> int:
    m = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=tokens, extra_key=None))
    )
    return len(m.device_indices)

def _reprefill_cost(pending: List[List[int]], cache: RadixCache) -> int:
    """Cost = sum over pending reqs of (len(tokens) - cache match) -- the
    prefix re-work needed after eviction."""
    total = 0
    for tokens in pending:
        total += len(tokens) - _match_len(cache, tokens)
    return total

def test_peek_eviction_saves_reprefill_vs_lru(capsys: pytest.CaptureFixture) -> None:
    """Construct a cache with one hot prefix (many pending reqs depend) and
    several cold leaves (no dependents). Artificially age the hot prefix
    (LRU would evict it first) and force eviction of ~same tokens as the
    hot prefix. LRU evicts hot -> high re-prefill; peek evicts cold -> zero."""
    hot_prefix = list(range(1, 41))                     # 40 tokens, shared system prompt
    hot_reqs = [hot_prefix + [100 + i] for i in range(20)]  # 20 pending reqs share it

    cold_leaves = [
        [500 + 10 * i + j for j in range(15)] for i in range(5)  # 5 cold paths, 15 tokens each
    ]

    # --- Build two identical caches ---
    cache_lru = _make_cache()
    cache_peek = _make_cache()
    for c in (cache_lru, cache_peek):
        _insert(c, hot_prefix)
        for cold in cold_leaves:
            _insert(c, cold)

    # Set strategies first.
    cache_lru.eviction_strategy = LRUStrategy()
    peek_tree = PendingTree()
    for i, tokens in enumerate(hot_reqs):
        peek_tree.insert(i + 1, tokens)
    cache_peek.eviction_strategy = PeekDemandStrategy(peek_tree)

    # Baseline re-prefill cost (before eviction). NOTE: match_prefix refreshes
    # last_access_time, so we must age AFTER baseline to avoid our aging
    # being overwritten.
    lru_baseline = _reprefill_cost(hot_reqs, cache_lru)
    peek_baseline = _reprefill_cost(hot_reqs, cache_peek)

    # Now age the hot prefix so LRU sees it as oldest.
    now = time.monotonic()
    for c in (cache_lru, cache_peek):
        def classify_and_age(n):
            toks = list(n.key.token_ids) if n.key else []
            if toks and 1 <= toks[0] <= 40:
                n.last_access_time = now - 60.0
            elif toks and toks[0] >= 500:
                n.last_access_time = now - 0.1
            for child in n.children.values():
                classify_and_age(child)
        classify_and_age(c.root_node)

    # --- Evict enough tokens to force choice between hot and cold ---
    num_tokens_to_evict = 40
    cache_lru.evict(EvictParams(num_tokens=num_tokens_to_evict))
    cache_peek.evict(EvictParams(num_tokens=num_tokens_to_evict))

    # Delta: re-prefill tokens CAUSED by the eviction decision.
    lru_cost = _reprefill_cost(hot_reqs, cache_lru) - lru_baseline
    peek_cost = _reprefill_cost(hot_reqs, cache_peek) - peek_baseline

    with capsys.disabled():
        print(
            f"\n  hot prefix: {len(hot_prefix)} tokens, {len(hot_reqs)} pending reqs depend on it"
        )
        print(f"  cold leaves: {len(cold_leaves)} x {len(cold_leaves[0])} tokens, 0 pending demand")
        print(f"  evicted {num_tokens_to_evict} tokens from each cache")
        print(f"  re-prefill tokens caused BY THE EVICTION (Δ from baseline):")
        print(f"    LRU:             {lru_cost:>6d}")
        print(f"    PeekDemand:      {peek_cost:>6d}")
        print(f"  peek saves {lru_cost - peek_cost} re-prefill tokens")

    # Peek should evict only cold nodes -> zero eviction-induced re-prefill.
    assert peek_cost == 0, (
        f"peek caused {peek_cost} extra re-prefill tokens -- should have evicted cold only"
    )
    # LRU should have evicted the aged hot prefix, forcing all pending reqs to re-prefill it.
    assert lru_cost >= len(hot_reqs) * len(hot_prefix), (
        f"LRU caused only {lru_cost} re-prefill tokens; expected it to evict the hot prefix"
    )

def test_peek_falls_back_to_lru_within_same_demand(capsys: pytest.CaptureFixture) -> None:
    """When two nodes have the same (zero) demand, peek should break ties by
    LRU -- evicting the older one first. Validates the tuple ordering."""
    cache = _make_cache()
    path_old = [100, 101, 102, 103]
    path_new = [200, 201, 202, 203]
    _insert(cache, path_old)
    _insert(cache, path_new)

    now = time.monotonic()
    def walk(n, when):
        n.last_access_time = when
        for child in n.children.values():
            walk(child, when)
    # age the "old" path
    for child in cache.root_node.children.values():
        if child.key and child.key.token_ids and child.key.token_ids[0] == 100:
            walk(child, now - 30)
        else:
            walk(child, now - 0.1)

    peek_tree = PendingTree()  # no pending -- everything is zero demand
    cache.eviction_strategy = PeekDemandStrategy(peek_tree)

    # Evict just enough to remove one path (4 tokens).
    cache.evict(EvictParams(num_tokens=4))

    # The OLD path should be gone; the NEW path should still match.
    assert _match_len(cache, path_old) < len(path_old), "old path should be evicted first (LRU tiebreak)"
    assert _match_len(cache, path_new) == len(path_new), "new path should remain (LRU protected it)"
