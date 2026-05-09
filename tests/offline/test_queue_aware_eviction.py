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

"""Tests for the queue-aware eviction policy.

Split into two groups:
  1. Unit tests for QueueAwareStrategy -- test priority logic directly on TreeNode
     objects without needing a full RadixCache (avoids CUDA import chain).
  2. Integration tests that import RadixCache -- skipped if sgl_kernel is
     unavailable (no GPU environment).
"""

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import heapq
import sys
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. Unit tests: QueueAwareStrategy priority logic
#    These only need evict_policy.py and a minimal TreeNode, no CUDA.
# ---------------------------------------------------------------------------

from sglang.srt.mem_cache.evict_policy import (
    QueueAwareStrategy,
    LRUStrategy,
)

class _MockNode:
    """Minimal TreeNode stand-in for priority tests."""

    def __init__(
        self,
        key_len: int = 1,
        queue_ref_count: int = 0,
        last_access_time: float = 0.0,
    ):
        self.key = [0] * key_len  # list with __len__
        self.queue_ref_count = queue_ref_count
        self.last_access_time = last_access_time

class TestQueueAwareStrategyPriority:
    def test_unreferenced_lower_priority_than_referenced(self):
        strategy = QueueAwareStrategy()

        node_unref = _MockNode(key_len=3, queue_ref_count=0, last_access_time=100.0)
        node_ref = _MockNode(key_len=3, queue_ref_count=3, last_access_time=50.0)

        prio_unref = strategy.get_priority(node_unref)
        prio_ref = strategy.get_priority(node_ref)

        # (0, ...) < (1, ...) -- unreferenced evicted first
        assert prio_unref < prio_ref

    def test_cost_weighted_among_referenced(self):
        strategy = QueueAwareStrategy()

        # Low cost: 1 ref * 2 tokens = 2
        node_low = _MockNode(key_len=2, queue_ref_count=1, last_access_time=100.0)
        # High cost: 5 refs * 4 tokens = 20
        node_high = _MockNode(key_len=4, queue_ref_count=5, last_access_time=100.0)

        prio_low = strategy.get_priority(node_low)
        prio_high = strategy.get_priority(node_high)

        assert prio_low < prio_high

    def test_lru_tiebreaker_among_unreferenced(self):
        strategy = QueueAwareStrategy()

        node_old = _MockNode(key_len=1, queue_ref_count=0, last_access_time=50.0)
        node_new = _MockNode(key_len=1, queue_ref_count=0, last_access_time=100.0)

        prio_old = strategy.get_priority(node_old)
        prio_new = strategy.get_priority(node_new)

        assert prio_old < prio_new

    def test_lru_tiebreaker_among_same_cost_referenced(self):
        strategy = QueueAwareStrategy()

        node_old = _MockNode(key_len=4, queue_ref_count=2, last_access_time=10.0)
        node_new = _MockNode(key_len=4, queue_ref_count=2, last_access_time=100.0)

        prio_old = strategy.get_priority(node_old)
        prio_new = strategy.get_priority(node_new)

        # Same tier and cost, older access -> lower priority
        assert prio_old < prio_new

    def test_zero_ref_with_none_key(self):
        """Node with key=None should not crash."""
        strategy = QueueAwareStrategy()
        node = _MockNode(key_len=0, queue_ref_count=0, last_access_time=0.0)
        node.key = None
        prio = strategy.get_priority(node)
        assert prio == (0, 0, 0.0)

    def test_heap_ordering_matches_expected_eviction_order(self):
        """Simulate a min-heap eviction: verify the pop order matches
        unreferenced-LRU-first, then referenced-low-cost-first."""
        strategy = QueueAwareStrategy()

        nodes = [
            _MockNode(key_len=4, queue_ref_count=0, last_access_time=10.0),   # A: unref, old
            _MockNode(key_len=4, queue_ref_count=0, last_access_time=50.0),   # B: unref, newer
            _MockNode(key_len=2, queue_ref_count=1, last_access_time=100.0),  # C: ref, cost=2
            _MockNode(key_len=8, queue_ref_count=3, last_access_time=5.0),    # D: ref, cost=24
        ]

        heap = [(strategy.get_priority(n), i) for i, n in enumerate(nodes)]
        heapq.heapify(heap)

        eviction_order = []
        while heap:
            _, idx = heapq.heappop(heap)
            eviction_order.append(idx)

        # Expected: A(unref,old) -> B(unref,new) -> C(ref,low cost) -> D(ref,high cost)
        assert eviction_order == [0, 1, 2, 3]

# ---------------------------------------------------------------------------
# 2. Integration tests: RadixCache + queue-aware eviction
#    Skipped if sgl_kernel cannot be loaded (no GPU / missing .so).
# ---------------------------------------------------------------------------

_radix_import_error = None
try:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    import torch

    _HAS_RADIX = True
except Exception as e:
    _HAS_RADIX = False
    _radix_import_error = str(e)

requires_radix = pytest.mark.skipif(
    not _HAS_RADIX,
    reason=f"Cannot import RadixCache (likely no GPU): {_radix_import_error}",
)

class _NullAllocator:
    """Stub allocator for unit tests that exercise eviction without a GPU.

    sglang's RadixCache reads ``self.token_to_kv_pool_allocator.device``
    in ``__init__`` and calls ``allocator.free(indices)`` from ``evict``.
    These unit tests don't actually allocate KV memory, so a no-op free
    + a CPU device handle is sufficient.
    """

    device = torch.device("cpu")

    def free(self, indices):  # noqa: ARG002 -- sglang interface
        return None


def _make_cache(eviction_policy="queue-aware", page_size=1, allocator=None):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator if allocator is not None else _NullAllocator(),
        page_size=page_size,
        eviction_policy=eviction_policy,
    )
    return RadixCache(params)

def _insert(cache, token_ids):
    key = RadixKey(token_ids=token_ids, extra_key=None)
    value = torch.arange(len(token_ids), dtype=torch.int64)
    cache.insert(InsertParams(key=key, value=value))

def _match(cache, token_ids):
    key = RadixKey(token_ids=token_ids, extra_key=None)
    result = cache.match_prefix(MatchPrefixParams(key=key))
    return len(result.device_indices), result.last_device_node

@requires_radix
class TestQueueRefCounting:
    def test_inc_propagates_to_ancestors(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3, 4, 5])
        _, last_node = _match(cache, [1, 2, 3, 4, 5])

        cache.inc_queue_ref(last_node)

        node = last_node
        while node != cache.root_node:
            assert node.queue_ref_count >= 1
            node = node.parent
        assert cache.root_node.queue_ref_count == 0

    def test_dec_queue_ref(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        _, last_node = _match(cache, [1, 2, 3])

        cache.inc_queue_ref(last_node)
        cache.dec_queue_ref(last_node)

        node = last_node
        while node != cache.root_node:
            assert node.queue_ref_count == 0
            node = node.parent

    def test_multiple_refs_accumulate(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3, 4])
        _insert(cache, [1, 2, 5, 6])

        _, node_a = _match(cache, [1, 2, 3, 4])
        _, node_b = _match(cache, [1, 2, 5, 6])

        cache.inc_queue_ref(node_a)
        cache.inc_queue_ref(node_b)

        _, shared_node = _match(cache, [1, 2])
        assert shared_node.queue_ref_count >= 2

    def test_reset_all_queue_refs(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        _insert(cache, [4, 5, 6])

        _, n1 = _match(cache, [1, 2, 3])
        _, n2 = _match(cache, [4, 5, 6])
        cache.inc_queue_ref(n1)
        cache.inc_queue_ref(n2)

        cache.reset_all_queue_refs()

        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            assert node.queue_ref_count == 0
            for child in node.children.values():
                stack.append(child)

    def test_dec_never_goes_negative(self):
        cache = _make_cache()
        _insert(cache, [1, 2])
        _, node = _match(cache, [1, 2])

        cache.dec_queue_ref(node)
        while node != cache.root_node:
            assert node.queue_ref_count == 0
            node = node.parent

@requires_radix
class TestQueueAwareEvictionIntegration:
    def test_unreferenced_evicted_before_referenced(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        _insert(cache, [4, 5, 6])

        _, node_ref = _match(cache, [1, 2, 3])
        cache.inc_queue_ref(node_ref)

        result = cache.evict(EvictParams(num_tokens=3))
        assert result.num_tokens_evicted >= 3

        matched, _ = _match(cache, [1, 2, 3])
        assert matched == 3, "Referenced sequence should survive"

        matched, _ = _match(cache, [4, 5, 6])
        assert matched == 0, "Unreferenced sequence should be evicted"

    def test_forced_eviction_of_referenced_by_cost(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        _insert(cache, [4, 5, 6, 7, 8, 9, 10])

        _, node_small = _match(cache, [1, 2, 3])
        _, node_large = _match(cache, [4, 5, 6, 7, 8, 9, 10])

        cache.inc_queue_ref(node_small)   # cost = 1 * 3 = 3
        for _ in range(5):
            cache.inc_queue_ref(node_large)  # cost = 5 * 7 = 35

        result = cache.evict(EvictParams(num_tokens=3))
        assert result.num_tokens_evicted >= 3

        matched, _ = _match(cache, [4, 5, 6, 7, 8, 9, 10])
        assert matched == 7, "High-cost referenced sequence should survive"

    def test_lru_fallback_for_unreferenced(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        time.sleep(0.01)
        _insert(cache, [4, 5, 6])
        time.sleep(0.01)
        _insert(cache, [7, 8, 9])

        result = cache.evict(EvictParams(num_tokens=3))
        assert result.num_tokens_evicted >= 3

        matched, _ = _match(cache, [1, 2, 3])
        assert matched == 0, "Oldest unreferenced should be evicted first"

        matched, _ = _match(cache, [7, 8, 9])
        assert matched == 3

    def test_cache_accepts_queue_aware_policy(self):
        cache = _make_cache("queue-aware")
        assert isinstance(cache.eviction_strategy, QueueAwareStrategy)

# ---------------------------------------------------------------------------
# CLI registration (lightweight import)
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    def test_queue_aware_in_choices(self):
        from sglang.srt.server_args import RADIX_EVICTION_POLICY_CHOICES
        assert "queue-aware" in RADIX_EVICTION_POLICY_CHOICES
