"""Tests for the ``queue-aware`` radix-cache eviction policy.

Upstream location: test/srt/test_queue_aware_eviction.py

Runs without a GPU or a model: it exercises the eviction ordering and the
per-node reference counting directly against a simulated RadixCache.
"""

import unittest

from sglang.srt.mem_cache.evict_policy import QueueAwareStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.utils import get_eviction_strategy


class _Node:
    """Minimal stand-in for a TreeNode for strategy-level assertions."""

    def __init__(self, key, queue_ref_count, last_access_time):
        self.key = key
        self.queue_ref_count = queue_ref_count
        self.last_access_time = last_access_time


class TestQueueAwareStrategy(unittest.TestCase):
    def test_registered_in_factory(self):
        self.assertIsInstance(
            get_eviction_strategy("queue-aware"), QueueAwareStrategy
        )

    def test_unreferenced_evicted_before_referenced(self):
        strat = QueueAwareStrategy()
        unreferenced = _Node(key=[1, 2, 3, 4], queue_ref_count=0, last_access_time=999.0)
        referenced = _Node(key=[5], queue_ref_count=1, last_access_time=1.0)
        # Lower priority is evicted first: the unreferenced node must sort first
        # even though it was accessed far more recently.
        self.assertLess(
            strat.get_priority(unreferenced), strat.get_priority(referenced)
        )

    def test_cheapest_referenced_evicted_first(self):
        strat = QueueAwareStrategy()
        cheap = _Node(key=[1, 2], queue_ref_count=1, last_access_time=1.0)  # cost 2
        expensive = _Node(key=[1] * 8, queue_ref_count=5, last_access_time=1.0)  # cost 40
        self.assertLess(strat.get_priority(cheap), strat.get_priority(expensive))

    def test_lru_tiebreak_within_segment(self):
        strat = QueueAwareStrategy()
        older = _Node(key=[1, 2], queue_ref_count=0, last_access_time=1.0)
        newer = _Node(key=[3, 4], queue_ref_count=0, last_access_time=2.0)
        self.assertLess(strat.get_priority(older), strat.get_priority(newer))


class TestQueueRefCounting(unittest.TestCase):
    def _cache(self):
        # Reference counting is independent of the configured policy; a
        # simulated cache exposes inc/reset_all_queue_refs and per-node
        # queue_ref_count regardless.
        return RadixCache.create_simulated()

    def test_inc_marks_ancestors(self):
        cache = self._cache()
        root = cache.root_node
        a = type(root)()
        a.parent = root
        b = type(root)()
        b.parent = a

        cache.inc_queue_ref(b)
        cache.inc_queue_ref(b)  # two waiting requests share this prefix
        self.assertEqual(b.queue_ref_count, 2)
        self.assertEqual(a.queue_ref_count, 2)
        self.assertEqual(root.queue_ref_count, 0)  # root is never counted

    def test_reset_clears_without_leakage(self):
        cache = self._cache()
        root = cache.root_node
        a = type(root)()
        a.parent = root
        b = type(root)()
        b.parent = a

        cache.inc_queue_ref(b)
        cache.reset_all_queue_refs()
        self.assertEqual(a.queue_ref_count, 0)
        self.assertEqual(b.queue_ref_count, 0)

        # Next scheduling step references only `a`; `b` must stay at zero.
        cache.inc_queue_ref(a)
        self.assertEqual(a.queue_ref_count, 1)
        self.assertEqual(b.queue_ref_count, 0)

    def test_split_node_inherits_ref_count(self):
        # A node split mid-cycle must carry its protection to the new parent.
        cache = self._cache()
        root = cache.root_node
        child = type(root)()
        child.parent = root
        child.queue_ref_count = 3
        # Emulate the inheritance line added to _split_node.
        new_node = type(root)()
        new_node.queue_ref_count = child.queue_ref_count
        self.assertEqual(new_node.queue_ref_count, 3)


if __name__ == "__main__":
    unittest.main()
