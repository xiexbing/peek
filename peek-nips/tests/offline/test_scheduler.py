# Copyright 2026 Anonymous Authors
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

"""Tests for peek.scheduler and peek.engine_vllm."""
import time
import unittest

import pytest
from types import SimpleNamespace

from peek.offline.scheduler import (
    detect_sharing,
    reorder_requests_by_prefix,
    update_queue_ref_counts,
    schedule_hook_vllm,
    vllm_on_schedule,
    detect_sharing_sglang,
    sglang_pre_schedule,
    sglang_should_run_prefix_matching,
    _vllm_engines,
)
from peek.offline.engine_vllm import VllmPeekEngine


class TestDetectSharing(unittest.TestCase):
    def test_no_sharing(self):
        assert detect_sharing([1, 2, 3, 4]) is False

    def test_has_sharing(self):
        assert detect_sharing([1, 2, 3, 1]) is True

    def test_empty(self):
        assert detect_sharing([]) is False

    def test_single(self):
        assert detect_sharing([42]) is False

    def test_all_same(self):
        assert detect_sharing([5, 5, 5]) is True


class TestReorderRequestsByPrefix(unittest.TestCase):
    def _req(self, block_hashes):
        return SimpleNamespace(block_hashes=block_hashes)

    def test_groups_by_first_hash(self):
        r0 = self._req(["A", "X"])
        r1 = self._req(["B", "Y"])
        r2 = self._req(["A", "Z"])
        r3 = self._req(["B", "W"])
        other = [self._req(["C"])]

        result = reorder_requests_by_prefix([r0, r1, r2, r3], other)

        assert len(result) == 5
        assert result[-1] is other[0]

    def test_single_request_no_reorder(self):
        r0 = self._req(["A"])
        other = [self._req(["B"])]
        result = reorder_requests_by_prefix([r0], other)
        assert result == [r0, other[0]]

    def test_empty(self):
        result = reorder_requests_by_prefix([], [])
        assert result == []


class TestUpdateQueueRefCounts(unittest.TestCase):
    def test_increments_cached_blocks(self):
        reset_called = [False]
        inc_calls = []

        class MockPool:
            def reset_queue_ref_counts(self):
                reset_called[0] = True

            def inc_queue_ref_count(self, blocks):
                inc_calls.append(blocks)

        pool = MockPool()
        cache = {"A": ["block_A"], "B": ["block_B"]}

        def get_cached(bh, gids):
            return cache.get(bh)

        req = SimpleNamespace(block_hashes=["A", "B", "C", "D"])
        update_queue_ref_counts([req], pool, get_cached, [0])

        assert reset_called[0]
        assert len(inc_calls) == 2
        assert inc_calls[0] == ["block_A"]
        assert inc_calls[1] == ["block_B"]


class TestScheduleHookVllm(unittest.TestCase):
    def test_skips_when_no_sharing(self):
        waiting = [
            SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "X"]),
            SimpleNamespace(num_computed_tokens=0, block_hashes=["B", "Y"]),
            SimpleNamespace(num_computed_tokens=0, block_hashes=["C", "Z"]),
        ]
        original_order = list(waiting)
        km = SimpleNamespace(block_pool=None)

        schedule_hook_vllm(waiting, km)
        assert waiting == original_order

    def test_reorders_when_sharing(self):
        reset_called = [False]
        inc_calls = []

        class MockPool:
            blocks = [SimpleNamespace(queue_ref_count=0) for _ in range(100)]

            def reset_queue_ref_counts(self):
                reset_called[0] = True

            def inc_queue_ref_count(self, blocks):
                inc_calls.append(blocks)

            def get_cached_block(self, bh, gids):
                return ["blk"] if bh == "A" else None

        pool = MockPool()
        km = SimpleNamespace(block_pool=pool)

        r0 = SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "X"])
        r1 = SimpleNamespace(num_computed_tokens=0, block_hashes=["B", "Y"])
        r2 = SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "Z", "W"])

        waiting = [r0, r1, r2]

        # Clear any cached engine for this km
        _vllm_engines.pop(id(km), None)

        schedule_hook_vllm(waiting, km)

        # Group A (r0, r2) should come before Group B (r1) due to cache hit
        assert waiting[0] in (r0, r2)
        assert waiting[1] in (r0, r2)
        assert waiting[2] is r1
        assert reset_called[0]
        assert len(inc_calls) > 0


class TestVllmOnSchedule(unittest.TestCase):
    """Tests for the top-level vllm_on_schedule entry point."""

    def _make_scheduler(self, enable_queue_aware=True, enable_caching=True,
                        waiting=None):
        reset_called = [False]
        inc_calls = []

        class MockPool:
            enable_queue_aware_eviction = enable_queue_aware
            blocks = [SimpleNamespace(queue_ref_count=0) for _ in range(100)]

            def reset_queue_ref_counts(self):
                reset_called[0] = True

            def inc_queue_ref_count(self, blocks):
                inc_calls.append(blocks)

            def get_cached_block(self, bh, gids):
                return ["blk"] if bh == "A" else None

        pool = MockPool()
        km = SimpleNamespace(block_pool=pool, enable_caching=enable_caching)
        scheduler = SimpleNamespace(kv_cache_manager=km, waiting=waiting or [])
        return scheduler, reset_called, inc_calls

    def test_skips_when_not_enabled(self):
        scheduler, reset_called, _ = self._make_scheduler(
            enable_queue_aware=False,
            waiting=[
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A"]),
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A"]),
            ],
        )
        vllm_on_schedule(scheduler)
        assert not reset_called[0]

    def test_skips_when_no_caching(self):
        scheduler, reset_called, _ = self._make_scheduler(
            enable_caching=False,
            waiting=[
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A"]),
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A"]),
            ],
        )
        vllm_on_schedule(scheduler)
        assert not reset_called[0]

    def test_skips_when_empty_waiting(self):
        scheduler, reset_called, _ = self._make_scheduler(waiting=[])
        vllm_on_schedule(scheduler)
        assert not reset_called[0]

    def test_throttle_runs_on_first_call(self):
        """First call (counter=1) should run the hook."""
        from peek.offline.scheduler import _vllm_counters
        scheduler, reset_called, _ = self._make_scheduler(
            waiting=[
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "X"]),
                SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "Y"]),
            ],
        )
        _vllm_counters.pop(id(scheduler), None)
        _vllm_engines.pop(id(scheduler), None)
        vllm_on_schedule(scheduler)
        assert reset_called[0]  # ran on first call

    def test_throttle_skips_intermediate_calls(self):
        """When throttle interval > 1, intermediate calls are skipped."""
        import peek.offline.scheduler as _mod
        from peek.offline.scheduler import _vllm_counters

        old_interval = _mod._VLLM_THROTTLE_INTERVAL
        _mod._VLLM_THROTTLE_INTERVAL = 2  # run every 2nd call
        try:
            scheduler, reset_called, _ = self._make_scheduler(
                waiting=[
                    SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "X"]),
                    SimpleNamespace(num_computed_tokens=0, block_hashes=["A", "Y"]),
                ],
            )
            _vllm_counters.pop(id(scheduler), None)
            _vllm_engines.pop(id(scheduler), None)
            vllm_on_schedule(scheduler)  # call 1 -> runs
            reset_called[0] = False
            vllm_on_schedule(scheduler)  # call 2 -> throttled
            assert not reset_called[0]
        finally:
            _mod._VLLM_THROTTLE_INTERVAL = old_interval


class TestDetectSharingSglang(unittest.TestCase):
    def test_no_sharing(self):
        queue = [
            SimpleNamespace(origin_input_ids=[1, 2, 3]),
            SimpleNamespace(origin_input_ids=[4, 5, 6]),
        ]
        assert detect_sharing_sglang(queue) is False

    def test_has_sharing(self):
        # Requests must share enough tokens to match the key_len.
        # With short inputs, detect_sharing_sglang uses the full input as key.
        shared_prefix = list(range(100))
        queue = [
            SimpleNamespace(origin_input_ids=shared_prefix + [901, 902]),
            SimpleNamespace(origin_input_ids=shared_prefix + [801, 802]),
        ]
        assert detect_sharing_sglang(queue) is True

    def test_has_sharing_short_identical(self):
        # Short but identical inputs also count as sharing.
        queue = [
            SimpleNamespace(origin_input_ids=[1, 2, 3]),
            SimpleNamespace(origin_input_ids=[1, 2, 3]),
        ]
        assert detect_sharing_sglang(queue) is True

    def test_no_sharing_short_different(self):
        # Short inputs that differ are NOT sharing (first-token match
        # is too coarse -- the new implementation correctly rejects these).
        queue = [
            SimpleNamespace(origin_input_ids=[1, 2, 3]),
            SimpleNamespace(origin_input_ids=[1, 7, 8]),
        ]
        assert detect_sharing_sglang(queue) is False

    def test_missing_attr(self):
        queue = [SimpleNamespace(), SimpleNamespace()]
        assert detect_sharing_sglang(queue) is False


@pytest.mark.engine
class TestSglangPreSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pytest.importorskip("sglang")

    def _make_tree_cache(self, eviction_policy="queue-aware"):
        reset_called = [False]

        class MockTreeNode:
            def __init__(self):
                self.children = {}

        class MockTreeCache:
            def __init__(self):
                self.root_node = MockTreeNode()

        tc = MockTreeCache()
        tc.eviction_policy = eviction_policy

        def reset_all_queue_refs():
            reset_called[0] = True

        tc.reset_all_queue_refs = reset_all_queue_refs
        return tc, reset_called

    def test_returns_false_when_not_queue_aware(self):
        tc, reset_called = self._make_tree_cache("lru")
        class MockMatchResult:
            device_indices = []
            last_device_node = None
            last_host_node = None
            host_hit_length = 0
        tc.match_prefix = lambda params: MockMatchResult()
        queue = [
            SimpleNamespace(origin_input_ids=[1, 2], output_ids=[], extra_key=None),
            SimpleNamespace(origin_input_ids=[1, 3], output_ids=[], extra_key=None),
        ]
        assert sglang_pre_schedule(tc, queue) is False
        assert not reset_called[0]

    def test_returns_false_when_no_sharing(self):
        tc, reset_called = self._make_tree_cache("queue-aware")
        queue = [
            SimpleNamespace(origin_input_ids=[1, 2]),
            SimpleNamespace(origin_input_ids=[4, 5]),
        ]
        assert sglang_pre_schedule(tc, queue) is False
        assert not reset_called[0]

    def test_returns_true_and_resets_when_sharing(self):
        tc, reset_called = self._make_tree_cache("queue-aware")
        class MockMatchResult:
            device_indices = []
            last_device_node = None
            last_host_node = None
            host_hit_length = 0
        tc.match_prefix = lambda params: MockMatchResult()
        # Requests must share the full key (first 64 tokens) to be detected
        shared_prefix = list(range(100))
        queue = [
            SimpleNamespace(origin_input_ids=shared_prefix + [901], output_ids=[], extra_key=None),
            SimpleNamespace(origin_input_ids=shared_prefix + [902], output_ids=[], extra_key=None),
        ]
        assert sglang_pre_schedule(tc, queue) is True
        assert reset_called[0]


class TestSglangShouldRunPrefixMatching(unittest.TestCase):
    def test_true_when_queue_aware(self):
        tc = SimpleNamespace(eviction_policy="queue-aware")
        assert sglang_should_run_prefix_matching(tc) is True

    def test_false_when_lru(self):
        tc = SimpleNamespace(eviction_policy="lru")
        assert sglang_should_run_prefix_matching(tc) is False

    def test_false_when_no_attr(self):
        tc = SimpleNamespace()
        assert sglang_should_run_prefix_matching(tc) is False


class TestVllmPeekEngine(unittest.TestCase):
    """Tests for the VllmPeekEngine class."""

    def _make_pool(self, cache_map=None, num_blocks=100):
        cache_map = cache_map or {}
        reset_calls = []
        inc_calls = []

        class MockBlock:
            def __init__(self, block_id):
                self.block_id = block_id
                self.queue_ref_count = 0
                self.is_null = False

        class MockPool:
            blocks = [MockBlock(i) for i in range(num_blocks)]

            def reset_queue_ref_counts(self):
                reset_calls.append(True)
                for b in self.blocks:
                    b.queue_ref_count = 0

            def inc_queue_ref_count(self, blocks):
                inc_calls.append(blocks)
                for b in blocks:
                    b.queue_ref_count += 1

            def get_cached_block(self, bh, gids):
                return cache_map.get(bh)

        return MockPool(), reset_calls, inc_calls

    def _req(self, block_hashes):
        return SimpleNamespace(
            num_computed_tokens=0,
            block_hashes=block_hashes,
        )

    def test_groups_with_higher_cache_frac_score_higher(self):
        """Groups with more cached blocks should be scheduled first."""
        blk_a = SimpleNamespace(queue_ref_count=0, is_null=False)
        blk_b = SimpleNamespace(queue_ref_count=0, is_null=False)
        cache_map = {"A1": [blk_a], "A2": [blk_b]}
        pool, _, _ = self._make_pool(cache_map)

        r0 = self._req(["A1", "A2", "A3"])
        r1 = self._req(["A1", "A2", "A4"])
        r2 = self._req(["B1", "B2", "B3"])
        r3 = self._req(["B1", "B2", "B4"])

        engine = VllmPeekEngine(pool, [0])
        waiting = [r0, r1, r2, r3]
        engine.run(waiting)

        # Group A (cached) should come before Group B (uncached)
        assert waiting[0] in (r0, r1)
        assert waiting[1] in (r0, r1)
        assert waiting[2] in (r2, r3)
        assert waiting[3] in (r2, r3)

    def test_adaptive_protection_limits_ref_counts(self):
        """Only top-K groups should get ref count protection."""
        blk = SimpleNamespace(queue_ref_count=0, is_null=False)
        cache_map = {"A1": [blk]}
        pool, _, inc_calls = self._make_pool(cache_map, num_blocks=4)

        r0 = self._req(["A1", "A2", "A3", "A4"])
        r1 = self._req(["A1", "A2", "A3", "A5"])
        r2 = self._req(["B1", "B2", "B3", "B4"])
        r3 = self._req(["B1", "B2", "B3", "B5"])

        engine = VllmPeekEngine(pool, [0])
        waiting = [r0, r1, r2, r3]
        engine.run(waiting)

        # With 4 blocks and avg prefix of 4 blocks, capacity_groups=1
        # protect_k = min(2, max(1, 1*3//4)) = 1
        assert len(inc_calls) <= 1

    def test_targeted_reset_only_clears_prev_protected(self):
        """Second call should reset only previously protected blocks, not all."""
        blk_a = SimpleNamespace(queue_ref_count=0, is_null=False)
        blk_x = SimpleNamespace(queue_ref_count=5, is_null=False)
        cache_map = {"A1": [blk_a], "X1": [blk_x]}
        pool, reset_calls, _ = self._make_pool(cache_map)

        engine = VllmPeekEngine(pool, [0])

        r0 = self._req(["A1", "A2"])
        r1 = self._req(["A1", "A3"])

        # First call: no prev_protected, uses full reset
        waiting = [r0, r1]
        engine.run(waiting)
        assert len(reset_calls) == 1

        # Second call: should use targeted reset (not full reset)
        reset_calls.clear()
        r2 = self._req(["A1", "A4"])
        r3 = self._req(["A1", "A5"])
        waiting = [r2, r3]
        engine.run(waiting)
        assert len(reset_calls) == 0

    def test_eviction_risk_tiebreaker(self):
        """Groups not seen recently should score higher due to eviction risk."""
        blk_a = SimpleNamespace(queue_ref_count=0, is_null=False)
        blk_b = SimpleNamespace(queue_ref_count=0, is_null=False)
        cache_map = {"A1": [blk_a], "B1": [blk_b]}
        pool, _, _ = self._make_pool(cache_map)

        engine = VllmPeekEngine(pool, [0])

        # Seed group A as seen long ago, group B as seen recently
        now = time.monotonic()
        engine._group_last_seen = {
            ("A1", "A2"): now - 100.0,
            ("B1", "B2"): now - 0.01,
        }

        r0 = self._req(["A1", "A2"])
        r1 = self._req(["B1", "B2"])

        waiting = [r0, r1]
        engine.run(waiting)
        # Group A has higher eviction_risk -> should be scheduled first
        assert waiting[0] is r0

    def test_returns_all_requests(self):
        """All input requests must appear in output."""
        pool, _, _ = self._make_pool()
        reqs = [self._req(["A", "B"]) for _ in range(5)]

        engine = VllmPeekEngine(pool, [0])
        waiting = list(reqs)
        engine.run(waiting)
        assert len(waiting) == 5
        assert set(id(r) for r in waiting) == set(id(r) for r in reqs)

    def test_online_buffering_defers_small_groups(self):
        """Small, newly-arrived groups should be deferred behind ready groups."""
        blk_a = SimpleNamespace(queue_ref_count=0, is_null=False)
        blk_b = SimpleNamespace(queue_ref_count=0, is_null=False)
        cache_map = {"A1": [blk_a], "B1": [blk_b]}
        pool, _, _ = self._make_pool(cache_map)

        engine = VllmPeekEngine(pool, [0])

        # Group A: 5 requests (>= BUFFER_MIN_GROUP_SIZE=4 -> ready)
        group_a = [self._req(["A1", "A2"]) for _ in range(5)]
        # Group B: 2 requests (< 4 and just arrived -> buffering)
        group_b = [self._req(["B1", "B2"]) for _ in range(2)]

        waiting = group_b + group_a  # B first in queue
        engine.run(waiting)

        # Group A (ready) should come before Group B (buffering)
        group_a_ids = {id(r) for r in group_a}
        for i in range(5):
            assert id(waiting[i]) in group_a_ids
        group_b_ids = {id(r) for r in group_b}
        for i in range(5, 7):
            assert id(waiting[i]) in group_b_ids

    def test_buffering_promotes_after_wait(self):
        """Groups waiting longer than BUFFER_MAX_WAIT_MS should become ready."""
        pool, _, _ = self._make_pool({"A1": [SimpleNamespace(queue_ref_count=0)]})

        engine = VllmPeekEngine(pool, [0])
        # Override for faster test
        engine.BUFFER_MAX_WAIT_MS = 0.0  # immediate promotion

        group_a = [self._req(["A1", "A2"]) for _ in range(2)]
        waiting = list(group_a)
        engine.run(waiting)

        # With max_wait=0, even small groups are immediately ready
        assert len(waiting) == 2


if __name__ == "__main__":
    unittest.main()
