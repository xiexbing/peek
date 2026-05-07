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

import pytest
from peek.offline.reorder import reorder_for_prefix_sharing


class TestReorder:
    def test_basic_reordering(self):
        sequences = [
            [10, 20, 30],  # idx 0 — shares prefix with idx 2
            [50, 60],      # idx 1 — unique
            [10, 20, 40],  # idx 2 — shares prefix with idx 0
        ]
        # These short prefixes are below default min_sharing_depth=32,
        # so force reorder with min_coverage=0.0, min_avg_sharing_depth=0
        # to test DFS logic.
        order = reorder_for_prefix_sharing(
            sequences, min_coverage=0.0, min_avg_sharing_depth=0,
        )

        # All indices present
        assert set(order) == {0, 1, 2}

        # 0 and 2 share prefix [10, 20] and must be adjacent
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[2]) == 1

    def test_all_unique(self):
        sequences = [[1], [2], [3]]
        order = reorder_for_prefix_sharing(sequences)
        assert set(order) == {0, 1, 2}

    def test_all_identical(self):
        sequences = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        order = reorder_for_prefix_sharing(sequences)
        assert order == [0, 1, 2]

    def test_empty_input(self):
        assert reorder_for_prefix_sharing([]) == []

    def test_single_prompt(self):
        assert reorder_for_prefix_sharing([[1, 2]]) == [0]

    def test_multiple_groups(self):
        sequences = [
            [1, 2, 3],  # group A
            [4, 5, 6],  # group B
            [1, 2, 7],  # group A
            [4, 5, 8],  # group B
        ]
        # Short prefixes — force reorder to test DFS logic.
        order = reorder_for_prefix_sharing(
            sequences, min_coverage=0.0, min_avg_sharing_depth=0,
        )
        pos = {idx: rank for rank, idx in enumerate(order)}

        # Group A (0, 2) adjacent
        assert abs(pos[0] - pos[2]) == 1
        # Group B (1, 3) adjacent
        assert abs(pos[1] - pos[3]) == 1

    def test_preserves_count(self):
        sequences = [[i, i + 1] for i in range(20)]
        order = reorder_for_prefix_sharing(sequences)
        assert len(order) == 20
        assert set(order) == set(range(20))

    def test_skips_reorder_when_no_sharing(self):
        """Unique short sequences should return identity order (no reorder)."""
        sequences = [[i, i + 1, i + 2] for i in range(10)]
        order = reorder_for_prefix_sharing(sequences)
        assert order == list(range(10))

    def test_reorders_when_strong_sharing(self):
        """Prompts with deep shared prefixes should be reordered."""
        shared = list(range(50))
        sequences = [
            shared + [1000],
            [900, 901, 902],  # unique — will be separated
            shared + [1001],
        ]
        # 50-token sharing is below default min_avg_sharing_depth=64,
        # so lower it to test DFS grouping.
        order = reorder_for_prefix_sharing(
            sequences, min_avg_sharing_depth=0,
        )
        assert set(order) == {0, 1, 2}
        # 0 and 2 share 50 tokens — must be adjacent after reorder
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[2]) == 1

    def test_preserves_order_for_shallow_sharing(self):
        """Prompts sharing only a few tokens should not be reordered."""
        sequences = [
            [1, 2, 100 + i] for i in range(5)
        ]
        order = reorder_for_prefix_sharing(sequences)
        # Sharing depth is 2, well below default 32 — identity expected
        assert order == list(range(5))

    def test_custom_thresholds(self):
        """Custom min_sharing_depth and min_coverage override defaults."""
        # Build sequences sharing 5 tokens
        shared = [10, 20, 30, 40, 50]
        sequences = [
            shared + [100],
            [999, 998],
            shared + [200],
        ]
        # Default min_sharing_depth=32 would skip — coverage 0
        order_default = reorder_for_prefix_sharing(sequences)
        assert order_default == [0, 1, 2]

        # With min_sharing_depth=5, coverage is 2/3 > 0.1 — should reorder
        order_custom = reorder_for_prefix_sharing(
            sequences, min_sharing_depth=5, min_coverage=0.1,
            min_avg_sharing_depth=0,
        )
        pos = {idx: rank for rank, idx in enumerate(order_custom)}
        assert abs(pos[0] - pos[2]) == 1

    def test_skips_reorder_for_shallow_avg_sharing_depth(self):
        """Workloads like code_completion with ~40-token shared prefix
        should skip reorder (avg_sharing_depth < 64)."""
        shared = list(range(40))
        sequences = [
            shared + [1000 + i] for i in range(5)
        ] + [[9000 + i] for i in range(5)]
        order = reorder_for_prefix_sharing(
            sequences, min_sharing_depth=10,
        )
        assert order == list(range(10))

    def test_reorders_for_deep_avg_sharing_depth(self):
        """Workloads with 128+ token shared prefix should reorder."""
        shared = list(range(200))
        sequences = [
            shared + [1000],
            [9000, 9001, 9002],
            shared + [1001],
        ]
        order = reorder_for_prefix_sharing(sequences)
        assert set(order) == {0, 1, 2}
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[2]) == 1

    def test_skips_reorder_for_high_duplication(self):
        """Exact duplicate prompts should return identity — DFS grouping
        hurts because concurrent identical requests can't reuse each other's
        KV cache."""
        prompt_a = list(range(50))
        prompt_b = list(range(100, 150))
        # 4 copies of A, 4 copies of B — duplication_ratio = 1.0
        sequences = [prompt_a] * 4 + [prompt_b] * 4
        order = reorder_for_prefix_sharing(sequences)
        assert order == list(range(8))

    def test_reorders_shared_prefix_not_duplicates(self):
        """Prompts sharing a deep prefix but with unique suffixes should
        still be reordered — they're not duplicates."""
        shared = list(range(50))
        sequences = [shared + [1000 + i] for i in range(5)]
        order = reorder_for_prefix_sharing(sequences)
        # All share 50 tokens, unique suffixes, 0% duplication → reorder
        assert set(order) == set(range(5))
        assert len(order) == 5

    def test_long_shared_prefix_different_suffix_not_duplicates(self):
        """Simulates shared_system_prompts: 200-token shared prefix with
        unique 20-token suffixes.  Even though the trie (max_depth=128)
        truncates them to look identical, full sequences differ — should
        still reorder."""
        shared = list(range(200))
        sequences = [shared + [5000 + i] for i in range(10)]
        order = reorder_for_prefix_sharing(sequences)
        # 0% full-sequence duplication, deep sharing → reorder
        assert set(order) == set(range(10))

    def test_max_duplication_threshold(self):
        """Custom max_duplication threshold controls the skip."""
        prompt = list(range(50))
        # All duplicates
        sequences = [prompt] * 6
        # Default max_duplication=0.5 → skip (duplication=1.0)
        assert reorder_for_prefix_sharing(sequences) == list(range(6))
        # With max_duplication=1.0 → never skip on duplication
        order = reorder_for_prefix_sharing(
            sequences, min_coverage=0.0, max_duplication=1.0,
        )
        assert set(order) == set(range(6))
