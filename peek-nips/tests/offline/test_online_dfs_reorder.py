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

"""Tests for Peek's online DFS trie reordering under Poisson arrivals.

Verifies that:
1. reorder_for_prefix_sharing correctly groups interleaved Poisson arrivals
   by shared prefix (DFS adjacency).
2. PeekDispatcher reorders buffered requests via trie DFS before dispatch.
3. Under realistic Poisson-interleaved workloads (Zipf-distributed groups),
   reordering produces prefix-adjacent output that maximises cache sharing.
4. Reordering is stable: same input always produces the same output.
"""
import random
import unittest

from peek.offline.reorder import reorder_for_prefix_sharing, PeekConfig, PeekDispatcher
from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poisson_interleaved(
    num_groups: int,
    prefix_len: int,
    n: int,
    seed: int = 42,
    suffix_len: int = 20,
    zipf_alpha: float = 1.0,
) -> tuple[list[list[int]], list[int]]:
    """Simulate Poisson-interleaved arrivals with shared prefixes.

    Returns (sequences, group_labels) where sequences are interleaved
    across groups as they would arrive under a Poisson process with
    Zipf-distributed group popularity.
    """
    rng = random.Random(seed)

    # Deterministic per-group prefixes
    group_prefixes = []
    for g in range(num_groups):
        grng = random.Random(g * 1000 + 7)
        group_prefixes.append([grng.randint(1, 31999) for _ in range(prefix_len)])

    # Zipf group assignment
    weights = [1.0 / (k + 1) ** zipf_alpha for k in range(num_groups)]
    total = sum(weights)
    cdf = []
    cumulative = 0.0
    for w in weights:
        cumulative += w / total
        cdf.append(cumulative)

    sequences = []
    labels = []
    for _ in range(n):
        u = rng.random()
        g = 0
        for gi, c in enumerate(cdf):
            if u <= c:
                g = gi
                break
        else:
            g = num_groups - 1
        suffix = [rng.randint(1, 31999) for _ in range(suffix_len)]
        sequences.append(group_prefixes[g] + suffix)
        labels.append(g)

    return sequences, labels


def _adjacency_score(order: list[int], labels: list[int]) -> float:
    """Fraction of consecutive pairs in *order* that share the same group.

    1.0 = perfectly grouped, 0.0 = no adjacent same-group pairs.
    """
    if len(order) <= 1:
        return 1.0
    same = sum(
        1 for i in range(len(order) - 1)
        if labels[order[i]] == labels[order[i + 1]]
    )
    return same / (len(order) - 1)


# ---------------------------------------------------------------------------
# Tests: reorder_for_prefix_sharing under Poisson arrivals
# ---------------------------------------------------------------------------

class TestDfsReorderPoissonArrivals(unittest.TestCase):
    """reorder_for_prefix_sharing must group interleaved Poisson requests."""

    def test_interleaved_arrivals_get_grouped(self):
        """Requests from different groups arriving interleaved (Poisson)
        must be reordered so same-group requests are adjacent."""
        sequences, labels = _make_poisson_interleaved(
            num_groups=5, prefix_len=200, n=50, seed=42,
        )
        order = reorder_for_prefix_sharing(sequences)

        # Verify all indices present
        self.assertEqual(set(order), set(range(50)))

        # Adjacency should be much better than random interleaving
        reordered_adj = _adjacency_score(order, labels)
        identity_adj = _adjacency_score(list(range(50)), labels)
        self.assertGreater(reordered_adj, identity_adj,
                           "DFS reorder must improve prefix adjacency over "
                           "Poisson-interleaved order")
        # With 5 groups and 50 requests, DFS should achieve near-perfect grouping
        self.assertGreater(reordered_adj, 0.85,
                           f"Expected high adjacency, got {reordered_adj:.2f}")

    def test_zipf_skewed_arrivals(self):
        """Under Zipf skew (alpha=1.5), the popular groups should still
        cluster together after reorder."""
        sequences, labels = _make_poisson_interleaved(
            num_groups=20, prefix_len=200, n=200, seed=99, zipf_alpha=1.5,
        )
        order = reorder_for_prefix_sharing(sequences)

        self.assertEqual(set(order), set(range(200)))
        adj = _adjacency_score(order, labels)
        self.assertGreater(adj, 0.80,
                           f"Zipf workload adjacency too low: {adj:.2f}")

    def test_many_groups_deep_prefix(self):
        """100 groups × 4096-token prefix (mirrors run_online_peek_vs_lpm.sh)."""
        # Use shorter prefix for test speed, but > max_depth(128) to test truncation
        sequences, labels = _make_poisson_interleaved(
            num_groups=100, prefix_len=256, n=500, seed=7, zipf_alpha=1.5,
        )
        order = reorder_for_prefix_sharing(sequences)

        self.assertEqual(len(order), 500)
        self.assertEqual(set(order), set(range(500)))

        adj = _adjacency_score(order, labels)
        self.assertGreater(adj, 0.75,
                           f"100-group adjacency too low: {adj:.2f}")

    def test_deterministic_across_calls(self):
        """Same input must produce same reorder output."""
        sequences, _ = _make_poisson_interleaved(
            num_groups=10, prefix_len=200, n=100, seed=42,
        )
        order1 = reorder_for_prefix_sharing(sequences)
        order2 = reorder_for_prefix_sharing(sequences)
        self.assertEqual(order1, order2)

    def test_single_group_no_disruption(self):
        """If all requests share the same prefix, reorder should keep them all."""
        prefix = list(range(200))
        sequences = [prefix + [1000 + i] for i in range(30)]
        labels = [0] * 30
        order = reorder_for_prefix_sharing(sequences)
        self.assertEqual(set(order), set(range(30)))
        # All same group → adjacency must be 1.0
        self.assertAlmostEqual(_adjacency_score(order, labels), 1.0)

    def test_no_sharing_identity(self):
        """Completely unique prefixes should return identity (no reorder)."""
        rng = random.Random(123)
        sequences = [
            [rng.randint(1, 31999) for _ in range(200)]
            for _ in range(20)
        ]
        order = reorder_for_prefix_sharing(sequences)
        self.assertEqual(order, list(range(20)))


# ---------------------------------------------------------------------------
# Tests: PrefixTrie DFS correctness with Poisson-style input
# ---------------------------------------------------------------------------

class TestTrieDfsPoissonInput(unittest.TestCase):
    """PrefixTrie.dfs_order must produce prefix-adjacent output."""

    def test_dfs_groups_interleaved_arrivals(self):
        """Insert interleaved arrivals into trie, verify DFS groups them."""
        sequences, labels = _make_poisson_interleaved(
            num_groups=5, prefix_len=100, n=40, seed=77,
        )
        trie = PrefixTrie()
        for idx, seq in enumerate(sequences):
            trie.insert(seq, idx)

        order = trie.dfs_order()
        self.assertEqual(set(order), set(range(40)))

        adj = _adjacency_score(order, labels)
        self.assertGreater(adj, 0.85)

    def test_count_aware_dfs_prefers_large_groups(self):
        """count_aware=True should place the largest group first."""
        prefix_a = list(range(100))
        prefix_b = list(range(100, 200))
        # Group A: 10 requests, Group B: 3 requests
        sequences = (
            [prefix_a + [5000 + i] for i in range(10)]
            + [prefix_b + [6000 + i] for i in range(3)]
        )
        # Interleave them
        rng = random.Random(42)
        indices = list(range(13))
        rng.shuffle(indices)
        shuffled = [sequences[i] for i in indices]

        trie = PrefixTrie()
        for idx, seq in enumerate(shuffled):
            trie.insert(seq, idx)

        order = trie.dfs_order(count_aware=True)
        self.assertEqual(set(order), set(range(13)))

        # First 10 entries should be from group A (the larger group)
        group_a_indices = {i for i, orig in enumerate(indices)
                          if orig < 10}
        first_10 = set(order[:10])
        self.assertEqual(first_10, group_a_indices,
                         "count_aware DFS should place largest group first")

    def test_sharing_score_detects_poisson_sharing(self):
        """sharing_score should detect prefix sharing in interleaved input."""
        sequences, _ = _make_poisson_interleaved(
            num_groups=5, prefix_len=200, n=50, seed=42,
        )
        trie = PrefixTrie()
        for idx, seq in enumerate(sequences):
            trie.insert(seq, idx)

        coverage, max_group, avg_depth = trie.sharing_score(min_depth=32)
        self.assertGreater(coverage, 0.5,
                           "Most requests share prefixes — coverage should be high")
        self.assertGreater(avg_depth, 100,
                           "Avg sharing depth should reflect the 200-token prefix")
        self.assertGreater(max_group, 1)


# ---------------------------------------------------------------------------
# Tests: PeekDispatcher with Poisson arrivals
# ---------------------------------------------------------------------------

class TestPeekDispatcherPoisson(unittest.TestCase):
    """PeekDispatcher tags requests via incremental trie on each submit()."""

    def test_dispatcher_tags_all_requests(self):
        """All submitted requests must be dispatched with peek tags."""
        sequences, labels = _make_poisson_interleaved(
            num_groups=5, prefix_len=200, n=40, seed=42,
        )
        requests = [
            {"id": f"req-{i}", "token_ids": seq, "group": labels[i]}
            for i, seq in enumerate(sequences)
        ]

        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        for req in requests:
            dispatcher.submit(req)

        self.assertEqual(len(dispatched), 40,
                         f"Expected 40 dispatched, got {len(dispatched)}")

        # All rids should carry peek tags
        for r in dispatched:
            self.assertTrue(r["rid"].startswith("peek:"),
                            f"Missing peek tag: {r['rid']}")

    def test_dispatcher_dispatches_all_requests(self):
        """All submitted requests must be dispatched."""
        rng = random.Random(99)
        prefix = list(range(200))
        requests = [
            {"id": f"req-{i}", "token_ids": prefix + [rng.randint(1, 31999)]}
            for i in range(20)
        ]

        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        for req in requests:
            dispatcher.submit(req)

        dispatched_ids = {r["id"] for r in dispatched}
        expected_ids = {f"req-{i}" for i in range(20)}
        self.assertEqual(dispatched_ids, expected_ids)

    def test_dispatcher_groups_same_prefix(self):
        """Requests sharing a prefix must get the same group hash."""
        sequences, labels = _make_poisson_interleaved(
            num_groups=3, prefix_len=200, n=30, seed=55,
        )
        requests = [
            {"id": f"req-{i}", "token_ids": seq, "group": labels[i]}
            for i, seq in enumerate(sequences)
        ]

        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        for req in requests:
            dispatcher.submit(req)

        self.assertEqual(len(dispatched), 30)

        # Group by hash, check that each hash maps to exactly one label
        from collections import defaultdict
        hash_to_labels = defaultdict(set)
        for r in dispatched:
            ghash = r["rid"].split(":", 3)[2]
            hash_to_labels[ghash].add(r["group"])

        for ghash, label_set in hash_to_labels.items():
            self.assertEqual(len(label_set), 1,
                             f"Hash {ghash} maps to multiple groups: {label_set}")


# ---------------------------------------------------------------------------
# Tests: End-to-end Poisson + DFS reorder pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPoissonDfsReorder(unittest.TestCase):
    """End-to-end: generate Poisson workload → reorder → verify grouping."""

    def test_shared_system_prompts_poisson_reorder(self):
        """Simulate the shared_system_prompts workload with Poisson arrivals
        and verify DFS reorder produces good prefix grouping."""
        from peek.offline.benchmarks.poisson_client import load_requests

        reqs = load_requests(
            "shared_system_prompts", n=100, num_groups=10,
            system_prompt_len=256,
        )
        # Extract group labels from metadata
        labels = []
        for r in reqs:
            group = r.get("group", "")
            # "sys-0", "sys-1", etc.
            try:
                labels.append(int(group.split("-")[1]))
            except (IndexError, ValueError):
                labels.append(-1)

        # Simulate Poisson-interleaved arrival (shuffle)
        rng = random.Random(42)
        indices = list(range(len(reqs)))
        rng.shuffle(indices)
        shuffled_seqs = [reqs[i]["token_ids"] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        # Reorder
        order = reorder_for_prefix_sharing(shuffled_seqs)

        self.assertEqual(set(order), set(range(100)))

        # Check adjacency
        arrival_adj = _adjacency_score(list(range(100)), shuffled_labels)
        reordered_labels = [shuffled_labels[i] for i in order]
        reordered_adj = _adjacency_score(list(range(100)), reordered_labels)

        self.assertGreater(reordered_adj, arrival_adj,
                           "DFS reorder must improve grouping over shuffled arrival")
        self.assertGreater(reordered_adj, 0.85,
                           f"Expected high adjacency for shared_system_prompts, "
                           f"got {reordered_adj:.2f}")

    def test_few_shot_mmlu_poisson_reorder(self):
        """few_shot_mmlu workload: subjects share shot prefixes."""
        from peek.offline.benchmarks.poisson_client import load_requests

        reqs = load_requests("few_shot_mmlu", n=60)
        labels = []
        for r in reqs:
            group = r.get("group", "")
            labels.append(group)

        # Shuffle to simulate Poisson interleaving
        rng = random.Random(77)
        indices = list(range(len(reqs)))
        rng.shuffle(indices)
        shuffled_seqs = [reqs[i]["token_ids"] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        order = reorder_for_prefix_sharing(shuffled_seqs)
        self.assertEqual(set(order), set(range(60)))

        # Verify improvement
        arrival_adj = _adjacency_score(
            list(range(60)),
            # Convert labels to ints for adjacency scoring
            [hash(l) for l in shuffled_labels],
        )
        reordered_labels = [shuffled_labels[i] for i in order]
        reordered_adj = _adjacency_score(
            list(range(60)),
            [hash(l) for l in reordered_labels],
        )
        self.assertGreaterEqual(reordered_adj, arrival_adj,
                                "DFS reorder must not degrade grouping")


if __name__ == "__main__":
    unittest.main()
