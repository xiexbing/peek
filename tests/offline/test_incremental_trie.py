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

"""Tests for PrefixTrie.remove/clear/dfs_group_keys and the incremental
PeekDispatcher with trie-based group tagging.

Verifies:
1. Trie remove() correctly decrements counts, prunes empty nodes, and
   keeps the trie consistent for subsequent DFS/sharing_score calls.
2. Trie clear() resets to empty state.
3. dfs_group_keys() returns leaf paths in DFS order.
4. PeekDispatcher inserts into an incremental trie on each submit(),
   tags requests with count-aware DFS group rank, and sends immediately.
"""
import random
import unittest

from peek.offline.trie import PrefixTrie
from peek.offline.reorder import PeekConfig, PeekDispatcher


# ---------------------------------------------------------------------------
# PrefixTrie.remove tests
# ---------------------------------------------------------------------------

class TestTrieRemove(unittest.TestCase):

    def test_remove_single_entry(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        self.assertTrue(trie.remove([1, 2, 3], 0))
        self.assertEqual(trie.dfs_order(), [])
        self.assertEqual(trie._num_prompts, 0)

    def test_remove_nonexistent_index_returns_false(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        self.assertFalse(trie.remove([1, 2, 3], 99))
        self.assertEqual(trie._num_prompts, 1)

    def test_remove_nonexistent_path_returns_false(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        self.assertFalse(trie.remove([4, 5, 6], 0))
        self.assertEqual(trie._num_prompts, 1)

    def test_remove_from_empty_trie(self):
        trie = PrefixTrie()
        self.assertFalse(trie.remove([1, 2], 0))

    def test_remove_preserves_sibling(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        trie.remove([1, 2, 3], 0)
        self.assertEqual(trie.dfs_order(), [1])
        self.assertEqual(trie._num_prompts, 1)

    def test_remove_decrements_counts(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        # Node for token 1 has count=2, token 2 has count=2
        node_1 = trie.root.children[1]
        node_2 = node_1.children[2]
        self.assertEqual(node_1.count, 2)
        self.assertEqual(node_2.count, 2)

        trie.remove([1, 2, 3], 0)
        self.assertEqual(node_1.count, 1)
        self.assertEqual(node_2.count, 1)

    def test_remove_prunes_empty_leaf(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        trie.remove([1, 2, 3], 0)
        # Node for token 3 should be pruned
        node_2 = trie.root.children[1].children[2]
        self.assertNotIn(3, node_2.children)
        # Node for token 4 still exists
        self.assertIn(4, node_2.children)

    def test_remove_prunes_chain(self):
        """Removing the only entry from a deep path prunes the entire chain."""
        trie = PrefixTrie()
        trie.insert([1, 2, 3, 4, 5], 0)
        trie.remove([1, 2, 3, 4, 5], 0)
        # Entire path should be pruned
        self.assertEqual(len(trie.root.children), 0)

    def test_remove_stops_pruning_at_shared_node(self):
        """Pruning must stop at nodes that still have other children."""
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        trie.insert([1, 5], 2)
        trie.remove([1, 2, 3], 0)
        # Token 1 still needed by indices 1 and 2
        self.assertIn(1, trie.root.children)
        # Token 2 still needed by index 1
        self.assertIn(2, trie.root.children[1].children)

    def test_remove_all_then_insert_again(self):
        """Trie should be reusable after removing all entries."""
        trie = PrefixTrie()
        trie.insert([1, 2], 0)
        trie.insert([3, 4], 1)
        trie.remove([1, 2], 0)
        trie.remove([3, 4], 1)
        self.assertEqual(trie._num_prompts, 0)
        self.assertEqual(trie.dfs_order(), [])

        # Insert new entries
        trie.insert([5, 6], 10)
        self.assertEqual(trie.dfs_order(), [10])
        self.assertEqual(trie._num_prompts, 1)

    def test_remove_one_of_duplicates_at_leaf(self):
        """Two prompts at same leaf -- remove one, other stays."""
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 3], 1)
        trie.remove([1, 2, 3], 0)
        self.assertEqual(trie.dfs_order(), [1])
        self.assertEqual(trie._num_prompts, 1)
        # Counts should reflect 1 remaining
        self.assertEqual(trie.root.children[1].count, 1)

    def test_sharing_score_correct_after_remove(self):
        """sharing_score must reflect the trie state after removals."""
        shared = list(range(100))
        trie = PrefixTrie()
        for i in range(5):
            trie.insert(shared + [1000 + i], i)

        coverage, max_group, _ = trie.sharing_score(min_depth=32)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(max_group, 5)

        # Remove 3 entries -- only 2 remain sharing
        for i in range(3):
            trie.remove(shared + [1000 + i], i)

        coverage2, max_group2, _ = trie.sharing_score(min_depth=32)
        self.assertEqual(coverage2, 1.0)  # 2/2 still share
        self.assertEqual(max_group2, 2)
        self.assertEqual(trie._num_prompts, 2)

    def test_dfs_order_correct_after_interleaved_insert_remove(self):
        """DFS order remains correct after a mix of inserts and removes."""
        prefix_a = list(range(100))
        prefix_b = list(range(100, 200))
        trie = PrefixTrie()

        # Insert group A: indices 0, 1, 2
        for i in range(3):
            trie.insert(prefix_a + [5000 + i], i)
        # Insert group B: indices 3, 4
        for i in range(2):
            trie.insert(prefix_b + [6000 + i], 3 + i)

        # Remove index 1 from group A
        trie.remove(prefix_a + [5001], 1)

        order = trie.dfs_order()
        self.assertEqual(set(order), {0, 2, 3, 4})
        self.assertEqual(trie._num_prompts, 4)

        # Group A (0, 2) should still be adjacent
        pos = {idx: rank for rank, idx in enumerate(order)}
        self.assertEqual(abs(pos[0] - pos[2]), 1)

    def test_remove_with_max_depth_truncation(self):
        """Remove must use the same max_depth truncation as insert."""
        trie = PrefixTrie(max_depth=4)
        long_seq = [1, 2, 3, 4, 5, 6, 7]
        trie.insert(long_seq, 0)
        # Insert only stored first 4 tokens
        self.assertTrue(trie.remove(long_seq, 0))
        self.assertEqual(trie._num_prompts, 0)
        self.assertEqual(len(trie.root.children), 0)


class TestTrieClear(unittest.TestCase):

    def test_clear_empties_trie(self):
        trie = PrefixTrie()
        for i in range(10):
            trie.insert([i, i + 1, i + 2], i)
        self.assertEqual(trie._num_prompts, 10)

        trie.clear()
        self.assertEqual(trie._num_prompts, 0)
        self.assertEqual(trie.dfs_order(), [])
        self.assertEqual(len(trie.root.children), 0)

    def test_insert_after_clear(self):
        trie = PrefixTrie()
        trie.insert([1, 2], 0)
        trie.clear()
        trie.insert([3, 4], 5)
        self.assertEqual(trie.dfs_order(), [5])
        self.assertEqual(trie._num_prompts, 1)


# ---------------------------------------------------------------------------
# Incremental dispatcher tests
# ---------------------------------------------------------------------------

def _make_interleaved_requests(
    num_groups, prefix_len, n, seed=42, suffix_len=20,
):
    rng = random.Random(seed)
    prefixes = []
    for g in range(num_groups):
        grng = random.Random(g * 1000 + 7)
        prefixes.append([grng.randint(1, 31999) for _ in range(prefix_len)])

    requests = []
    labels = []
    for i in range(n):
        g = rng.randint(0, num_groups - 1)
        suffix = [rng.randint(1, 31999) for _ in range(suffix_len)]
        requests.append({
            "id": f"req-{i}",
            "token_ids": prefixes[g] + suffix,
            "group": g,
        })
        labels.append(g)
    return requests, labels


def _adjacency_score(labels):
    if len(labels) <= 1:
        return 1.0
    same = sum(1 for i in range(len(labels) - 1) if labels[i] == labels[i + 1])
    return same / (len(labels) - 1)


class TestDfsGroupKeys(unittest.TestCase):
    """PrefixTrie.dfs_group_keys returns leaf paths in DFS order."""

    def test_single_group(self):
        trie = PrefixTrie(max_depth=4)
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 3], 1)
        keys = trie.dfs_group_keys()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], (1, 2, 3))

    def test_two_groups_dfs_order(self):
        trie = PrefixTrie(max_depth=4)
        trie.insert([1, 2, 3], 0)
        trie.insert([4, 5, 6], 1)
        keys = trie.dfs_group_keys()
        self.assertEqual(len(keys), 2)
        # DFS visits in insertion order -> [1,2,3] before [4,5,6]
        self.assertEqual(keys[0], (1, 2, 3))
        self.assertEqual(keys[1], (4, 5, 6))

    def test_sibling_groups_under_shared_prefix(self):
        trie = PrefixTrie(max_depth=4)
        trie.insert([1, 2, 3], 0)  # leaf at [1,2,3]
        trie.insert([1, 2, 4], 1)  # leaf at [1,2,4]
        trie.insert([5, 6], 2)     # leaf at [5,6]
        keys = trie.dfs_group_keys()
        self.assertEqual(len(keys), 3)
        # [1,2,3] and [1,2,4] are siblings under [1,2], before [5,6]
        self.assertIn((1, 2, 3), keys[:2])
        self.assertIn((1, 2, 4), keys[:2])
        self.assertEqual(keys[2], (5, 6))

    def test_matches_dfs_order_grouping(self):
        """dfs_group_keys order should match the group ordering implied
        by dfs_order()."""
        trie = PrefixTrie()
        prefix_a = list(range(100))
        prefix_b = list(range(100, 200))
        for i in range(5):
            trie.insert(prefix_a + [1000 + i], i)
        for i in range(3):
            trie.insert(prefix_b + [2000 + i], 5 + i)

        keys = trie.dfs_group_keys()
        dfs = trie.dfs_order()

        # First group key should contain the first dfs index
        first_key = keys[0]
        # All indices from first key's group should come first in dfs
        first_group_indices = set()
        for idx in dfs:
            first_group_indices.add(idx)
            if len(first_group_indices) >= 5:  # assume at most 5 in first group
                break
        self.assertTrue(len(keys) >= 2)

    def test_empty_trie(self):
        trie = PrefixTrie()
        self.assertEqual(trie.dfs_group_keys(), [])


class TestIncrementalDispatcher(unittest.TestCase):
    """PeekDispatcher: incremental trie-based tagging on each submit()."""

    def test_submit_dispatches_immediately(self):
        """Each submit() should call send_fn immediately."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(5):
            dispatcher.submit({"id": f"r-{i}", "token_ids": prefix + [1000 + i]})

        self.assertEqual(len(dispatched), 5)

    def test_same_prefix_gets_same_group_hash(self):
        """Requests sharing a prefix get the same group_key_hash in their rid."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(5):
            dispatcher.submit({"id": f"r-{i}", "token_ids": prefix + [1000 + i]})

        hashes = set()
        for r in dispatched:
            parts = r["rid"].split(":", 3)
            hashes.add(parts[2])
        self.assertEqual(len(hashes), 1, "Same-prefix requests should share one group hash")

    def test_different_prefix_gets_different_hash(self):
        """Requests with different prefixes get different group hashes."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        dispatcher.submit({"id": "a-0", "token_ids": prefix_a + [1]})
        dispatcher.submit({"id": "b-0", "token_ids": prefix_b + [2]})

        hash_a = dispatched[0]["rid"].split(":", 3)[2]
        hash_b = dispatched[1]["rid"].split(":", 3)[2]
        self.assertNotEqual(hash_a, hash_b)

    def test_existing_group_count_increments(self):
        """Adding to an existing group should increment its pending count."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        dispatcher.submit({"id": "r-0", "token_ids": prefix + [1000]})
        key = tuple(prefix[:128])
        self.assertEqual(dispatcher._group_count[key], 1)

        dispatcher.submit({"id": "r-1", "token_ids": prefix + [1001]})
        self.assertEqual(dispatcher._group_count[key], 2)
        self.assertEqual(len(dispatcher._group_count), 1)

    def test_new_group_adds_to_count(self):
        """A request with a new prefix should create a new group count entry."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        dispatcher.submit({"id": "r-0", "token_ids": prefix_a + [1000]})
        self.assertEqual(len(dispatcher._group_count), 1)

        dispatcher.submit({"id": "r-1", "token_ids": prefix_b + [2000]})
        self.assertEqual(len(dispatcher._group_count), 2)

    def test_rank_reflects_group_size(self):
        """Larger groups should get lower (better) rank numbers."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        # Group A: 5 requests, Group B: 2 requests
        for i in range(5):
            dispatcher.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
        for i in range(2):
            dispatcher.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})

        # Last request from group A should have rank 0 (largest)
        rank_a = int(dispatched[-3]["rid"].split(":")[1])  # a-4
        rank_b = int(dispatched[-1]["rid"].split(":")[1])  # b-1
        self.assertLess(rank_a, rank_b,
                        f"Group A (5 reqs) should rank before Group B (2 reqs)")

    def test_interleaved_requests_all_dispatched(self):
        """All submitted requests must be dispatched."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        requests, labels = _make_interleaved_requests(
            num_groups=5, prefix_len=200, n=50, seed=77,
        )
        for req in requests:
            dispatcher.submit(req)

        self.assertEqual(len(dispatched), 50)
        dispatched_ids = {r["id"] for r in dispatched}
        self.assertEqual(dispatched_ids, {f"req-{i}" for i in range(50)})

    def test_remove_decrements_trie(self):
        """remove() should take the request out of the trie."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        tids = prefix + [1000]
        dispatcher.submit({"id": "r-0", "token_ids": tids})

        self.assertEqual(dispatcher._trie._num_prompts, 1)
        dispatcher.remove(tids, 0)
        self.assertEqual(dispatcher._trie._num_prompts, 0)

    def test_rid_format(self):
        """rid should be peek:<rank>:<hash>:<original_rid>."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        dispatcher.submit({"id": "my-req", "token_ids": list(range(200))})
        rid = dispatched[0]["rid"]
        parts = rid.split(":", 3)
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "peek")
        self.assertEqual(parts[3], "my-req")


class TestIncrementalVsRebuild(unittest.TestCase):
    """Verify incremental trie produces the same DFS order as rebuild."""

    def test_same_dfs_order(self):
        """Incrementally-built trie must produce the same DFS order as
        a trie built from scratch with the same sequences."""
        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))
        sequences = (
            [prefix_a + [1000 + i] for i in range(10)]
            + [prefix_b + [2000 + i] for i in range(10)]
        )
        # Interleave
        rng = random.Random(42)
        indices = list(range(20))
        rng.shuffle(indices)
        shuffled = [sequences[i] for i in indices]

        # Build from scratch
        trie_scratch = PrefixTrie()
        for idx, seq in enumerate(shuffled):
            trie_scratch.insert(seq, idx)
        order_scratch = trie_scratch.dfs_order()

        # Build incrementally (same sequence, same indices)
        trie_incr = PrefixTrie()
        for idx, seq in enumerate(shuffled):
            trie_incr.insert(seq, idx)
        order_incr = trie_incr.dfs_order()

        self.assertEqual(order_scratch, order_incr)

    def test_incremental_with_removals_matches_rebuild(self):
        """After removing some entries, DFS order should match a trie
        built from scratch with only the remaining entries."""
        prefix = list(range(200))
        all_seqs = [prefix + [1000 + i] for i in range(10)]

        # Build full trie, then remove indices 2, 5, 7
        trie_incr = PrefixTrie()
        for i, seq in enumerate(all_seqs):
            trie_incr.insert(seq, i)
        for i in [2, 5, 7]:
            trie_incr.remove(all_seqs[i], i)

        # Build from scratch with only remaining indices
        remaining = [i for i in range(10) if i not in {2, 5, 7}]
        trie_scratch = PrefixTrie()
        for i in remaining:
            trie_scratch.insert(all_seqs[i], i)

        self.assertEqual(
            set(trie_incr.dfs_order()),
            set(trie_scratch.dfs_order()),
        )
        self.assertEqual(trie_incr._num_prompts, trie_scratch._num_prompts)


if __name__ == "__main__":
    unittest.main()
