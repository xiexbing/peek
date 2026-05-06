#!/usr/bin/env python3
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

"""Test DFS trie construction, add, and remove with Qwen 2.5 32B workload.

Builds the full trie from 1000 Poisson-interleaved requests (100 groups,
Zipf 1.5, 2048-token system prompts), then verifies that a single add
and a single remove each update the trie correctly: group keys, DFS
order, counts, sharing score, and dispatcher ranks all stay consistent.
"""
import random
import unittest
from collections import Counter, defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.reorder import PeekDispatcher
from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Workload — Qwen 2.5 32B, 100 groups, 1000 requests, Zipf 1.5
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
MAX_DEPTH = 128
SEED = 42


def _generate_workload():
    reqs = load_requests(
        "shared_system_prompts",
        n=N_REQUESTS,
        num_groups=NUM_GROUPS,
        system_prompt_len=SYSTEM_PROMPT_LEN,
        workload_kwargs={
            "group_distribution": "zipf",
            "zipf_alpha": ZIPF_ALPHA,
            "question_len": QUESTION_LEN,
        },
    )
    labels = []
    for r in reqs:
        try:
            labels.append(int(r.get("group", "sys-0").split("-")[1]))
        except (IndexError, ValueError):
            labels.append(-1)
    return reqs, labels


def _poisson_order(n, seed=SEED):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order


# ===================================================================
# Test: bare PrefixTrie — build, add, remove
# ===================================================================

class TestTrieBuildAddRemove(unittest.TestCase):
    """Bare PrefixTrie: build from 1000 requests, add one, remove one."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _poisson_order(N_REQUESTS)
        cls.group_counts = Counter(cls.labels)

    # ---------------------------------------------------------------
    # Phase 1: Build trie from 1000 Poisson arrivals
    # ---------------------------------------------------------------

    def _build_trie(self):
        trie = PrefixTrie(max_depth=MAX_DEPTH)
        for idx in self.arrival_order:
            trie.insert(self.reqs[idx]["token_ids"], idx)
        return trie

    def test_build_prompt_count(self):
        trie = self._build_trie()
        self.assertEqual(trie._num_prompts, N_REQUESTS)

    def test_build_group_count(self):
        """Number of leaf groups equals number of distinct prefix keys."""
        trie = self._build_trie()
        expected = len({
            tuple(self.reqs[i]["token_ids"][:MAX_DEPTH])
            for i in range(N_REQUESTS)
        })
        actual = len(trie.dfs_group_keys())
        self.assertEqual(actual, expected)

    def test_build_dfs_order_covers_all(self):
        trie = self._build_trie()
        dfs = trie.dfs_order(count_aware=True)
        self.assertEqual(len(dfs), N_REQUESTS)
        self.assertEqual(set(dfs), set(range(N_REQUESTS)))

    def test_build_dfs_groups_same_label_adjacent(self):
        """In the DFS order, consecutive entries from the same group
        must be adjacent (no interleaving with other groups)."""
        trie = self._build_trie()
        dfs = trie.dfs_order(count_aware=True)
        dfs_labels = [self.labels[i] for i in dfs]

        # Walk through DFS: each time the label changes, the previous
        # label must not appear again later.
        seen_complete: set[int] = set()
        prev = dfs_labels[0]
        for lbl in dfs_labels[1:]:
            if lbl != prev:
                seen_complete.add(prev)
                self.assertNotIn(
                    lbl, seen_complete,
                    f"Group {lbl} reappears after being fully traversed — "
                    f"DFS interleaving detected",
                )
                prev = lbl

    def test_build_count_aware_largest_first(self):
        """count_aware DFS must place the largest group first."""
        trie = self._build_trie()
        dfs = trie.dfs_order(count_aware=True)

        largest_group = self.group_counts.most_common(1)[0][0]
        largest_size = self.group_counts[largest_group]

        first_n_labels = [self.labels[dfs[i]] for i in range(largest_size)]
        self.assertTrue(
            all(lbl == largest_group for lbl in first_n_labels),
            f"First {largest_size} DFS entries should all be group "
            f"{largest_group}, got {Counter(first_n_labels)}",
        )

    def test_build_sharing_score(self):
        trie = self._build_trie()
        coverage, max_group, avg_depth = trie.sharing_score(min_depth=32)
        self.assertGreater(coverage, 0.90)
        self.assertGreater(avg_depth, 64)
        self.assertGreater(max_group, 50)

    # ---------------------------------------------------------------
    # Phase 2: Add one new request
    # ---------------------------------------------------------------

    def test_add_to_existing_group(self):
        """Insert a 1001st request into the largest group.  Verify:
        - prompt count increments
        - group count unchanged (same prefix key)
        - DFS order includes the new index
        - new index is adjacent to its group in DFS
        - count of the largest group increases by 1
        """
        trie = self._build_trie()
        largest_group = self.group_counts.most_common(1)[0][0]

        # Pick a request from the largest group as template
        template_idx = next(i for i, l in enumerate(self.labels) if l == largest_group)
        template_tids = self.reqs[template_idx]["token_ids"]

        # Build new request: same prefix, different question suffix
        rng = random.Random(9999)
        new_tids = template_tids[:SYSTEM_PROMPT_LEN] + [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]
        new_idx = N_REQUESTS  # 1000

        groups_before = len(trie.dfs_group_keys())
        count_before = trie._num_prompts

        trie.insert(new_tids, new_idx)

        # Prompt count
        self.assertEqual(trie._num_prompts, count_before + 1)
        # Group count unchanged — same prefix key
        self.assertEqual(len(trie.dfs_group_keys()), groups_before)
        # DFS includes new index
        dfs = trie.dfs_order(count_aware=True)
        self.assertIn(new_idx, dfs)
        self.assertEqual(len(dfs), N_REQUESTS + 1)

        # New index is adjacent to its group in DFS
        pos = dfs.index(new_idx)
        key_new = tuple(new_tids[:MAX_DEPTH])
        # Check neighbor has same prefix key
        neighbor_found = False
        if pos > 0:
            neighbor_key = tuple(self.reqs[dfs[pos - 1]]["token_ids"][:MAX_DEPTH]) if dfs[pos - 1] < N_REQUESTS else key_new
            if neighbor_key == key_new:
                neighbor_found = True
        if pos < len(dfs) - 1:
            neighbor_key = tuple(self.reqs[dfs[pos + 1]]["token_ids"][:MAX_DEPTH]) if dfs[pos + 1] < N_REQUESTS else key_new
            if neighbor_key == key_new:
                neighbor_found = True
        self.assertTrue(neighbor_found,
                        f"New index at DFS pos {pos} has no same-group neighbor")

    def test_add_new_group(self):
        """Insert a request with a completely new prefix.  Verify:
        - prompt count increments
        - group count increments by 1
        - new group key appears in dfs_group_keys
        - DFS order includes the new index
        - sharing score coverage stays high (one more singleton doesn't hurt much)
        """
        trie = self._build_trie()
        groups_before = len(trie.dfs_group_keys())
        keys_before = set(trie.dfs_group_keys())

        # Create a request with a brand-new prefix
        rng = random.Random(77777)
        new_tids = [rng.randint(1, 31999) for _ in range(SYSTEM_PROMPT_LEN + QUESTION_LEN)]
        new_key = tuple(new_tids[:MAX_DEPTH])
        new_idx = N_REQUESTS

        # Ensure this key doesn't already exist
        self.assertNotIn(new_key, keys_before)

        trie.insert(new_tids, new_idx)

        # Group count +1
        keys_after = set(trie.dfs_group_keys())
        self.assertEqual(len(keys_after), groups_before + 1)
        # New key present
        self.assertIn(new_key, keys_after)
        # Old keys unchanged
        self.assertEqual(keys_before, keys_after - {new_key})
        # DFS includes new
        dfs = trie.dfs_order(count_aware=True)
        self.assertIn(new_idx, dfs)
        # Coverage still high (one more singleton barely affects it)
        coverage, _, _ = trie.sharing_score(min_depth=32)
        self.assertGreater(coverage, 0.88)

    # ---------------------------------------------------------------
    # Phase 3: Remove one request
    # ---------------------------------------------------------------

    def test_remove_from_large_group(self):
        """Remove one request from the largest group.  Verify:
        - prompt count decrements
        - group count unchanged (group still has members)
        - removed index gone from DFS order
        - remaining DFS order is still correctly grouped
        - largest group still leads the DFS (count_aware)
        """
        trie = self._build_trie()
        largest_group = self.group_counts.most_common(1)[0][0]
        largest_size = self.group_counts[largest_group]

        # Pick the first request from the largest group
        victim_idx = next(i for i, l in enumerate(self.labels) if l == largest_group)
        victim_tids = self.reqs[victim_idx]["token_ids"]

        groups_before = len(trie.dfs_group_keys())

        trie.remove(victim_tids, victim_idx)

        # Prompt count
        self.assertEqual(trie._num_prompts, N_REQUESTS - 1)
        # Group count unchanged
        self.assertEqual(len(trie.dfs_group_keys()), groups_before)
        # Removed index gone from DFS
        dfs = trie.dfs_order(count_aware=True)
        self.assertNotIn(victim_idx, dfs)
        self.assertEqual(len(dfs), N_REQUESTS - 1)

        # DFS still groups correctly (no interleaving)
        dfs_labels = [self.labels[i] for i in dfs]
        seen_complete: set[int] = set()
        prev = dfs_labels[0]
        for lbl in dfs_labels[1:]:
            if lbl != prev:
                seen_complete.add(prev)
                self.assertNotIn(lbl, seen_complete,
                                 f"Group {lbl} interleaved after remove")
                prev = lbl

        # Largest group still first (lost 1 member but still biggest)
        first_label = self.labels[dfs[0]]
        self.assertEqual(first_label, largest_group)

    def test_remove_singleton_group_disappears(self):
        """Remove the only request in a singleton group.  Verify:
        - prompt count decrements
        - group count decrements by 1
        - group key gone from dfs_group_keys
        - remaining groups unaffected
        """
        trie = self._build_trie()
        groups_before = set(trie.dfs_group_keys())

        # Find a singleton group (count == 1)
        singleton_group = None
        for grp, cnt in self.group_counts.items():
            if cnt == 1:
                singleton_group = grp
                break
        if singleton_group is None:
            self.skipTest("No singleton group in this workload")

        victim_idx = next(i for i, l in enumerate(self.labels) if l == singleton_group)
        victim_tids = self.reqs[victim_idx]["token_ids"]
        victim_key = tuple(victim_tids[:MAX_DEPTH])

        trie.remove(victim_tids, victim_idx)

        # Group gone
        groups_after = set(trie.dfs_group_keys())
        self.assertEqual(len(groups_after), len(groups_before) - 1)
        self.assertNotIn(victim_key, groups_after)
        # Others intact
        self.assertEqual(groups_before - {victim_key}, groups_after)
        # Prompt count
        self.assertEqual(trie._num_prompts, N_REQUESTS - 1)

    def test_remove_then_add_same_group(self):
        """Remove a request, then add a new one to the same group.
        The group should survive with correct count."""
        trie = self._build_trie()
        largest_group = self.group_counts.most_common(1)[0][0]
        largest_size = self.group_counts[largest_group]

        # Remove one
        victim_idx = next(i for i, l in enumerate(self.labels) if l == largest_group)
        victim_tids = self.reqs[victim_idx]["token_ids"]
        trie.remove(victim_tids, victim_idx)

        self.assertEqual(trie._num_prompts, N_REQUESTS - 1)

        # Add new to same group
        rng = random.Random(12345)
        new_tids = victim_tids[:SYSTEM_PROMPT_LEN] + [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]
        new_idx = N_REQUESTS
        trie.insert(new_tids, new_idx)

        self.assertEqual(trie._num_prompts, N_REQUESTS)

        # DFS: new_idx present, victim_idx gone
        dfs = trie.dfs_order(count_aware=True)
        self.assertIn(new_idx, dfs)
        self.assertNotIn(victim_idx, dfs)
        self.assertEqual(len(dfs), N_REQUESTS)


# ===================================================================
# Test: PeekDispatcher — add and remove update tags correctly
# ===================================================================

class TestDispatcherAddRemove(unittest.TestCase):
    """PeekDispatcher: build from 1000 arrivals, add one, remove one,
    verify tags and ranks stay consistent."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _poisson_order(N_REQUESTS)
        cls.group_counts = Counter(cls.labels)

    def _build_dispatcher(self):
        dispatched = []
        dispatcher = PeekDispatcher(
            send_fn=lambda r: dispatched.append(dict(r)),
        )
        # Track original_index → dispatcher prompt_index
        self._orig_to_prompt = {}
        for idx in self.arrival_order:
            prompt_idx = dispatcher._next_idx
            dispatcher.submit(dict(self.reqs[idx]))
            self._orig_to_prompt[idx] = prompt_idx
        return dispatcher, dispatched

    def test_add_existing_group_rank_unchanged(self):
        """Adding to the already-largest group should not change its rank."""
        dispatcher, dispatched = self._build_dispatcher()
        largest_group = self.group_counts.most_common(1)[0][0]

        # Last request from largest group should be rank 0
        last_largest = [d for d in dispatched if d["group"] == f"sys-{largest_group}"][-1]
        rank_before = int(last_largest["rid"].split(":")[1])
        self.assertEqual(rank_before, 0)

        # Add new request to same group
        template_idx = next(i for i, l in enumerate(self.labels) if l == largest_group)
        rng = random.Random(5555)
        new_tids = self.reqs[template_idx]["token_ids"][:SYSTEM_PROMPT_LEN] + \
                   [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]

        dispatched.clear()
        dispatcher.submit({"id": "new-largest", "token_ids": new_tids, "group": f"sys-{largest_group}"})

        rank_after = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(rank_after, 0,
                         f"Largest group should still be rank 0 after add, got {rank_after}")

    def test_add_new_group_gets_bottom_rank(self):
        """A brand-new singleton group should rank in the bottom half.

        It can't be guaranteed the very last rank because other
        singletons also have count=1 and count-aware DFS ordering
        among equal-count groups depends on trie structure.
        """
        dispatcher, dispatched = self._build_dispatcher()
        n_groups_before = len(dispatcher._group_count)

        rng = random.Random(88888)
        new_tids = [rng.randint(1, 31999) for _ in range(SYSTEM_PROMPT_LEN + QUESTION_LEN)]

        dispatched.clear()
        dispatcher.submit({"id": "new-singleton", "token_ids": new_tids, "group": "sys-new"})

        rank = int(dispatched[0]["rid"].split(":")[1])
        n_groups_after = len(dispatcher._group_count)
        self.assertEqual(n_groups_after, n_groups_before + 1)
        # Singleton (count=1) should be in the bottom half
        self.assertGreater(rank, n_groups_after // 2,
                           f"New singleton should be in bottom half, "
                           f"got rank {rank} of {n_groups_after}")

    def test_remove_updates_rank_on_next_submit(self):
        """Remove requests from the largest group until a smaller group
        overtakes it, then verify the next submit reflects the new ranking."""
        dispatcher, dispatched = self._build_dispatcher()

        top2 = self.group_counts.most_common(2)
        group_a, count_a = top2[0]  # largest
        group_b, count_b = top2[1]  # second largest

        # Collect original indices for group A → dispatcher prompt indices
        group_a_orig_indices = [idx for idx in self.arrival_order if self.labels[idx] == group_a]

        # Remove enough from A so that B has more pending
        to_remove = count_a - count_b + 1
        for i in range(to_remove):
            orig_idx = group_a_orig_indices[i]
            prompt_idx = self._orig_to_prompt[orig_idx]
            dispatcher.remove(self.reqs[orig_idx]["token_ids"], prompt_idx)

        a_remaining = count_a - to_remove
        self.assertLess(a_remaining, count_b,
                        f"A should have fewer pending ({a_remaining}) than B ({count_b})")

        # Next submit for group B should get rank 0
        dispatched.clear()
        b_sample = next(i for i, l in enumerate(self.labels) if l == group_b)
        rng = random.Random(3333)
        new_tids = self.reqs[b_sample]["token_ids"][:SYSTEM_PROMPT_LEN] + \
                   [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]
        dispatcher.submit({"id": "b-after-remove", "token_ids": new_tids, "group": f"sys-{group_b}"})

        rank_b = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(rank_b, 0,
                         f"Group B ({count_b} pending) should be rank 0 after "
                         f"removing from A ({a_remaining} pending), got rank {rank_b}")

        # Submit for group A should get rank 1
        dispatched.clear()
        a_sample = next(i for i, l in enumerate(self.labels) if l == group_a)
        new_tids_a = self.reqs[a_sample]["token_ids"][:SYSTEM_PROMPT_LEN] + \
                     [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]
        dispatcher.submit({"id": "a-after-remove", "token_ids": new_tids_a, "group": f"sys-{group_a}"})

        rank_a = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(rank_a, 1,
                         f"Group A ({a_remaining + 1} pending) should be rank 1, got {rank_a}")

    def test_remove_all_group_then_add_back(self):
        """Remove every request from a group, then add one back.
        The group should reappear with rank reflecting its count=1."""
        dispatcher, dispatched = self._build_dispatcher()

        # Pick a small group (count ~5-10)
        small_group = None
        for grp, cnt in self.group_counts.most_common():
            if 3 <= cnt <= 10:
                small_group = grp
                break
        self.assertIsNotNone(small_group, "Need a group with 3-10 requests")
        small_count = self.group_counts[small_group]

        # Collect its original indices → dispatcher prompt indices
        small_orig_indices = [idx for idx in self.arrival_order if self.labels[idx] == small_group]
        self.assertEqual(len(small_orig_indices), small_count)

        # Remove all using correct dispatcher prompt indices
        for orig_idx in small_orig_indices:
            prompt_idx = self._orig_to_prompt[orig_idx]
            dispatcher.remove(self.reqs[orig_idx]["token_ids"], prompt_idx)

        # Group should be gone from trie
        group_keys = dispatcher._trie.dfs_group_keys()
        small_key = tuple(self.reqs[small_orig_indices[0]]["token_ids"][:MAX_DEPTH])
        keys_set = set(group_keys)
        self.assertNotIn(small_key, keys_set,
                         "Removed group should be gone from trie")

        # Add one back
        dispatched.clear()
        rng = random.Random(4444)
        new_tids = self.reqs[small_orig_indices[0]]["token_ids"][:SYSTEM_PROMPT_LEN] + \
                   [rng.randint(1, 31999) for _ in range(QUESTION_LEN)]
        dispatcher.submit({"id": "comeback", "token_ids": new_tids, "group": f"sys-{small_group}"})

        # Group reappears
        group_keys = dispatcher._trie.dfs_group_keys()
        self.assertIn(small_key, set(group_keys))

        # Rank should be near the bottom (count=1 vs groups with 50+ pending)
        rank = int(dispatched[0]["rid"].split(":")[1])
        n_groups = len(group_keys)
        self.assertGreater(rank, n_groups // 2,
                           f"Returning singleton should be in bottom half, "
                           f"got rank {rank} of {n_groups}")

    def test_trie_count_equals_pending_throughout(self):
        """Through a sequence of adds and removes, the trie prompt count
        must always equal the number of pending requests."""
        dispatcher, dispatched = self._build_dispatcher()
        expected_pending = N_REQUESTS

        self.assertEqual(dispatcher._trie._num_prompts, expected_pending)

        # Remove 50 from the largest group (using correct prompt indices)
        largest_group = self.group_counts.most_common(1)[0][0]
        orig_indices = [idx for idx in self.arrival_order if self.labels[idx] == largest_group]

        for orig_idx in orig_indices[:50]:
            prompt_idx = self._orig_to_prompt[orig_idx]
            dispatcher.remove(self.reqs[orig_idx]["token_ids"], prompt_idx)
            expected_pending -= 1
            self.assertEqual(dispatcher._trie._num_prompts, expected_pending,
                             f"After removing orig_idx={orig_idx}")

        # Add 10 new requests
        rng = random.Random(6666)
        for i in range(10):
            new_tids = [rng.randint(1, 31999) for _ in range(SYSTEM_PROMPT_LEN + QUESTION_LEN)]
            dispatcher.submit({"id": f"extra-{i}", "token_ids": new_tids})
            expected_pending += 1
            self.assertEqual(dispatcher._trie._num_prompts, expected_pending,
                             f"After adding extra-{i}")

        # Remove 5 more from the largest group
        for orig_idx in orig_indices[50:55]:
            prompt_idx = self._orig_to_prompt[orig_idx]
            dispatcher.remove(self.reqs[orig_idx]["token_ids"], prompt_idx)
            expected_pending -= 1
            self.assertEqual(dispatcher._trie._num_prompts, expected_pending,
                             f"After second-round remove orig_idx={orig_idx}")


if __name__ == "__main__":
    unittest.main()
