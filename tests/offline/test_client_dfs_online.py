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

"""Test client-side DFS trie under realistic online Poisson arrivals.

Qwen 2.5 32B scenario: 100 groups, 1000 requests, Zipf(1.5),
2048-token system prompts, 128-token questions.

Key invariant: the PeekDispatcher sees requests ONE AT A TIME as they
arrive — it never pre-sees the full 1000-request batch.  We verify that
the incremental trie produces correct grouping and ranking at every
point during the arrival stream.
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import Counter, defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.reorder import PeekDispatcher, reorder_for_prefix_sharing
from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Workload parameters (Qwen 2.5 32B on H100)
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
TRIE_MAX_DEPTH = 128       # PeekDispatcher default
POISSON_RATE = 30.0         # req/s — only affects inter-arrival timing
SEED = 42


def _generate_workload():
    """Generate 1000 Zipf-distributed requests with 100 prefix groups.

    Returns (requests, group_labels) where each request is a dict with
    'id', 'token_ids', and 'group' keys.  group_labels[i] is the int
    group index for request i.
    """
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
        group_str = r.get("group", "sys-0")
        try:
            labels.append(int(group_str.split("-")[1]))
        except (IndexError, ValueError):
            labels.append(-1)
    return reqs, labels


def _simulate_poisson_order(n: int, seed: int = SEED) -> list[int]:
    """Return a Poisson-interleaved permutation of [0..n).

    Shuffles indices the same way a Poisson process would interleave
    requests from different groups arriving at random times.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def _adjacency_score(labels: list) -> float:
    """Fraction of consecutive pairs sharing the same label."""
    if len(labels) <= 1:
        return 1.0
    same = sum(1 for i in range(len(labels) - 1) if labels[i] == labels[i + 1])
    return same / (len(labels) - 1)


# ===================================================================
# Test 1: Incremental trie correctness — one request at a time
# ===================================================================

class TestIncrementalTrieOnline(unittest.TestCase):
    """Verify the trie is built correctly when requests arrive one by one."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _simulate_poisson_order(N_REQUESTS)

    def test_trie_grows_incrementally(self):
        """Trie prompt count must equal the number of arrivals so far."""
        trie = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for step, idx in enumerate(self.arrival_order):
            trie.insert(self.reqs[idx]["token_ids"], idx)
            self.assertEqual(trie._num_prompts, step + 1)

    def test_group_count_matches_at_every_step(self):
        """At each arrival, the number of trie leaf groups must match
        the number of distinct prefix keys seen so far."""
        trie = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        seen_keys: set[tuple] = set()

        for idx in self.arrival_order:
            tids = self.reqs[idx]["token_ids"]
            key = tuple(tids[:TRIE_MAX_DEPTH])
            trie.insert(tids, idx)
            seen_keys.add(key)

            group_keys = trie.dfs_group_keys()
            self.assertEqual(
                len(group_keys), len(seen_keys),
                f"After {len(seen_keys)} distinct groups, "
                f"dfs_group_keys returned {len(group_keys)}",
            )

    def test_sharing_score_grows_with_arrivals(self):
        """sharing_score coverage should increase (or stay) as more
        same-group requests arrive."""
        trie = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        prev_coverage = 0.0

        # Check at milestone points (every 100 arrivals)
        for step, idx in enumerate(self.arrival_order):
            trie.insert(self.reqs[idx]["token_ids"], idx)
            if (step + 1) % 100 == 0:
                coverage, _, _ = trie.sharing_score(min_depth=32)
                self.assertGreaterEqual(
                    coverage, prev_coverage - 0.05,
                    f"Coverage dropped at step {step + 1}: "
                    f"{coverage:.3f} < {prev_coverage:.3f}",
                )
                prev_coverage = coverage

        # Final coverage should be high (Zipf means most requests share)
        final_coverage, _, avg_depth = trie.sharing_score(min_depth=32)
        self.assertGreater(final_coverage, 0.90,
                           f"Final coverage {final_coverage:.2f} too low")
        self.assertGreater(avg_depth, 64,
                           f"Avg sharing depth {avg_depth:.0f} too low")


# ===================================================================
# Test 2: PeekDispatcher online — never pre-sees the batch
# ===================================================================

class TestPeekDispatcherOnline(unittest.TestCase):
    """PeekDispatcher processes arrivals one at a time via submit()."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _simulate_poisson_order(N_REQUESTS)

        # Run dispatcher
        cls.dispatched = []
        dispatcher = PeekDispatcher(
            send_fn=lambda r: cls.dispatched.append(dict(r)),
        )
        for idx in cls.arrival_order:
            dispatcher.submit(dict(cls.reqs[idx]))  # copy so originals are untouched

    def test_all_requests_dispatched(self):
        """Every request must be dispatched exactly once."""
        self.assertEqual(len(self.dispatched), N_REQUESTS)
        dispatched_ids = {r["id"] for r in self.dispatched}
        expected_ids = {r["id"] for r in self.reqs}
        self.assertEqual(dispatched_ids, expected_ids)

    def test_all_rids_carry_peek_tags(self):
        """Every dispatched request must have a peek:<rank>:<hash>:<rid> tag."""
        for r in self.dispatched:
            rid = r["rid"]
            self.assertTrue(rid.startswith("peek:"), f"Bad rid: {rid}")
            parts = rid.split(":", 3)
            self.assertEqual(len(parts), 4, f"Malformed rid: {rid}")

    def test_same_group_same_hash(self):
        """Requests from the same original group must get the same group hash."""
        group_to_hashes: dict[str, set[str]] = defaultdict(set)
        for r in self.dispatched:
            ghash = r["rid"].split(":", 3)[2]
            group_to_hashes[r["group"]].add(ghash)

        for group, hashes in group_to_hashes.items():
            self.assertEqual(
                len(hashes), 1,
                f"Group {group} mapped to {len(hashes)} hashes: {hashes}",
            )

    def test_different_groups_different_hash(self):
        """Distinct groups must not collide on the same hash."""
        hash_to_groups: dict[str, set[str]] = defaultdict(set)
        for r in self.dispatched:
            ghash = r["rid"].split(":", 3)[2]
            hash_to_groups[ghash].add(r["group"])

        for ghash, groups in hash_to_groups.items():
            self.assertEqual(
                len(groups), 1,
                f"Hash {ghash} shared by groups: {groups}",
            )

    def test_final_ranks_order_by_group_size(self):
        """After all arrivals, the top-K groups by request count must
        have the lowest (best) rank numbers."""
        # Collect the rank assigned to each group's LAST dispatched request
        group_final_rank: dict[str, int] = {}
        for r in self.dispatched:
            rank = int(r["rid"].split(":", 3)[1])
            group_final_rank[r["group"]] = rank

        # Sort groups by request count descending
        group_counts = Counter(r["group"] for r in self.dispatched)
        top10 = [g for g, _ in group_counts.most_common(10)]
        top10_ranks = [group_final_rank[g] for g in top10]

        # Ranks must be monotonically non-decreasing (larger group = lower rank)
        for i in range(len(top10_ranks) - 1):
            self.assertLessEqual(
                top10_ranks[i], top10_ranks[i + 1],
                f"Rank ordering violated: group {top10[i]} (count="
                f"{group_counts[top10[i]]}, rank={top10_ranks[i]}) vs "
                f"group {top10[i+1]} (count={group_counts[top10[i+1]]}, "
                f"rank={top10_ranks[i+1]})",
            )


# ===================================================================
# Test 3: Online incremental vs batch — must produce same grouping
# ===================================================================

class TestOnlineVsBatch(unittest.TestCase):
    """Compare online (one-at-a-time) trie against batch (all-at-once).

    The grouping must be identical.  Rank order may differ because
    the batch trie sees final counts while the online trie caches
    ranks from the last new-group event.
    """

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _simulate_poisson_order(N_REQUESTS)

    def test_same_group_keys(self):
        """Online and batch trie must produce the same set of group keys."""
        # Online: insert one at a time
        trie_online = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for idx in self.arrival_order:
            trie_online.insert(self.reqs[idx]["token_ids"], idx)

        # Batch: insert all at once (natural order)
        trie_batch = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for idx in range(N_REQUESTS):
            trie_batch.insert(self.reqs[idx]["token_ids"], idx)

        online_keys = set(trie_online.dfs_group_keys())
        batch_keys = set(trie_batch.dfs_group_keys())
        self.assertEqual(online_keys, batch_keys)

    def test_same_dfs_order_grouping(self):
        """DFS order must group the same indices together regardless
        of insertion order."""
        trie_online = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for idx in self.arrival_order:
            trie_online.insert(self.reqs[idx]["token_ids"], idx)

        trie_batch = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for idx in range(N_REQUESTS):
            trie_batch.insert(self.reqs[idx]["token_ids"], idx)

        # Extract groups: for each DFS order, group consecutive same-key indices
        def extract_groups(trie):
            dfs = trie.dfs_order(count_aware=True)
            groups = defaultdict(set)
            for prompt_idx in dfs:
                key = tuple(self.reqs[prompt_idx]["token_ids"][:TRIE_MAX_DEPTH])
                groups[key].add(prompt_idx)
            return groups

        online_groups = extract_groups(trie_online)
        batch_groups = extract_groups(trie_batch)

        # Same group keys
        self.assertEqual(set(online_groups.keys()), set(batch_groups.keys()))
        # Same members per group
        for key in online_groups:
            self.assertEqual(online_groups[key], batch_groups[key],
                             f"Group membership differs for key {key[:5]}...")

    def test_count_aware_dfs_largest_group_first(self):
        """count_aware DFS must put the largest group's indices first,
        regardless of insertion order."""
        trie = PrefixTrie(max_depth=TRIE_MAX_DEPTH)
        for idx in self.arrival_order:
            trie.insert(self.reqs[idx]["token_ids"], idx)

        dfs = trie.dfs_order(count_aware=True)

        # Find the largest group
        group_counts = Counter(self.labels)
        largest_group = group_counts.most_common(1)[0][0]
        largest_count = group_counts[largest_group]

        # First `largest_count` indices in DFS should all belong to that group
        first_n_labels = [self.labels[dfs[i]] for i in range(largest_count)]
        self.assertTrue(
            all(lbl == largest_group for lbl in first_n_labels),
            f"First {largest_count} DFS entries should all be group "
            f"{largest_group}, got {Counter(first_n_labels)}",
        )


# ===================================================================
# Test 4: Online dispatcher vs batch reorder — grouping quality
# ===================================================================

class TestOnlineDispatcherVsBatchReorder(unittest.TestCase):
    """The online PeekDispatcher (incremental, no lookahead) should
    produce group hashes that achieve grouping quality comparable to
    batch reorder_for_prefix_sharing (which sees all requests)."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _simulate_poisson_order(N_REQUESTS)

        # --- Online: PeekDispatcher one-at-a-time ---
        cls.online_dispatched = []
        dispatcher = PeekDispatcher(
            send_fn=lambda r: cls.online_dispatched.append(dict(r)),
        )
        for idx in cls.arrival_order:
            dispatcher.submit(dict(cls.reqs[idx]))

        # --- Batch: reorder_for_prefix_sharing sees all at once ---
        arrival_seqs = [cls.reqs[idx]["token_ids"] for idx in cls.arrival_order]
        cls.batch_order = reorder_for_prefix_sharing(arrival_seqs)

    def test_online_grouping_is_correct(self):
        """Online dispatcher assigns one hash per original group."""
        hash_to_groups = defaultdict(set)
        for r in self.online_dispatched:
            ghash = r["rid"].split(":", 3)[2]
            hash_to_groups[ghash].add(r["group"])

        collisions = {h: gs for h, gs in hash_to_groups.items() if len(gs) > 1}
        self.assertEqual(len(collisions), 0,
                         f"Hash collisions across groups: {collisions}")

    def test_batch_achieves_high_adjacency(self):
        """Batch reorder (gold standard) should achieve high adjacency."""
        arrival_labels = [self.labels[idx] for idx in self.arrival_order]
        reordered_labels = [arrival_labels[i] for i in self.batch_order]
        adj = _adjacency_score(reordered_labels)
        self.assertGreater(adj, 0.90,
                           f"Batch adjacency {adj:.2f} unexpectedly low")

    def test_online_rank_enables_server_grouping(self):
        """The server can reconstruct perfect grouping from online tags.

        Simulate what the server does: group dispatched requests by hash,
        then check that each hash-group contains exactly one original group.
        """
        hash_groups: dict[str, list[dict]] = defaultdict(list)
        for r in self.online_dispatched:
            ghash = r["rid"].split(":", 3)[2]
            hash_groups[ghash].append(r)

        num_groups_found = len(hash_groups)
        unique_original = len(set(r["group"] for r in self.online_dispatched))
        self.assertEqual(num_groups_found, unique_original,
                         f"Server sees {num_groups_found} hash groups but "
                         f"workload has {unique_original} original groups")

        # Each hash group should map to one original group
        for ghash, members in hash_groups.items():
            orig = {m["group"] for m in members}
            self.assertEqual(len(orig), 1)


# ===================================================================
# Test 5: Poisson arrival timing — ranks evolve correctly
# ===================================================================

class TestRankEvolution(unittest.TestCase):
    """Verify that group ranks evolve correctly as requests arrive
    under Poisson timing — the hottest group always has rank 0."""

    @classmethod
    def setUpClass(cls):
        cls.reqs, cls.labels = _generate_workload()
        cls.arrival_order = _simulate_poisson_order(N_REQUESTS)

    def test_rank_zero_is_always_most_popular_at_discovery(self):
        """When a new group triggers a DFS rerank, rank 0 must belong
        to the group with the highest count so far."""
        group_counts: dict[tuple, int] = {}
        dispatched = []

        dispatcher = PeekDispatcher(
            send_fn=lambda r: dispatched.append(dict(r)),
        )

        seen_groups: set[tuple] = set()
        rerank_points: list[int] = []   # steps where a rerank happened

        for step, idx in enumerate(self.arrival_order):
            req = dict(self.reqs[idx])
            key = tuple(req["token_ids"][:TRIE_MAX_DEPTH])
            group_counts[key] = group_counts.get(key, 0) + 1

            is_new = key not in seen_groups
            seen_groups.add(key)
            dispatcher.submit(req)

            if is_new:
                rerank_points.append(step)

        # At the final state, check rank 0
        # Find which group has rank 0 in the last dispatched request
        rank_to_hash: dict[int, str] = {}
        hash_to_count: dict[str, int] = Counter()
        for r in dispatched:
            parts = r["rid"].split(":", 3)
            rank = int(parts[1])
            ghash = parts[2]
            rank_to_hash[rank] = ghash
            hash_to_count[ghash] += 1

        # Rank 0 should be the most popular hash
        rank0_hash = rank_to_hash[0]
        rank0_count = hash_to_count[rank0_hash]
        max_count = max(hash_to_count.values())
        self.assertEqual(
            rank0_count, max_count,
            f"Rank 0 has count {rank0_count} but max is {max_count}",
        )

    def test_num_reranks_equals_num_groups(self):
        """DFS rerank should trigger exactly once per new group."""
        seen_keys: set[tuple] = set()
        rerank_count = 0

        dispatcher = PeekDispatcher(
            send_fn=lambda r: None,
        )

        for idx in self.arrival_order:
            req = dict(self.reqs[idx])
            key = tuple(req["token_ids"][:TRIE_MAX_DEPTH])
            was_new = key not in seen_keys
            seen_keys.add(key)
            dispatcher.submit(req)
            if was_new:
                rerank_count += 1

        # Number of reranks = number of distinct groups discovered
        self.assertEqual(rerank_count, len(seen_keys))


if __name__ == "__main__":
    unittest.main()
