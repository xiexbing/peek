#!/usr/bin/env python3
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

"""Test that PeekEngine server-side correctly receives and processes
client tags from PeekDispatcher.

Simulates the full path:
  1. Client PeekDispatcher tags requests with peek:<rank>:<hash>:<rid>
  2. Server PeekEngine._run_tagged parses tags and groups requests
  3. Verify: grouping is correct, no requests are dropped, rank is parsed

Since PeekEngine depends on SGLang's radix cache (not available in unit
tests), we test the parsing and grouping logic directly.
"""
import unittest
from collections import defaultdict
from types import SimpleNamespace

from peek.offline.reorder import PeekDispatcher
from peek.offline.engine import PeekEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tagged_requests(num_groups, per_group, prefix_len=2048, max_depth=128):
    """Generate requests through PeekDispatcher, return as SimpleNamespace
    objects (mimicking SGLang Req objects) with 'rid' and 'origin_input_ids'."""
    import random

    dispatched = []
    dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

    rng = random.Random(42)
    group_prefixes = {}
    for g in range(num_groups):
        grng = random.Random(g * 1000 + 7)
        group_prefixes[g] = [grng.randint(1, 31999) for _ in range(prefix_len)]

    for g in range(num_groups):
        for i in range(per_group):
            suffix = [rng.randint(1, 31999) for _ in range(128)]
            tids = group_prefixes[g] + suffix
            dispatcher.submit({"id": f"g{g}-r{i}", "token_ids": tids, "group": f"grp-{g}"})

    # Convert to SGLang-like request objects
    reqs = []
    for d in dispatched:
        req = SimpleNamespace(
            rid=d["rid"],
            origin_input_ids=d["token_ids"],
            output_ids=[],
            extra_key=None,
            prefix_indices=[],
            last_node=None,
            last_host_node=None,
            host_hit_length=0,
            group=d["group"],
        )
        reqs.append(req)
    return reqs, dispatcher


# ===================================================================
# Test: Tag detection
# ===================================================================

class TestHasPeekTags(unittest.TestCase):

    def test_tagged_queue_detected(self):
        reqs, _ = _make_tagged_requests(3, 5)
        self.assertTrue(PeekEngine._has_peek_tags(reqs))

    def test_empty_queue(self):
        self.assertFalse(PeekEngine._has_peek_tags([]))

    def test_untagged_queue(self):
        reqs = [SimpleNamespace(rid=f"req-{i}") for i in range(5)]
        self.assertFalse(PeekEngine._has_peek_tags(reqs))

    def test_mixed_queue_first_untagged(self):
        """BUG: if the first request is untagged, the whole queue
        falls back to _run_full even if most requests have tags."""
        tagged, _ = _make_tagged_requests(2, 3)
        untagged = SimpleNamespace(rid="plain-req")
        mixed = [untagged] + list(tagged)

        # Current behavior: checks only first element
        result = PeekEngine._has_peek_tags(mixed)
        # This is False because first element is untagged -- KNOWN ISSUE
        self.assertFalse(result)


# ===================================================================
# Test: Tag parsing
# ===================================================================

class TestParseRid(unittest.TestCase):

    def test_valid_tag(self):
        rank, group_key, orig = PeekEngine._parse_peek_rid("peek:3:12345678:my-req-42")
        self.assertEqual(rank, 3)
        self.assertEqual(group_key, 12345678)
        self.assertEqual(orig, "my-req-42")

    def test_rank_zero(self):
        rank, group_key, orig = PeekEngine._parse_peek_rid("peek:0:99999:req-0")
        self.assertEqual(rank, 0)

    def test_untagged_rid(self):
        rank, group_key, orig = PeekEngine._parse_peek_rid("plain-request-id")
        self.assertEqual(rank, 0)
        self.assertEqual(group_key, 0)
        self.assertEqual(orig, "plain-request-id")

    def test_original_rid_with_colons(self):
        """Original rid may contain colons (e.g. UUIDs)."""
        rank, group_key, orig = PeekEngine._parse_peek_rid("peek:5:111:abc:def:ghi")
        self.assertEqual(rank, 5)
        self.assertEqual(group_key, 111)
        self.assertEqual(orig, "abc:def:ghi")


# ===================================================================
# Test: Grouping logic
# ===================================================================

class TestTagGrouping(unittest.TestCase):
    """Verify that _run_tagged groups requests correctly by tag hash."""

    def test_grouping_matches_client(self):
        """Server groups must match client groups: same hash = same group."""
        reqs, _ = _make_tagged_requests(5, 10)

        # Simulate Step 2 of _run_tagged: group by tag hash
        tag_groups: dict[int, list] = {}
        for r in reqs:
            rid = r.rid
            _, group_key, _ = PeekEngine._parse_peek_rid(rid)
            tag_groups.setdefault(group_key, []).append(r)

        # Should have exactly 5 groups
        self.assertEqual(len(tag_groups), 5)

        # Each group should have 10 members
        for gk, members in tag_groups.items():
            self.assertEqual(len(members), 10,
                             f"Group {gk} has {len(members)} members, expected 10")

        # Each group's members should all be from the same original group
        for gk, members in tag_groups.items():
            orig_groups = {m.group for m in members}
            self.assertEqual(len(orig_groups), 1,
                             f"Hash group {gk} contains mixed original groups: {orig_groups}")

    def test_no_cross_group_collision(self):
        """Different original groups must not map to the same hash."""
        reqs, _ = _make_tagged_requests(100, 5)

        hash_to_orig: dict[int, set] = defaultdict(set)
        for r in reqs:
            _, group_key, _ = PeekEngine._parse_peek_rid(r.rid)
            hash_to_orig[group_key].add(r.group)

        for gk, origs in hash_to_orig.items():
            self.assertEqual(len(origs), 1,
                             f"Hash {gk} maps to multiple groups: {origs}")

    def test_all_requests_grouped(self):
        """No request should be silently dropped during grouping."""
        reqs, _ = _make_tagged_requests(10, 20)

        tag_groups: dict[int, list] = {}
        for r in reqs:
            rid = r.rid
            if isinstance(rid, str) and rid.startswith("peek:"):
                _, group_key, _ = PeekEngine._parse_peek_rid(rid)
                tag_groups.setdefault(group_key, []).append(r)

        total_grouped = sum(len(m) for m in tag_groups.values())
        self.assertEqual(total_grouped, 200,
                         f"Only {total_grouped}/200 requests grouped -- "
                         f"{200 - total_grouped} silently dropped")

    def test_untagged_requests_would_be_dropped(self):
        """KNOWN ISSUE: untagged requests in a tagged queue get dropped
        from the grouped output."""
        tagged, _ = _make_tagged_requests(2, 5)

        # Inject 3 untagged requests
        untagged = [
            SimpleNamespace(rid=f"plain-{i}", origin_input_ids=list(range(100)),
                            group="untagged")
            for i in range(3)
        ]
        mixed = list(tagged) + untagged  # 13 total

        # Simulate grouping (same logic as _run_tagged step 2)
        tag_groups: dict[int, list] = {}
        for r in mixed:
            rid = getattr(r, "rid", "")
            if isinstance(rid, str) and rid.startswith("peek:"):
                _, group_key, _ = PeekEngine._parse_peek_rid(rid)
                tag_groups.setdefault(group_key, []).append(r)

        total_grouped = sum(len(m) for m in tag_groups.values())
        # Only 10 tagged requests are grouped; 3 untagged are DROPPED
        self.assertEqual(total_grouped, 10)
        self.assertEqual(len(mixed), 13)
        # This documents the known issue: 3 requests would vanish


# ===================================================================
# Test: Client rank is parsed but not used in scoring
# ===================================================================

class TestClientRankUsage(unittest.TestCase):
    """The server parses the client rank but currently discards it."""

    def test_rank_is_parsed(self):
        """_parse_peek_rid extracts the rank correctly."""
        reqs, _ = _make_tagged_requests(3, 5)

        ranks_seen = set()
        for r in reqs:
            rank, _, _ = PeekEngine._parse_peek_rid(r.rid)
            ranks_seen.add(rank)

        # With 3 groups, ranks should be {0, 1, 2}
        self.assertEqual(ranks_seen, {0, 1, 2})

    def test_rank_discarded_in_grouping(self):
        """_run_tagged uses _ for rank -- it's discarded during grouping."""
        reqs, _ = _make_tagged_requests(3, 5)

        # Simulate the exact line from _run_tagged:
        #   _, group_key, _ = self._parse_peek_rid(rid)
        # The first _ is rank, the last _ is original_rid -- both discarded
        for r in reqs:
            result = PeekEngine._parse_peek_rid(r.rid)
            self.assertEqual(len(result), 3)
            rank, group_key, orig_rid = result
            # rank is an int -- it's parseable but unused
            self.assertIsInstance(rank, int)

    def test_largest_group_gets_rank_zero_from_client(self):
        """PeekDispatcher assigns rank 0 to the group with most pending."""
        reqs, _ = _make_tagged_requests(3, per_group=0)
        # 3 groups with different sizes: 20, 10, 5
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        import random
        rng = random.Random(42)
        prefix_a = [rng.randint(1, 31999) for _ in range(200)]
        prefix_b = [rng.randint(1, 31999) for _ in range(200)]
        prefix_c = [rng.randint(1, 31999) for _ in range(200)]

        for i in range(20):
            dispatcher.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
        for i in range(10):
            dispatcher.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})
        for i in range(5):
            dispatcher.submit({"id": f"c-{i}", "token_ids": prefix_c + [i]})

        # Check last request from each group
        last_a = [d for d in dispatched if d["id"].startswith("a-")][-1]
        last_b = [d for d in dispatched if d["id"].startswith("b-")][-1]
        last_c = [d for d in dispatched if d["id"].startswith("c-")][-1]

        rank_a = int(last_a["rid"].split(":")[1])
        rank_b = int(last_b["rid"].split(":")[1])
        rank_c = int(last_c["rid"].split(":")[1])

        self.assertEqual(rank_a, 0, f"Group A (20 reqs) should be rank 0, got {rank_a}")
        self.assertEqual(rank_b, 1, f"Group B (10 reqs) should be rank 1, got {rank_b}")
        self.assertEqual(rank_c, 2, f"Group C (5 reqs) should be rank 2, got {rank_c}")


if __name__ == "__main__":
    unittest.main()
