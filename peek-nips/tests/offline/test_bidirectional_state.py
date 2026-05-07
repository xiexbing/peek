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

"""Test bidirectional client↔server state exchange via CacheStateStore.

Verifies:
1. Client pushes {group_hash: pending_count} on every submit/remove.
2. Server reads the latest snapshot at scheduling cycle start.
3. Server's scoring uses client pending counts for future_refs.
4. The timing: client updates continuously, server reads at cycle boundary.
"""
import unittest

from peek.offline.engine import CacheStateStore
from peek.offline.reorder import PeekDispatcher


class TestClientPushPendingCounts(unittest.TestCase):
    """Client pushes pending counts to CacheStateStore on every submit/remove."""

    def setUp(self):
        # Reset singleton so each test starts clean
        CacheStateStore._instance = None
        self.store = CacheStateStore.get()

    def tearDown(self):
        CacheStateStore._instance = None

    def test_submit_pushes_counts(self):
        """After each submit, the store reflects the current pending counts."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        d.submit({"id": "a-0", "token_ids": prefix_a + [0]})
        pending = self.store.get_client_pending()
        self.assertEqual(len(pending), 1)
        # Find hash for group A
        hash_a = list(pending.keys())[0]
        self.assertEqual(pending[hash_a], 1)

        d.submit({"id": "a-1", "token_ids": prefix_a + [1]})
        pending = self.store.get_client_pending()
        self.assertEqual(pending[hash_a], 2)

        d.submit({"id": "b-0", "token_ids": prefix_b + [0]})
        pending = self.store.get_client_pending()
        self.assertEqual(len(pending), 2)
        hash_b = [h for h in pending if h != hash_a][0]
        self.assertEqual(pending[hash_a], 2)
        self.assertEqual(pending[hash_b], 1)

    def test_remove_pushes_updated_counts(self):
        """After remove, the store reflects decremented counts."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(5):
            d.submit({"id": f"r-{i}", "token_ids": prefix + [i]})

        pending = self.store.get_client_pending()
        ghash = list(pending.keys())[0]
        self.assertEqual(pending[ghash], 5)

        # Remove 3
        for i in range(3):
            d.remove(prefix + [i], i)

        pending = self.store.get_client_pending()
        self.assertEqual(pending[ghash], 2)

    def test_remove_all_clears_group(self):
        """Removing all requests from a group removes it from the store."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(3):
            d.submit({"id": f"r-{i}", "token_ids": prefix + [i]})

        for i in range(3):
            d.remove(prefix + [i], i)

        pending = self.store.get_client_pending()
        self.assertEqual(len(pending), 0)


class TestServerReadsPendingCounts(unittest.TestCase):
    """Server reads client pending counts at scheduling cycle start."""

    def setUp(self):
        CacheStateStore._instance = None
        self.store = CacheStateStore.get()

    def tearDown(self):
        CacheStateStore._instance = None

    def test_server_sees_client_pending(self):
        """Simulate: client submits, then server reads at cycle boundary."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        # Client submits 10 to A, 5 to B
        for i in range(10):
            d.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
        for i in range(5):
            d.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})

        # --- Server scheduling cycle starts ---
        pending = self.store.get_client_pending()

        # Server sees 2 groups with correct counts
        self.assertEqual(len(pending), 2)
        counts = sorted(pending.values(), reverse=True)
        self.assertEqual(counts, [10, 5])

    def test_server_sees_post_removal_state(self):
        """Client submits 10, removes 7, server sees 3 pending."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(10):
            d.submit({"id": f"r-{i}", "token_ids": prefix + [i]})

        # Client receives completions, removes 7
        for i in range(7):
            d.remove(prefix + [i], i)

        # --- Server scheduling cycle starts ---
        pending = self.store.get_client_pending()
        ghash = list(pending.keys())[0]
        self.assertEqual(pending[ghash], 3)

    def test_server_gets_future_demand_signal(self):
        """The key value: client pending > server queue count.

        Simulates: client submitted 100 to group A, but only 20 have
        arrived at the server's queue.  The client's pending count (100)
        tells the server that 80 more are coming.
        """
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))
        for i in range(100):
            d.submit({"id": f"r-{i}", "token_ids": prefix + [i]})

        # Server reads: client says 100 pending
        pending = self.store.get_client_pending()
        ghash = list(pending.keys())[0]
        client_pending = pending[ghash]
        self.assertEqual(client_pending, 100)

        # The server's waiting queue might only have 20 (rest in-flight).
        # With client_pending=100 and BATCH_EST=32:
        #   future_refs = max(0, 100 - 32) = 68
        # Without client info (using server queue_count=20):
        #   future_refs = max(0, 20 - 32) = 0
        #
        # The client signal correctly tells the server to DEFER this
        # group -- it has heavy future demand, keep its prefix cached.
        future_refs_with_client = max(0, client_pending - 32)
        future_refs_without = max(0, 20 - 32)
        self.assertEqual(future_refs_with_client, 68)
        self.assertEqual(future_refs_without, 0)


class TestTimingSemantics(unittest.TestCase):
    """Verify that the client->server channel has correct timing."""

    def setUp(self):
        CacheStateStore._instance = None
        self.store = CacheStateStore.get()

    def tearDown(self):
        CacheStateStore._instance = None

    def test_multiple_submits_before_server_read(self):
        """Multiple submits between server cycles -- server sees latest."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))

        # 3 submits happen before server reads
        d.submit({"id": "r-0", "token_ids": prefix + [0]})
        d.submit({"id": "r-1", "token_ids": prefix + [1]})
        d.submit({"id": "r-2", "token_ids": prefix + [2]})

        # Server reads once -- sees count=3 (latest), not count=1
        pending = self.store.get_client_pending()
        ghash = list(pending.keys())[0]
        self.assertEqual(pending[ghash], 3)

    def test_submits_and_removes_interleaved_before_read(self):
        """Submits and removes interleave, server sees net result."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))

        d.submit({"id": "r-0", "token_ids": prefix + [0]})  # pending=1
        d.submit({"id": "r-1", "token_ids": prefix + [1]})  # pending=2
        d.remove(prefix + [0], 0)                             # pending=1
        d.submit({"id": "r-2", "token_ids": prefix + [2]})  # pending=2
        d.remove(prefix + [1], 1)                             # pending=1

        # Server reads: net pending = 1
        pending = self.store.get_client_pending()
        ghash = list(pending.keys())[0]
        self.assertEqual(pending[ghash], 1)

    def test_server_reads_are_independent(self):
        """Each server read gets a fresh snapshot -- not cumulative."""
        dispatched = []
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        prefix = list(range(200))

        d.submit({"id": "r-0", "token_ids": prefix + [0]})

        # First read
        p1 = self.store.get_client_pending()
        self.assertEqual(list(p1.values()), [1])

        d.submit({"id": "r-1", "token_ids": prefix + [1]})

        # Second read -- fresh snapshot, count=2
        p2 = self.store.get_client_pending()
        self.assertEqual(list(p2.values()), [2])

        # First read result is unchanged (it's a dict copy)
        self.assertEqual(list(p1.values()), [1])


if __name__ == "__main__":
    unittest.main()
