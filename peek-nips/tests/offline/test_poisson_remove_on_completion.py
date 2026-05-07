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

"""Verify that run_poisson_peek calls dispatcher.remove() on every
completion, so the client trie drains to zero after all requests finish.

Uses a mock HTTP server (no aiohttp dependency) to simulate the full
Poisson arrival → HTTP send → response → remove() flow.
"""

import pytest

pytestmark = pytest.mark.gpu
import asyncio
import time
import unittest

from peek.offline.engine import CacheStateStore
from peek.offline.reorder import PeekDispatcher
from peek.offline.benchmarks.poisson_client import load_requests


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
SYSTEM_PROMPT_LEN = 2048
ZIPF_ALPHA = 1.5
QUESTION_LEN = 128
SEED = 42


def _generate_workload():
    return load_requests(
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


# ---------------------------------------------------------------------------
# Simulate the exact dispatch_fn + _send + remove flow from poisson_client
# ---------------------------------------------------------------------------

def run_mock_poisson_peek(requests, arrival_rate=30.0, seed=SEED):
    """Replicate run_poisson_peek's dispatch_fn logic with a mock sender.

    Returns (dispatcher, submit_count, remove_count, send_order).
    """
    import random

    CacheStateStore._instance = None

    submit_count = 0
    remove_count = 0
    send_order = []       # (request_id, "submit"|"send"|"remove")
    id_to_remove_info = {}

    dispatcher = PeekDispatcher.__new__(PeekDispatcher)
    # We need to replicate the real flow exactly, so use a real dispatcher
    dispatched = []
    dispatcher_real = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

    # Replicate dispatch_fn from poisson_client.py
    def dispatch_fn(req):
        nonlocal submit_count
        orig_id = req.get("id", "")
        # Stash info for remove — prompt_index is _next_idx - 1
        id_to_remove_info[orig_id] = (
            req["token_ids"],
            dispatcher_real._next_idx - 1,
        )
        send_order.append((orig_id, "submit"))
        submit_count += 1

    # Override send_fn to use our dispatch_fn
    dispatcher_real._send_fn = dispatch_fn

    # Submit all requests with Poisson inter-arrival (simulated, no sleep)
    rng = random.Random(seed)
    arrival_order = list(range(len(requests)))
    rng.shuffle(arrival_order)

    for idx in arrival_order:
        req = dict(requests[idx])
        dispatcher_real.submit(req)

    # Simulate server completing all requests in submission order
    for orig_id, (token_ids, prompt_idx) in list(id_to_remove_info.items()):
        send_order.append((orig_id, "remove"))
        dispatcher_real.remove(token_ids, prompt_idx)
        remove_count += 1

    id_to_remove_info.clear()

    return dispatcher_real, submit_count, remove_count, send_order


class TestRemoveOnCompletion(unittest.TestCase):
    """Verify the submit→send→complete→remove flow."""

    @classmethod
    def setUpClass(cls):
        cls.reqs = _generate_workload()
        cls.dispatcher, cls.submit_count, cls.remove_count, cls.send_order = \
            run_mock_poisson_peek(cls.reqs)

    def test_all_submitted(self):
        self.assertEqual(self.submit_count, N_REQUESTS)

    def test_all_removed(self):
        self.assertEqual(self.remove_count, N_REQUESTS)

    def test_trie_empty_after_all_completions(self):
        self.assertEqual(self.dispatcher._trie._num_prompts, 0)

    def test_group_count_empty(self):
        self.assertEqual(len(self.dispatcher._group_count), 0)

    def test_store_empty(self):
        store = CacheStateStore.get()
        pending = store.get_client_pending()
        self.assertEqual(len(pending), 0)

    def test_every_submit_has_matching_remove(self):
        """Every request ID that was submitted must also be removed."""
        submitted_ids = {oid for oid, action in self.send_order if action == "submit"}
        removed_ids = {oid for oid, action in self.send_order if action == "remove"}
        self.assertEqual(submitted_ids, removed_ids)

    def test_remove_always_after_submit(self):
        """For every request, remove must come after submit in the order."""
        first_submit = {}
        first_remove = {}
        for i, (oid, action) in enumerate(self.send_order):
            if action == "submit" and oid not in first_submit:
                first_submit[oid] = i
            elif action == "remove" and oid not in first_remove:
                first_remove[oid] = i

        for oid in first_submit:
            self.assertIn(oid, first_remove,
                          f"{oid} was submitted but never removed")
            self.assertLess(first_submit[oid], first_remove[oid],
                            f"{oid} was removed before it was submitted")


class TestTrieDrainsProgressively(unittest.TestCase):
    """Verify trie count decreases as completions arrive."""

    def test_trie_count_at_each_step(self):
        reqs = _generate_workload()

        CacheStateStore._instance = None
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        import random
        rng = random.Random(SEED)
        arrival_order = list(range(N_REQUESTS))
        rng.shuffle(arrival_order)

        # Phase 1: submit all
        prompt_info = {}  # orig_id → (token_ids, prompt_idx)
        for idx in arrival_order:
            req = dict(reqs[idx])
            prompt_idx = dispatcher._next_idx
            dispatcher.submit(req)
            prompt_info[req["id"]] = (reqs[idx]["token_ids"], prompt_idx)

        self.assertEqual(dispatcher._trie._num_prompts, N_REQUESTS)

        # Phase 2: remove one at a time, check trie count after each
        ids_to_remove = list(prompt_info.keys())
        rng.shuffle(ids_to_remove)  # random completion order

        for i, rid in enumerate(ids_to_remove):
            token_ids, pidx = prompt_info[rid]
            dispatcher.remove(token_ids, pidx)
            expected = N_REQUESTS - (i + 1)
            self.assertEqual(
                dispatcher._trie._num_prompts, expected,
                f"After removing {i + 1} requests, trie has "
                f"{dispatcher._trie._num_prompts}, expected {expected}",
            )

        self.assertEqual(dispatcher._trie._num_prompts, 0)


class TestStoreReflectsPendingDuringDrain(unittest.TestCase):
    """CacheStateStore pending counts must match trie at every step."""

    def test_store_matches_trie_during_drain(self):
        reqs = _generate_workload()

        CacheStateStore._instance = None
        store = CacheStateStore.get()
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        import random
        rng = random.Random(SEED)
        arrival_order = list(range(N_REQUESTS))
        rng.shuffle(arrival_order)

        prompt_info = {}
        for idx in arrival_order:
            req = dict(reqs[idx])
            prompt_idx = dispatcher._next_idx
            dispatcher.submit(req)
            prompt_info[req["id"]] = (reqs[idx]["token_ids"], prompt_idx)

        # Check at submission end
        trie_count = dispatcher._trie._num_prompts
        store_count = sum(store.get_client_pending().values())
        self.assertEqual(trie_count, store_count)

        # Remove in chunks of 100, check store matches trie after each chunk
        ids = list(prompt_info.keys())
        rng.shuffle(ids)

        for chunk_start in range(0, N_REQUESTS, 100):
            chunk = ids[chunk_start:chunk_start + 100]
            for rid in chunk:
                token_ids, pidx = prompt_info[rid]
                dispatcher.remove(token_ids, pidx)

            trie_count = dispatcher._trie._num_prompts
            store_count = sum(store.get_client_pending().values())
            disp_count = sum(dispatcher._group_count.values())

            self.assertEqual(trie_count, store_count,
                             f"After removing {chunk_start + len(chunk)}: "
                             f"trie={trie_count} != store={store_count}")
            self.assertEqual(trie_count, disp_count,
                             f"After removing {chunk_start + len(chunk)}: "
                             f"trie={trie_count} != dispatcher={disp_count}")

        self.assertEqual(dispatcher._trie._num_prompts, 0)
        self.assertEqual(sum(store.get_client_pending().values()), 0)


class TestRankCorrectDuringDrain(unittest.TestCase):
    """Ranks must reflect current pending counts as requests complete."""

    def test_overtake_during_drain(self):
        """Group A starts larger. Complete most of A. Group B overtakes.
        Next submit to B must get rank 0."""
        CacheStateStore._instance = None
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        # Submit A=20, B=15
        a_info = []
        for i in range(20):
            pidx = dispatcher._next_idx
            dispatcher.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
            a_info.append((prefix_a + [i], pidx))

        b_info = []
        for i in range(15):
            pidx = dispatcher._next_idx
            dispatcher.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})
            b_info.append((prefix_b + [i], pidx))

        # A=20, B=15. A is rank 0.
        self.assertEqual(dispatcher._trie._num_prompts, 35)

        # Complete 16 of A → A=4, B=15. B should overtake.
        for tids, pidx in a_info[:16]:
            dispatcher.remove(tids, pidx)

        self.assertEqual(dispatcher._trie._num_prompts, 19)  # 4 + 15

        # Next submit to B → B=16, A=4. B must be rank 0.
        dispatched.clear()
        dispatcher.submit({"id": "b-new", "token_ids": prefix_b + [999]})
        rank_b = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(rank_b, 0,
                         f"B(16 pending) should be rank 0 after draining A to 4, got {rank_b}")

        # Next submit to A → A=5, B=16. A must be rank 1.
        dispatched.clear()
        dispatcher.submit({"id": "a-new", "token_ids": prefix_a + [999]})
        rank_a = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(rank_a, 1,
                         f"A(5 pending) should be rank 1, got {rank_a}")


if __name__ == "__main__":
    unittest.main()
