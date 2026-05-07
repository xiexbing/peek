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

"""Test the full client → server → client feedback loop.

Simulates Qwen 2.5 32B online serving: 100 groups, 1000 requests,
Poisson arrivals.  The loop:

  1. Client PeekDispatcher.submit() — inserts into trie, tags, sends.
  2. Server receives tagged request, schedules it.
  3. Server completes request, sends completion back.
  4. Client PeekDispatcher.remove() — removes from trie, invalidates
     cached ranks so the next submit() sees fresh group counts.

Key invariant: at every point, the trie reflects ONLY in-flight
(pending) requests — not completed ones and not future ones.
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import Counter, defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.reorder import PeekDispatcher
from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Workload parameters (Qwen 2.5 32B on H100)
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
TRIE_MAX_DEPTH = 128
SEED = 42

# Server simulation parameters
BATCH_SIZE = 32         # server processes up to 32 requests per cycle
PREFILL_CYCLES = 2      # request takes ~2 cycles to complete


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
        group_str = r.get("group", "sys-0")
        try:
            labels.append(int(group_str.split("-")[1]))
        except (IndexError, ValueError):
            labels.append(-1)
    return reqs, labels


def _poisson_arrival_times(n: int, rate: float, seed: int = SEED) -> list[float]:
    """Return absolute arrival times for n requests under Poisson(rate)."""
    rng = random.Random(seed)
    times = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        times.append(t)
    return times


# ---------------------------------------------------------------------------
# Simulated server
# ---------------------------------------------------------------------------

class SimulatedServer:
    """Minimal server that receives tagged requests, batches them,
    and completes them after a fixed number of cycles.

    On each cycle:
      - Accept up to BATCH_SIZE from the waiting queue.
      - Decrement TTL on running requests.
      - Complete requests whose TTL reaches 0 → return to client.
    """

    def __init__(self, batch_size: int = BATCH_SIZE, prefill_cycles: int = PREFILL_CYCLES):
        self.batch_size = batch_size
        self.prefill_cycles = prefill_cycles
        self.waiting: list[dict] = []
        self.running: list[tuple[dict, int]] = []  # (request, remaining_cycles)

    def receive(self, request: dict) -> None:
        """Server receives a tagged request from the client."""
        self.waiting.append(request)

    def step(self) -> list[dict]:
        """Run one scheduling cycle.  Returns list of completed requests."""
        # Admit from waiting → running
        to_admit = min(self.batch_size, len(self.waiting))
        for _ in range(to_admit):
            req = self.waiting.pop(0)
            self.running.append((req, self.prefill_cycles))

        # Tick running requests
        still_running = []
        completed = []
        for req, ttl in self.running:
            if ttl <= 1:
                completed.append(req)
            else:
                still_running.append((req, ttl - 1))
        self.running = still_running

        return completed

    @property
    def pending_count(self) -> int:
        return len(self.waiting) + len(self.running)


# ---------------------------------------------------------------------------
# Full feedback loop simulation
# ---------------------------------------------------------------------------

def run_feedback_simulation():
    """Run the full client→server→client loop.

    Returns a list of snapshots, one per simulation cycle:
        {
            'cycle': int,
            'trie_count': int,            # prompts in client trie
            'pending_count': int,          # requests in-flight (waiting+running)
            'arrivals_this_cycle': int,
            'completions_this_cycle': int,
            'group_counts': dict,          # per-group pending count in trie
            'dispatched': list[dict],      # requests dispatched this cycle
            'completed': list[dict],       # requests completed this cycle
        }
    """
    reqs, labels = _generate_workload()
    arrival_times = _poisson_arrival_times(N_REQUESTS, rate=30.0)

    # Shuffle arrival order (Poisson interleaves groups)
    rng = random.Random(SEED)
    arrival_order = list(range(N_REQUESTS))
    rng.shuffle(arrival_order)

    server = SimulatedServer()
    dispatched_all = []

    def on_dispatch(req):
        dispatched_all.append(req)
        server.receive(req)

    dispatcher = PeekDispatcher(send_fn=on_dispatch)

    # Track which prompt_index each request got (for remove)
    rid_to_prompt_idx: dict[str, int] = {}
    rid_to_token_ids: dict[str, list[int]] = {}

    # Time simulation
    CYCLE_DURATION = 0.033  # ~30ms per cycle (matches 30 req/s)
    cycle_time = 0.0
    next_arrival = 0
    snapshots = []

    for cycle in range(1200):  # 1000 req / 30 req/s ≈ 33s / 0.033s ≈ 1000 cycles + margin
        cycle_time = cycle * CYCLE_DURATION
        cycle_dispatched = []
        cycle_completed = []

        # 1. Arrivals: submit requests whose arrival_time <= cycle_time
        while next_arrival < N_REQUESTS and arrival_times[next_arrival] <= cycle_time:
            idx = arrival_order[next_arrival]
            req = dict(reqs[idx])
            prompt_index = dispatcher._next_idx  # peek at what index will be assigned

            dispatched_before = len(dispatched_all)
            dispatcher.submit(req)
            dispatched_after = len(dispatched_all)

            if dispatched_after > dispatched_before:
                sent_req = dispatched_all[-1]
                rid_to_prompt_idx[sent_req["id"]] = prompt_index
                rid_to_token_ids[sent_req["id"]] = reqs[idx]["token_ids"]
                cycle_dispatched.append(sent_req)

            next_arrival += 1

        # 2. Server step: schedule + complete
        completed = server.step()
        cycle_completed = completed

        # 3. Client feedback: remove completed from trie
        for comp_req in completed:
            req_id = comp_req["id"]
            if req_id in rid_to_prompt_idx:
                prompt_idx = rid_to_prompt_idx.pop(req_id)
                token_ids = rid_to_token_ids.pop(req_id)
                dispatcher.remove(token_ids, prompt_idx)

        # 4. Snapshot
        # Count per-group pending in the trie
        group_counts: dict[str, int] = Counter()
        for req_id in rid_to_prompt_idx:
            # Find original group from dispatched_all
            for d in dispatched_all:
                if d["id"] == req_id:
                    group_counts[d.get("group", "")] += 1
                    break

        snapshots.append({
            "cycle": cycle,
            "trie_count": dispatcher._trie._num_prompts,
            "pending_count": len(rid_to_prompt_idx),
            "server_pending": server.pending_count,
            "arrivals_this_cycle": len(cycle_dispatched),
            "completions_this_cycle": len(cycle_completed),
            "group_counts": dict(group_counts),
            "n_groups_in_trie": len(dispatcher._trie.dfs_group_keys()),
        })

        # Early exit if everything is done
        if next_arrival >= N_REQUESTS and server.pending_count == 0 and dispatcher._trie._num_prompts == 0:
            break

    return snapshots, dispatcher


# ===================================================================
# Tests
# ===================================================================

class TestFeedbackLoop(unittest.TestCase):
    """Full client→server→client feedback loop."""

    @classmethod
    def setUpClass(cls):
        cls.snapshots, cls.dispatcher = run_feedback_simulation()

    def test_trie_equals_pending_at_every_cycle(self):
        """The trie prompt count must equal the number of in-flight
        requests at every simulation cycle — never more, never less."""
        for s in self.snapshots:
            self.assertEqual(
                s["trie_count"], s["pending_count"],
                f"Cycle {s['cycle']}: trie has {s['trie_count']} prompts "
                f"but {s['pending_count']} requests are pending",
            )

    def test_trie_drains_to_zero(self):
        """After all requests are completed, the trie must be empty."""
        final = self.snapshots[-1]
        self.assertEqual(final["trie_count"], 0,
                         f"Trie not empty at end: {final['trie_count']} prompts remain")
        self.assertEqual(final["pending_count"], 0)

    def test_all_requests_eventually_complete(self):
        """Every request must pass through the loop: arrive → dispatch → complete → remove."""
        total_arrivals = sum(s["arrivals_this_cycle"] for s in self.snapshots)
        total_completions = sum(s["completions_this_cycle"] for s in self.snapshots)
        self.assertEqual(total_arrivals, N_REQUESTS,
                         f"Only {total_arrivals}/{N_REQUESTS} requests arrived")
        self.assertEqual(total_completions, N_REQUESTS,
                         f"Only {total_completions}/{N_REQUESTS} requests completed")

    def test_trie_group_count_matches_pending_groups(self):
        """Number of groups in the trie must match the number of distinct
        groups with at least one pending request."""
        for s in self.snapshots:
            expected_groups = len(s["group_counts"])
            actual_groups = s["n_groups_in_trie"]
            self.assertEqual(
                actual_groups, expected_groups,
                f"Cycle {s['cycle']}: trie has {actual_groups} groups "
                f"but {expected_groups} groups have pending requests",
            )

    def test_pending_rises_then_falls(self):
        """Pending count should rise as arrivals outpace completions,
        then fall as the arrival stream ends."""
        counts = [s["pending_count"] for s in self.snapshots]
        peak = max(counts)
        peak_idx = counts.index(peak)

        self.assertGreater(peak, 0, "Peak pending should be > 0")
        self.assertEqual(counts[-1], 0, "Final pending should be 0")
        # There should be some rise before the peak
        self.assertGreater(peak_idx, 0, "Peak should not be at cycle 0")


class TestRankUpdateAfterRemoval(unittest.TestCase):
    """Verify that ranks update correctly when the server completes
    requests and the client removes them."""

    def test_rank_inversion_after_bulk_completion(self):
        """If the largest group's requests all complete, a smaller group
        should take rank 0 on the next submit."""
        reqs, labels = _generate_workload()

        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        # Find the two largest groups
        group_counter = Counter(labels)
        top2 = group_counter.most_common(2)
        group_a, count_a = top2[0]
        group_b, count_b = top2[1]

        # Submit all requests for group A and group B
        group_a_indices = []
        group_b_indices = []
        for i, lbl in enumerate(labels):
            if lbl == group_a:
                dispatcher.submit(dict(reqs[i]))
                group_a_indices.append((i, dispatcher._next_idx - 1))
            elif lbl == group_b:
                dispatcher.submit(dict(reqs[i]))
                group_b_indices.append((i, dispatcher._next_idx - 1))

        # A is rank 0, B is rank 1
        last_a_rank = int(dispatched[len(group_a_indices) - 1]["rid"].split(":")[1])
        self.assertEqual(last_a_rank, 0, f"Group A should be rank 0, got {last_a_rank}")

        # Server completes ALL of group A
        for orig_idx, prompt_idx in group_a_indices:
            dispatcher.remove(reqs[orig_idx]["token_ids"], prompt_idx)

        # Trie should now only have group B
        self.assertEqual(dispatcher._trie._num_prompts, len(group_b_indices))

        # Submit a new request for group B — should be rank 0 now
        dispatched.clear()
        # Find one more group B request (or reuse a token_ids pattern)
        b_sample_idx = group_b_indices[0][0]
        new_req = dict(reqs[b_sample_idx])
        new_req["id"] = "new-b"
        dispatcher.submit(new_req)

        new_rank = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(new_rank, 0,
                         f"After removing all of group A, group B should be rank 0, got {new_rank}")

    def test_ranks_track_pending_not_historical(self):
        """Ranks should reflect current pending counts, not cumulative
        historical counts.  Simulate: A sends 100, 80 complete;
        B sends 30, 0 complete.  B (30 pending) should outrank A (20 pending)."""
        reqs, labels = _generate_workload()

        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        group_counter = Counter(labels)
        largest_group = group_counter.most_common(1)[0][0]

        # Submit first 100 requests from the largest group
        submitted_largest = []
        count = 0
        for i, lbl in enumerate(labels):
            if lbl == largest_group and count < 100:
                dispatcher.submit(dict(reqs[i]))
                submitted_largest.append((i, dispatcher._next_idx - 1))
                count += 1

        # Submit 30 requests from a different group
        other_group = None
        submitted_other = []
        for i, lbl in enumerate(labels):
            if lbl != largest_group:
                if other_group is None:
                    other_group = lbl
                if lbl == other_group and len(submitted_other) < 30:
                    dispatcher.submit(dict(reqs[i]))
                    submitted_other.append((i, dispatcher._next_idx - 1))

        # Largest has rank 0 (100 > 30)
        # Now complete 80 from largest group → 20 pending
        for orig_idx, prompt_idx in submitted_largest[:80]:
            dispatcher.remove(reqs[orig_idx]["token_ids"], prompt_idx)

        # State: largest=20 pending, other=30 pending
        # Submit new request for other group — should be rank 0
        dispatched.clear()
        other_sample = submitted_other[0][0]
        new_req = dict(reqs[other_sample])
        new_req["id"] = "new-other"
        dispatcher.submit(new_req)

        new_rank = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(new_rank, 0,
                         f"Group with 31 pending should outrank group with 20 pending, "
                         f"but got rank {new_rank}")

    def test_group_disappears_and_returns(self):
        """A group that empties out and reappears should get a fresh rank."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        # Submit 3 to A, 5 to B
        a_indices = []
        for i in range(3):
            dispatcher.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
            a_indices.append((prefix_a + [i], dispatcher._next_idx - 1))

        for i in range(5):
            dispatcher.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})

        # Remove all of A → A disappears from trie
        for tids, pidx in a_indices:
            dispatcher.remove(tids, pidx)

        self.assertEqual(dispatcher._trie._num_prompts, 5)  # only B

        # A returns with 1 request
        dispatched.clear()
        dispatcher.submit({"id": "a-new", "token_ids": prefix_a + [99]})

        # B has 5 pending, A has 1 → B should be rank 0, A rank 1
        a_rank = int(dispatched[0]["rid"].split(":")[1])
        self.assertEqual(a_rank, 1,
                         f"Returning group A (1 pending) should be rank 1, got {a_rank}")


class TestRemoveIdempotent(unittest.TestCase):
    """Edge cases for remove()."""

    def test_double_remove_is_safe(self):
        """Removing the same prompt_index twice should not crash."""
        dispatched = []
        dispatcher = PeekDispatcher(send_fn=lambda r: dispatched.append(r))

        tids = list(range(200))
        dispatcher.submit({"id": "r-0", "token_ids": tids})
        dispatcher.remove(tids, 0)
        # Second remove — prompt_index 0 no longer in trie
        dispatcher.remove(tids, 0)  # should not crash
        self.assertEqual(dispatcher._trie._num_prompts, 0)

    def test_remove_unknown_index_is_safe(self):
        """Removing a never-inserted index should not crash."""
        dispatcher = PeekDispatcher(send_fn=lambda r: None)
        dispatcher.remove(list(range(200)), 999)  # never inserted
        self.assertEqual(dispatcher._trie._num_prompts, 0)


if __name__ == "__main__":
    unittest.main()
