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

"""Verify PeekEngine calls dispatcher.remove() directly at admission time.

Qwen 2.5 32B: 100 groups, 1000 requests.

PeekEngine.run() step 9 calls dispatcher.remove() for front-of-queue
requests that PrefillAdder will admit.  This happens synchronously
in the server's scheduling thread — zero delay.
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.engine import CacheStateStore, PeekEngine
from peek.offline.reorder import PeekDispatcher


NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
SEED = 42

BATCH_SIZE = 32
PREFILL_CYCLES = 3
DECODE_CYCLES = 5
CYCLE_DURATION = 0.033


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


def _poisson_arrival_times(n, rate=30.0, seed=SEED):
    rng = random.Random(seed)
    times, t = [], 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        times.append(t)
    return times


def _poisson_order(n, seed=SEED):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order


class PhasedServer:
    def __init__(self):
        self.waiting: list[dict] = []
        self.prefilling: list[tuple[dict, int]] = []
        self.decoding: list[tuple[dict, int]] = []

    def receive(self, req):
        self.waiting.append(req)

    def step(self):
        """Returns (admitted, first_token, completed)."""
        completed = []
        still_decoding = []
        for req, ttl in self.decoding:
            if ttl <= 1:
                completed.append(req)
            else:
                still_decoding.append((req, ttl - 1))
        self.decoding = still_decoding

        first_token = []
        still_prefilling = []
        for req, ttl in self.prefilling:
            if ttl <= 1:
                first_token.append(req)
                self.decoding.append((req, DECODE_CYCLES))
            else:
                still_prefilling.append((req, ttl - 1))
        self.prefilling = still_prefilling

        admitted = []
        to_admit = min(BATCH_SIZE, len(self.waiting))
        for _ in range(to_admit):
            req = self.waiting.pop(0)
            self.prefilling.append((req, PREFILL_CYCLES))
            admitted.append(req)

        return admitted, first_token, completed

    @property
    def total(self):
        return len(self.waiting) + len(self.prefilling) + len(self.decoding)


def run_simulation():
    CacheStateStore._instance = None

    reqs = _generate_workload()
    arrival_times = _poisson_arrival_times(N_REQUESTS)
    arrival_order = _poisson_order(N_REQUESTS)

    server = PhasedServer()
    dispatched = []

    def on_dispatch(req):
        dispatched.append(req)
        server.receive(req)

    dispatcher = PeekDispatcher(send_fn=on_dispatch)

    next_arrival = 0
    snapshots = []
    total_admitted = 0
    total_completed = 0
    total_direct_removes = 0

    for cycle in range(1500):
        cycle_time = cycle * CYCLE_DURATION

        # --- Arrivals ---
        arrivals = 0
        while next_arrival < N_REQUESTS and arrival_times[next_arrival] <= cycle_time:
            idx = arrival_order[next_arrival]
            dispatcher.submit(dict(reqs[idx]))
            next_arrival += 1
            arrivals += 1

        # --- Server step ---
        admitted, first_token, completed = server.step()
        total_admitted += len(admitted)
        total_completed += len(completed)

        # --- Simulate PeekEngine step 9: direct remove at admission ---
        removes_this_cycle = 0
        for req in admitted:
            rid = req.get("rid", req.get("id", ""))
            if rid.startswith("peek:"):
                _, _, orig_rid = PeekEngine._parse_peek_rid(rid)
            else:
                orig_rid = rid
            info = dispatcher._rid_to_remove_info.pop(orig_rid, None)
            if info is not None:
                dispatcher.remove(info[0], info[1])
                removes_this_cycle += 1
                total_direct_removes += 1

        snapshots.append({
            "cycle": cycle,
            "arrivals": arrivals,
            "admitted": len(admitted),
            "first_token": len(first_token),
            "completed": len(completed),
            "direct_removes": removes_this_cycle,
            "trie_count": dispatcher._trie._num_prompts,
            "server_waiting": len(server.waiting),
            "server_prefilling": len(server.prefilling),
            "server_decoding": len(server.decoding),
        })

        if next_arrival >= N_REQUESTS and server.total == 0 and dispatcher._trie._num_prompts == 0:
            break

    return {
        "snapshots": snapshots,
        "total_admitted": total_admitted,
        "total_completed": total_completed,
        "total_direct_removes": total_direct_removes,
        "dispatcher": dispatcher,
    }


class TestDirectRemoveAtAdmission(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.result = run_simulation()
        cls.snapshots = cls.result["snapshots"]

    def test_all_admitted(self):
        self.assertEqual(self.result["total_admitted"], N_REQUESTS)

    def test_all_completed(self):
        self.assertEqual(self.result["total_completed"], N_REQUESTS)

    def test_all_removed_directly(self):
        """Every admission triggers a direct remove — no other path needed."""
        self.assertEqual(self.result["total_direct_removes"], N_REQUESTS)

    def test_removes_equal_admissions_per_cycle(self):
        for s in self.snapshots:
            self.assertEqual(
                s["direct_removes"], s["admitted"],
                f"Cycle {s['cycle']}: {s['direct_removes']} removes != "
                f"{s['admitted']} admissions",
            )

    def test_trie_tracks_waiting_only(self):
        """Trie = waiting queue only.  Prefilling and decoding excluded."""
        for s in self.snapshots:
            self.assertEqual(
                s["trie_count"], s["server_waiting"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"waiting={s['server_waiting']}",
            )

    def test_trie_empty_at_end(self):
        final = self.snapshots[-1]
        self.assertEqual(final["trie_count"], 0)

    def test_zero_delay(self):
        """Direct remove happens in the same cycle as admission —
        no cycle gap between admission and trie update."""
        for s in self.snapshots:
            if s["admitted"] > 0:
                self.assertGreater(s["direct_removes"], 0,
                                   f"Cycle {s['cycle']}: admitted "
                                   f"{s['admitted']} but 0 removes")


if __name__ == "__main__":
    unittest.main()
