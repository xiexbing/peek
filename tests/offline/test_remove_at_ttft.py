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

"""Verify remove() fires at TTFT (first token), not at completion.

Qwen 2.5 32B workload: 100 groups, 1000 requests, Poisson arrivals.

Simulates the streaming HTTP path: each request has a prefill phase
(waiting → scheduled → first token) and a decode phase (generating
remaining tokens).  remove() must fire at first-token time, while
the request is still decoding.
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import Counter, defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.engine import CacheStateStore
from peek.offline.reorder import PeekDispatcher


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
MAX_DEPTH = 128
SEED = 42

# Simulated timing (cycles)
PREFILL_CYCLES = 2    # cycles from admit to first token
DECODE_CYCLES = 5     # cycles from first token to completion
BATCH_SIZE = 32
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


# ---------------------------------------------------------------------------
# Simulated server with distinct prefill/decode phases
# ---------------------------------------------------------------------------

class PhasedServer:
    """Server with explicit prefill → first_token → decode → complete phases.

    Each request goes through:
      waiting → prefill (PREFILL_CYCLES) → FIRST TOKEN → decode (DECODE_CYCLES) → complete
    """

    def __init__(self, batch_size=BATCH_SIZE,
                 prefill_cycles=PREFILL_CYCLES,
                 decode_cycles=DECODE_CYCLES):
        self.batch_size = batch_size
        self.prefill_cycles = prefill_cycles
        self.decode_cycles = decode_cycles
        self.waiting: list[dict] = []
        self.prefilling: list[tuple[dict, int]] = []    # (req, remaining_prefill)
        self.decoding: list[tuple[dict, int]] = []      # (req, remaining_decode)

    def receive(self, request):
        self.waiting.append(request)

    def step(self):
        """Returns (first_token_reqs, completed_reqs) for this cycle."""
        first_token = []
        completed = []

        # Tick decoding → complete
        still_decoding = []
        for req, ttl in self.decoding:
            if ttl <= 1:
                completed.append(req)
            else:
                still_decoding.append((req, ttl - 1))
        self.decoding = still_decoding

        # Tick prefilling → first token
        still_prefilling = []
        for req, ttl in self.prefilling:
            if ttl <= 1:
                first_token.append(req)
                self.decoding.append((req, self.decode_cycles))
            else:
                still_prefilling.append((req, ttl - 1))
        self.prefilling = still_prefilling

        # Admit from waiting → prefilling
        to_admit = min(self.batch_size, len(self.waiting))
        for _ in range(to_admit):
            req = self.waiting.pop(0)
            self.prefilling.append((req, self.prefill_cycles))

        return first_token, completed

    @property
    def total_in_flight(self):
        return len(self.waiting) + len(self.prefilling) + len(self.decoding)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_ttft_simulation():
    CacheStateStore._instance = None

    reqs = _generate_workload()
    arrival_times = _poisson_arrival_times(N_REQUESTS)
    arrival_order = _poisson_order(N_REQUESTS)

    server = PhasedServer()
    all_dispatched = []

    def on_dispatch(req):
        all_dispatched.append(req)
        server.receive(req)

    dispatcher = PeekDispatcher(send_fn=on_dispatch)

    id_to_prompt_idx: dict[str, int] = {}
    id_to_token_ids: dict[str, list[int]] = {}

    next_arrival = 0
    snapshots = []
    total_first_tokens = 0
    total_completions = 0
    total_removes = 0

    for cycle in range(1500):
        cycle_time = cycle * CYCLE_DURATION

        # --- Arrivals ---
        arrivals_this_cycle = 0
        while next_arrival < N_REQUESTS and arrival_times[next_arrival] <= cycle_time:
            idx = arrival_order[next_arrival]
            req = dict(reqs[idx])
            prompt_idx = dispatcher._next_idx
            dispatcher.submit(req)
            id_to_prompt_idx[req["id"]] = prompt_idx
            id_to_token_ids[req["id"]] = reqs[idx]["token_ids"]
            next_arrival += 1
            arrivals_this_cycle += 1

        # --- Server step ---
        first_token_reqs, completed_reqs = server.step()
        total_first_tokens += len(first_token_reqs)
        total_completions += len(completed_reqs)

        # --- Client remove at FIRST TOKEN (not at completion) ---
        removes_this_cycle = 0
        for ft_req in first_token_reqs:
            rid = ft_req["id"]
            if rid in id_to_prompt_idx:
                dispatcher.remove(
                    id_to_token_ids.pop(rid),
                    id_to_prompt_idx.pop(rid),
                )
                removes_this_cycle += 1
                total_removes += 1

        snapshots.append({
            "cycle": cycle,
            "arrivals": arrivals_this_cycle,
            "first_tokens": len(first_token_reqs),
            "completions": len(completed_reqs),
            "removes": removes_this_cycle,
            "trie_count": dispatcher._trie._num_prompts,
            "server_waiting": len(server.waiting),
            "server_prefilling": len(server.prefilling),
            "server_decoding": len(server.decoding),
            "client_pending_total": sum(dispatcher._group_count.values()),
        })

        if (next_arrival >= N_REQUESTS and server.total_in_flight == 0
                and dispatcher._trie._num_prompts == 0):
            break

    return {
        "snapshots": snapshots,
        "dispatcher": dispatcher,
        "total_first_tokens": total_first_tokens,
        "total_completions": total_completions,
        "total_removes": total_removes,
    }


# ===================================================================
# Tests
# ===================================================================

class TestRemoveAtTTFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.result = run_ttft_simulation()
        cls.snapshots = cls.result["snapshots"]

    def test_all_requests_get_first_token(self):
        self.assertEqual(self.result["total_first_tokens"], N_REQUESTS)

    def test_all_requests_complete(self):
        self.assertEqual(self.result["total_completions"], N_REQUESTS)

    def test_removes_equal_first_tokens(self):
        """remove() must fire exactly once per first-token, not per completion."""
        self.assertEqual(self.result["total_removes"], N_REQUESTS)
        # Also check per-cycle: removes == first_tokens
        for s in self.snapshots:
            self.assertEqual(
                s["removes"], s["first_tokens"],
                f"Cycle {s['cycle']}: {s['removes']} removes != "
                f"{s['first_tokens']} first_tokens",
            )

    def test_trie_tracks_waiting_not_decoding(self):
        """Trie must track waiting+prefilling requests only.
        Once a request gets first-token (= starts decoding), it's
        removed from the trie.  So trie count should match
        server waiting + prefilling, NOT include decoding."""
        for s in self.snapshots:
            expected = s["server_waiting"] + s["server_prefilling"]
            self.assertEqual(
                s["trie_count"], expected,
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"waiting+prefilling={expected} "
                f"(waiting={s['server_waiting']}, "
                f"prefilling={s['server_prefilling']}, "
                f"decoding={s['server_decoding']})",
            )

    def test_trie_does_not_include_decoding_requests(self):
        """Find a cycle where requests are decoding but NOT in the trie."""
        found_decoding_cycle = False
        for s in self.snapshots:
            if s["server_decoding"] > 0:
                found_decoding_cycle = True
                # Trie should NOT include decoding requests
                self.assertLess(
                    s["trie_count"],
                    s["server_waiting"] + s["server_prefilling"] + s["server_decoding"],
                    f"Cycle {s['cycle']}: trie includes decoding requests",
                )
        self.assertTrue(found_decoding_cycle, "No cycles with decoding found")

    def test_trie_drains_before_all_completions(self):
        """Trie should reach zero BEFORE the last completion, because
        remove happens at first-token which precedes completion by
        DECODE_CYCLES."""
        trie_zero_cycle = None
        last_completion_cycle = None
        for s in self.snapshots:
            if trie_zero_cycle is None and s["trie_count"] == 0 and s["arrivals"] == 0:
                # Only count as "drained" if no more arrivals coming
                remaining_arrivals = sum(
                    ss["arrivals"] for ss in self.snapshots[s["cycle"]:]
                )
                if remaining_arrivals == 0:
                    trie_zero_cycle = s["cycle"]
            if s["completions"] > 0:
                last_completion_cycle = s["cycle"]

        self.assertIsNotNone(trie_zero_cycle)
        self.assertIsNotNone(last_completion_cycle)
        self.assertLess(
            trie_zero_cycle, last_completion_cycle,
            f"Trie drained at cycle {trie_zero_cycle} but last completion "
            f"at cycle {last_completion_cycle} — trie should drain first",
        )

    def test_client_pending_matches_trie(self):
        for s in self.snapshots:
            self.assertEqual(
                s["trie_count"], s["client_pending_total"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"client_pending={s['client_pending_total']}",
            )

    def test_trie_empty_at_end(self):
        final = self.snapshots[-1]
        self.assertEqual(final["trie_count"], 0)
        self.assertEqual(final["client_pending_total"], 0)


if __name__ == "__main__":
    unittest.main()
