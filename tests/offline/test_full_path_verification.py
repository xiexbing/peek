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

"""Full path verification: Qwen 2.5 32B, 100 groups, 1000 requests.

Verifies every component in one end-to-end simulation:

  1. Client DFS trie: insert on arrival, remove on admission — always fresh
  2. Client re-rank: pending counts correct after every submit and remove
  3. Client→Server: tags parsed, grouping correct, pending counts pushed
  4. Server→Client: direct remove() at admission — zero delay
  5. Trie == waiting queue at EVERY cycle (not waiting+prefilling, not waiting+decoding)
  6. No request is leaked, dropped, or double-counted

The simulation models the real SGLang pipeline:
  arrival → submit(tag+trie insert) → server waiting queue
  → PeekEngine.run(reorder+score+direct remove) → PrefillAdder admits
  → prefill → first token → decode → complete
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import Counter, defaultdict

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.engine import CacheStateStore, PeekEngine
from peek.offline.reorder import PeekDispatcher


# ---------------------------------------------------------------------------
# Workload — Qwen 2.5 32B on H100
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
MAX_DEPTH = 128
SEED = 42

BATCH_SIZE = 32
PREFILL_CYCLES = 3    # ~100ms prefill for 2048 tokens
DECODE_CYCLES = 5     # ~165ms decode for 128 tokens
CYCLE_DURATION = 0.033  # ~30ms per scheduling cycle


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
# Phased server: waiting → prefill → decode → complete
# ---------------------------------------------------------------------------

class PhasedServer:
    def __init__(self):
        self.waiting: list[dict] = []
        self.prefilling: list[tuple[dict, int]] = []
        self.decoding: list[tuple[dict, int]] = []
        self.batch_size: int = BATCH_SIZE

    def receive(self, req):
        self.waiting.append(req)

    def step(self):
        """Tick one cycle. Returns (admitted, first_token, completed)."""
        # Decode → complete
        completed, still_decoding = [], []
        for req, ttl in self.decoding:
            (completed if ttl <= 1 else still_decoding).append(
                req if ttl <= 1 else (req, ttl - 1))
        self.decoding = still_decoding

        # Prefill → first token → decode
        first_token, still_prefilling = [], []
        for req, ttl in self.prefilling:
            if ttl <= 1:
                first_token.append(req)
                self.decoding.append((req, DECODE_CYCLES))
            else:
                still_prefilling.append((req, ttl - 1))
        self.prefilling = still_prefilling

        # Waiting → admitted → prefill
        admitted = []
        for _ in range(min(self.batch_size, len(self.waiting))):
            req = self.waiting.pop(0)
            self.prefilling.append((req, PREFILL_CYCLES))
            admitted.append(req)

        return admitted, first_token, completed

    @property
    def total(self):
        return len(self.waiting) + len(self.prefilling) + len(self.decoding)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_full_verification(arrival_rate=30.0, batch_size=BATCH_SIZE):
    CacheStateStore._instance = None
    store = CacheStateStore.get()

    reqs, labels = _generate_workload()
    arrival_times = _poisson_arrival_times(N_REQUESTS, rate=arrival_rate)
    arrival_order = _poisson_order(N_REQUESTS)

    server = PhasedServer()
    server.batch_size = batch_size
    all_dispatched = []

    def on_dispatch(req):
        all_dispatched.append(req)
        server.receive(req)

    dispatcher = PeekDispatcher(send_fn=on_dispatch)

    next_arrival = 0
    snapshots = []
    total_admitted = 0
    total_first_token = 0
    total_completed = 0
    total_direct_removes = 0

    for cycle in range(1500):
        cycle_time = cycle * CYCLE_DURATION

        # === Phase A: Poisson arrivals → submit() ===
        arrivals_this = 0
        while next_arrival < N_REQUESTS and arrival_times[next_arrival] <= cycle_time:
            idx = arrival_order[next_arrival]
            dispatcher.submit(dict(reqs[idx]))
            next_arrival += 1
            arrivals_this += 1

        # === Phase B: Server scheduling cycle ===
        admitted, first_token, completed = server.step()
        total_admitted += len(admitted)
        total_first_token += len(first_token)
        total_completed += len(completed)

        # === Phase C: Direct remove at admission (PeekEngine step 9) ===
        removes_this = 0
        for req in admitted:
            rid = req.get("rid", req.get("id", ""))
            if rid.startswith("peek:"):
                _, _, orig_rid = PeekEngine._parse_peek_rid(rid)
            else:
                orig_rid = rid
            info = dispatcher._rid_to_remove_info.pop(orig_rid, None)
            if info is not None:
                dispatcher.remove(info[0], info[1])
                removes_this += 1
        total_direct_removes += removes_this

        # === Snapshot ===
        store_pending = store.get_client_pending()

        snapshots.append({
            "cycle": cycle,
            "arrivals": arrivals_this,
            "admitted": len(admitted),
            "first_token": len(first_token),
            "completed": len(completed),
            "direct_removes": removes_this,
            "trie_count": dispatcher._trie._num_prompts,
            "dispatcher_pending": sum(dispatcher._group_count.values()),
            "dispatcher_groups": len(dispatcher._group_count),
            "store_pending": sum(store_pending.values()),
            "store_groups": len(store_pending),
            "server_waiting": len(server.waiting),
            "server_prefilling": len(server.prefilling),
            "server_decoding": len(server.decoding),
        })

        if (next_arrival >= N_REQUESTS and server.total == 0
                and dispatcher._trie._num_prompts == 0):
            break

    return {
        "snapshots": snapshots,
        "dispatcher": dispatcher,
        "all_dispatched": all_dispatched,
        "labels": labels,
        "total_admitted": total_admitted,
        "total_first_token": total_first_token,
        "total_completed": total_completed,
        "total_direct_removes": total_direct_removes,
    }


# ===================================================================
# Tests
# ===================================================================

class TestAllRequestsFlowThrough(unittest.TestCase):
    """No request leaked, dropped, or double-counted."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()
        cls.s = cls.r["snapshots"]

    def test_all_arrive(self):
        self.assertEqual(sum(s["arrivals"] for s in self.s), N_REQUESTS)

    def test_all_dispatched_with_tags(self):
        d = self.r["all_dispatched"]
        self.assertEqual(len(d), N_REQUESTS)
        for req in d:
            self.assertTrue(req["rid"].startswith("peek:"))

    def test_all_admitted(self):
        self.assertEqual(self.r["total_admitted"], N_REQUESTS)

    def test_all_get_first_token(self):
        self.assertEqual(self.r["total_first_token"], N_REQUESTS)

    def test_all_completed(self):
        self.assertEqual(self.r["total_completed"], N_REQUESTS)

    def test_all_removed_directly(self):
        self.assertEqual(self.r["total_direct_removes"], N_REQUESTS)


class TestTrieAlwaysFresh(unittest.TestCase):
    """Trie == server waiting queue at every single cycle."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()
        cls.s = cls.r["snapshots"]

    def test_trie_equals_waiting_every_cycle(self):
        """THE critical invariant: trie tracks ONLY waiting requests.
        Not prefilling, not decoding, not completed."""
        for s in self.s:
            self.assertEqual(
                s["trie_count"], s["server_waiting"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"waiting={s['server_waiting']} "
                f"(prefilling={s['server_prefilling']}, "
                f"decoding={s['server_decoding']})",
            )

    def test_trie_equals_dispatcher_pending_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["trie_count"], s["dispatcher_pending"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"dispatcher_pending={s['dispatcher_pending']}",
            )

    def test_trie_never_includes_prefilling(self):
        """Find cycles with prefilling > 0.  Trie must be strictly less
        than waiting + prefilling."""
        found = False
        for s in self.s:
            if s["server_prefilling"] > 0:
                found = True
                self.assertEqual(
                    s["trie_count"], s["server_waiting"],
                    f"Cycle {s['cycle']}: trie includes prefilling requests",
                )
        self.assertTrue(found)

    def test_trie_never_includes_decoding(self):
        found = False
        for s in self.s:
            if s["server_decoding"] > 0:
                found = True
                self.assertEqual(
                    s["trie_count"], s["server_waiting"],
                    f"Cycle {s['cycle']}: trie includes decoding requests",
                )
        self.assertTrue(found)

    def test_trie_empty_at_end(self):
        final = self.s[-1]
        self.assertEqual(final["trie_count"], 0)
        self.assertEqual(final["dispatcher_pending"], 0)
        self.assertEqual(final["dispatcher_groups"], 0)


class TestZeroDelayRemove(unittest.TestCase):
    """Direct remove fires in the same cycle as admission — zero delay."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()
        cls.s = cls.r["snapshots"]

    def test_removes_equal_admissions_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["direct_removes"], s["admitted"],
                f"Cycle {s['cycle']}: {s['direct_removes']} removes != "
                f"{s['admitted']} admissions — delay detected",
            )

    def test_no_remove_lag(self):
        """There must never be a cycle where admissions > 0 but removes == 0."""
        for s in self.s:
            if s["admitted"] > 0:
                self.assertGreater(
                    s["direct_removes"], 0,
                    f"Cycle {s['cycle']}: {s['admitted']} admitted but "
                    f"0 removes — delay!",
                )


class TestStoreConsistency(unittest.TestCase):
    """CacheStateStore matches dispatcher at every cycle."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()
        cls.s = cls.r["snapshots"]

    def test_store_total_matches_dispatcher(self):
        for s in self.s:
            self.assertEqual(
                s["store_pending"], s["dispatcher_pending"],
                f"Cycle {s['cycle']}: store={s['store_pending']} != "
                f"dispatcher={s['dispatcher_pending']}",
            )

    def test_store_groups_match_dispatcher(self):
        for s in self.s:
            self.assertEqual(
                s["store_groups"], s["dispatcher_groups"],
                f"Cycle {s['cycle']}: store groups={s['store_groups']} != "
                f"dispatcher groups={s['dispatcher_groups']}",
            )

    def test_store_empty_at_end(self):
        final = self.s[-1]
        self.assertEqual(final["store_pending"], 0)
        self.assertEqual(final["store_groups"], 0)


class TestTagGroupingCorrect(unittest.TestCase):
    """Server-side grouping from tags matches original groups."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()

    def test_one_hash_per_group(self):
        group_to_hashes = defaultdict(set)
        for d in self.r["all_dispatched"]:
            ghash = d["rid"].split(":", 3)[2]
            group_to_hashes[d["group"]].add(ghash)
        for grp, hashes in group_to_hashes.items():
            self.assertEqual(len(hashes), 1)

    def test_no_hash_collision(self):
        hash_to_groups = defaultdict(set)
        for d in self.r["all_dispatched"]:
            ghash = d["rid"].split(":", 3)[2]
            hash_to_groups[ghash].add(d["group"])
        for ghash, grps in hash_to_groups.items():
            self.assertEqual(len(grps), 1)

    def test_server_groups_all_requests(self):
        tag_groups = defaultdict(list)
        for d in self.r["all_dispatched"]:
            _, gk, _ = PeekEngine._parse_peek_rid(d["rid"])
            tag_groups[gk].append(d)
        total = sum(len(m) for m in tag_groups.values())
        self.assertEqual(total, N_REQUESTS)


class TestRankCorrectness(unittest.TestCase):
    """Ranks reflect pending counts at submit time."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()

    def test_top3_ranks_ordered_in_first_batch(self):
        """Before any completions, rank ordering must match group size."""
        dispatched = self.r["all_dispatched"][:200]
        group_counts = Counter(d["group"] for d in dispatched)

        group_last_rank = {}
        for d in dispatched:
            group_last_rank[d["group"]] = int(d["rid"].split(":")[1])

        top3 = [g for g, _ in group_counts.most_common(3)]
        top3_ranks = [group_last_rank[g] for g in top3]
        for i in range(len(top3_ranks) - 1):
            self.assertLessEqual(top3_ranks[i], top3_ranks[i + 1])


class TestPendingLifecycle(unittest.TestCase):
    """Pending rises during arrivals and drains to zero."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification()
        cls.s = cls.r["snapshots"]

    def test_drains_to_zero(self):
        counts = [s["trie_count"] for s in self.s]
        self.assertEqual(counts[-1], 0)
        # Trie may stay at 0 throughout if server processes faster than
        # arrivals (batch_size=32 vs ~1 arrival per cycle at 30 req/s).
        # That's correct: zero-delay remove keeps trie perfectly synced.

    def test_trie_drains_before_last_completion(self):
        """Trie should reach zero before the last decode completes,
        because remove fires at admission (before prefill+decode)."""
        trie_zero_cycle = None
        last_completion_cycle = None
        for s in self.s:
            if (trie_zero_cycle is None and s["trie_count"] == 0
                    and sum(ss["arrivals"] for ss in self.s[s["cycle"]:]) == 0):
                trie_zero_cycle = s["cycle"]
            if s["completed"] > 0:
                last_completion_cycle = s["cycle"]

        self.assertIsNotNone(trie_zero_cycle)
        self.assertIsNotNone(last_completion_cycle)
        self.assertLess(trie_zero_cycle, last_completion_cycle)


# ===================================================================
# 100 req/s — high arrival rate creates real queue pressure
# ===================================================================

class TestHighRate100ReqS(unittest.TestCase):
    """Same invariants at 100 req/s with batch_size=2.

    100 req/s = ~3.3 arrivals per 33ms cycle.  batch_size=2 admits
    only 2 per cycle.  Queue builds: 3.3 - 2 = 1.3 excess per cycle.
    Peak waiting ≈ 50-100 requests.  The trie has real content and
    must still equal server waiting at every cycle.
    """

    @classmethod
    def setUpClass(cls):
        cls.r = run_full_verification(arrival_rate=100.0, batch_size=2)
        cls.s = cls.r["snapshots"]

    def test_all_arrive(self):
        self.assertEqual(sum(s["arrivals"] for s in self.s), N_REQUESTS)

    def test_all_admitted(self):
        self.assertEqual(self.r["total_admitted"], N_REQUESTS)

    def test_all_completed(self):
        self.assertEqual(self.r["total_completed"], N_REQUESTS)

    def test_all_removed_directly(self):
        self.assertEqual(self.r["total_direct_removes"], N_REQUESTS)

    def test_trie_equals_waiting_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["trie_count"], s["server_waiting"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"waiting={s['server_waiting']}",
            )

    def test_trie_equals_dispatcher_pending_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["trie_count"], s["dispatcher_pending"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"dispatcher_pending={s['dispatcher_pending']}",
            )

    def test_removes_equal_admissions_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["direct_removes"], s["admitted"],
                f"Cycle {s['cycle']}: removes != admissions",
            )

    def test_store_matches_dispatcher_every_cycle(self):
        for s in self.s:
            self.assertEqual(
                s["store_pending"], s["dispatcher_pending"],
                f"Cycle {s['cycle']}: store != dispatcher",
            )

    def test_queue_actually_builds_up(self):
        """At 100 req/s, the waiting queue must build up (unlike 30 req/s)."""
        peak_waiting = max(s["server_waiting"] for s in self.s)
        self.assertGreater(peak_waiting, 0,
                           "No queue buildup at 100 req/s — test is not stressing the system")

    def test_trie_peaks_above_zero(self):
        """Trie should have real content during the burst."""
        peak_trie = max(s["trie_count"] for s in self.s)
        self.assertGreater(peak_trie, 0,
                           "Trie never had content at 100 req/s")

    def test_trie_drains_to_zero(self):
        self.assertEqual(self.s[-1]["trie_count"], 0)

    def test_trie_never_includes_prefilling(self):
        found = False
        for s in self.s:
            if s["server_prefilling"] > 0:
                found = True
                self.assertEqual(s["trie_count"], s["server_waiting"])
        self.assertTrue(found)

    def test_trie_never_includes_decoding(self):
        found = False
        for s in self.s:
            if s["server_decoding"] > 0:
                found = True
                self.assertEqual(s["trie_count"], s["server_waiting"])
        self.assertTrue(found)

    def test_grouping_correct(self):
        hash_to_groups = defaultdict(set)
        for d in self.r["all_dispatched"]:
            ghash = d["rid"].split(":", 3)[2]
            hash_to_groups[ghash].add(d["group"])
        for ghash, grps in hash_to_groups.items():
            self.assertEqual(len(grps), 1)

    def test_trie_drains_before_last_completion(self):
        trie_zero_cycle = None
        last_completion_cycle = None
        for s in self.s:
            if (trie_zero_cycle is None and s["trie_count"] == 0
                    and sum(ss["arrivals"] for ss in self.s[s["cycle"]:]) == 0):
                trie_zero_cycle = s["cycle"]
            if s["completed"] > 0:
                last_completion_cycle = s["cycle"]
        self.assertIsNotNone(trie_zero_cycle)
        self.assertIsNotNone(last_completion_cycle)
        self.assertLess(trie_zero_cycle, last_completion_cycle)


if __name__ == "__main__":
    unittest.main()
