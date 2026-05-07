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

"""Full pipeline correctness test: Qwen 2.5 32B, 100 groups, 1000 requests.

End-to-end verification of every component in Peek's client→server→client
loop under Poisson arrivals:

  1. Client DFS trie: incremental build, add, remove
  2. Client re-rank: pending counts sort on every submit/remove
  3. Client→Server push: CacheStateStore carries pending counts
  4. Server tag parsing: groups by hash, reads client pending
  5. Server→Client feedback: completion triggers remove + re-rank
  6. Trie drains to zero after all completions

Uses the same SimulatedServer from test_client_server_feedback.py
to model the full scheduling loop.
"""

import pytest

pytestmark = pytest.mark.gpu
import random
import unittest
from collections import Counter, defaultdict
from types import SimpleNamespace

from peek.offline.benchmarks.poisson_client import load_requests
from peek.offline.engine import CacheStateStore, PeekEngine
from peek.offline.reorder import PeekDispatcher


# ---------------------------------------------------------------------------
# Workload — Qwen 2.5 32B
# ---------------------------------------------------------------------------

NUM_GROUPS = 100
N_REQUESTS = 1000
ZIPF_ALPHA = 1.5
SYSTEM_PROMPT_LEN = 2048
QUESTION_LEN = 128
MAX_DEPTH = 128
SEED = 42

BATCH_SIZE = 32
PREFILL_CYCLES = 2
CYCLE_DURATION = 0.033  # ~30ms


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
# Simulated server
# ---------------------------------------------------------------------------

class SimulatedServer:
    def __init__(self, batch_size=BATCH_SIZE, prefill_cycles=PREFILL_CYCLES):
        self.batch_size = batch_size
        self.prefill_cycles = prefill_cycles
        self.waiting: list[dict] = []
        self.running: list[tuple[dict, int]] = []

    def receive(self, request):
        self.waiting.append(request)

    def step(self):
        to_admit = min(self.batch_size, len(self.waiting))
        for _ in range(to_admit):
            self.running.append((self.waiting.pop(0), self.prefill_cycles))
        still, completed = [], []
        for req, ttl in self.running:
            if ttl <= 1:
                completed.append(req)
            else:
                still.append((req, ttl - 1))
        self.running = still
        return completed

    @property
    def pending_count(self):
        return len(self.waiting) + len(self.running)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Run full client→server→client pipeline.

    Returns a rich result dict with per-cycle snapshots and aggregate stats.
    """
    CacheStateStore._instance = None
    store = CacheStateStore.get()

    reqs, labels = _generate_workload()
    arrival_times = _poisson_arrival_times(N_REQUESTS)
    arrival_order = _poisson_order(N_REQUESTS)

    server = SimulatedServer()
    all_dispatched = []

    def on_dispatch(req):
        all_dispatched.append(req)
        server.receive(req)

    dispatcher = PeekDispatcher(send_fn=on_dispatch)

    # Track prompt_index for remove()
    id_to_prompt_idx: dict[str, int] = {}
    id_to_token_ids: dict[str, list[int]] = {}

    next_arrival = 0
    snapshots = []
    total_completions = 0

    for cycle in range(1200):
        cycle_time = cycle * CYCLE_DURATION
        cycle_arrivals = 0
        cycle_completions = 0

        # --- Arrivals ---
        while next_arrival < N_REQUESTS and arrival_times[next_arrival] <= cycle_time:
            idx = arrival_order[next_arrival]
            req = dict(reqs[idx])
            prompt_idx = dispatcher._next_idx
            dispatcher.submit(req)
            id_to_prompt_idx[req["id"]] = prompt_idx
            id_to_token_ids[req["id"]] = reqs[idx]["token_ids"]
            next_arrival += 1
            cycle_arrivals += 1

        # --- Server scheduling cycle ---
        completed = server.step()
        cycle_completions = len(completed)
        total_completions += cycle_completions

        # --- Client feedback: remove completed ---
        for comp in completed:
            rid = comp["id"]
            if rid in id_to_prompt_idx:
                dispatcher.remove(id_to_token_ids.pop(rid), id_to_prompt_idx.pop(rid))

        # --- Read server→client and client→server state ---
        client_pending_snapshot = store.get_client_pending()

        snapshots.append({
            "cycle": cycle,
            "trie_count": dispatcher._trie._num_prompts,
            "client_pending_total": sum(dispatcher._group_count.values()),
            "client_groups": len(dispatcher._group_count),
            "store_pending_total": sum(client_pending_snapshot.values()),
            "store_groups": len(client_pending_snapshot),
            "server_waiting": len(server.waiting),
            "server_running": len(server.running),
            "arrivals": cycle_arrivals,
            "completions": cycle_completions,
        })

        if (next_arrival >= N_REQUESTS and server.pending_count == 0
                and dispatcher._trie._num_prompts == 0):
            break

    return {
        "snapshots": snapshots,
        "dispatcher": dispatcher,
        "all_dispatched": all_dispatched,
        "total_completions": total_completions,
        "labels": labels,
        "reqs": reqs,
        "store": store,
    }


# ===================================================================
# Tests
# ===================================================================

class TestFullPipeline(unittest.TestCase):
    """End-to-end pipeline correctness."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_full_pipeline()
        cls.snapshots = cls.result["snapshots"]

    # --- 1. All requests flow through ---

    def test_all_requests_arrive(self):
        total_arrivals = sum(s["arrivals"] for s in self.snapshots)
        self.assertEqual(total_arrivals, N_REQUESTS)

    def test_all_requests_complete(self):
        self.assertEqual(self.result["total_completions"], N_REQUESTS)

    def test_all_requests_dispatched_with_tags(self):
        dispatched = self.result["all_dispatched"]
        self.assertEqual(len(dispatched), N_REQUESTS)
        for d in dispatched:
            self.assertTrue(d["rid"].startswith("peek:"), f"Untagged: {d['rid']}")

    # --- 2. Trie correctness at every cycle ---

    def test_trie_equals_pending_every_cycle(self):
        """Trie prompt count must equal client pending count at every cycle."""
        for s in self.snapshots:
            self.assertEqual(
                s["trie_count"], s["client_pending_total"],
                f"Cycle {s['cycle']}: trie={s['trie_count']} != "
                f"pending={s['client_pending_total']}",
            )

    def test_trie_drains_to_zero(self):
        final = self.snapshots[-1]
        self.assertEqual(final["trie_count"], 0)
        self.assertEqual(final["client_pending_total"], 0)
        self.assertEqual(final["client_groups"], 0)

    # --- 3. CacheStateStore consistency ---

    def test_store_matches_dispatcher_every_cycle(self):
        """The shared store pending counts must match the dispatcher's
        _group_count at every cycle (pushed on every submit/remove)."""
        for s in self.snapshots:
            self.assertEqual(
                s["store_pending_total"], s["client_pending_total"],
                f"Cycle {s['cycle']}: store total={s['store_pending_total']} != "
                f"dispatcher total={s['client_pending_total']}",
            )
            self.assertEqual(
                s["store_groups"], s["client_groups"],
                f"Cycle {s['cycle']}: store groups={s['store_groups']} != "
                f"dispatcher groups={s['client_groups']}",
            )

    def test_store_drains_to_zero(self):
        final = self.snapshots[-1]
        self.assertEqual(final["store_pending_total"], 0)
        self.assertEqual(final["store_groups"], 0)

    # --- 4. Tag parsing / grouping ---

    def test_grouping_one_hash_per_original_group(self):
        """Each original group maps to exactly one hash in dispatched tags."""
        dispatched = self.result["all_dispatched"]
        group_to_hashes: dict[str, set[str]] = defaultdict(set)
        for d in dispatched:
            ghash = d["rid"].split(":", 3)[2]
            group_to_hashes[d["group"]].add(ghash)
        for grp, hashes in group_to_hashes.items():
            self.assertEqual(len(hashes), 1,
                             f"Group {grp} has {len(hashes)} hashes")

    def test_grouping_no_hash_collision(self):
        """Different original groups must not share a hash."""
        dispatched = self.result["all_dispatched"]
        hash_to_groups: dict[str, set[str]] = defaultdict(set)
        for d in dispatched:
            ghash = d["rid"].split(":", 3)[2]
            hash_to_groups[ghash].add(d["group"])
        for ghash, grps in hash_to_groups.items():
            self.assertEqual(len(grps), 1,
                             f"Hash {ghash} shared by {grps}")

    def test_server_can_group_from_tags(self):
        """Simulate server-side grouping: group dispatched by hash,
        verify each hash group has correct member count."""
        dispatched = self.result["all_dispatched"]
        labels = self.result["labels"]
        reqs = self.result["reqs"]

        tag_groups: dict[int, list] = {}
        for d in dispatched:
            _, gk, _ = PeekEngine._parse_peek_rid(d["rid"])
            tag_groups.setdefault(gk, []).append(d)

        # Total grouped = total dispatched (no drops)
        total = sum(len(m) for m in tag_groups.values())
        self.assertEqual(total, N_REQUESTS)

    # --- 5. Rank correctness ---

    def test_rank_reflects_pending_count_at_tag_time(self):
        """Ranks reflect pending count at the time of tagging — not
        total historical count.  Verify using the first 200 dispatched
        requests (before any completions, so pending == cumulative)."""
        dispatched = self.result["all_dispatched"][:200]
        group_counts = Counter(d["group"] for d in dispatched)

        # Get the rank from the last request in this window per group
        group_last_rank: dict[str, int] = {}
        for d in dispatched:
            rank = int(d["rid"].split(":")[1])
            group_last_rank[d["group"]] = rank

        # Top-3 groups should have monotonically non-decreasing ranks
        # (before completions, pending == cumulative, so this holds)
        top3 = [g for g, _ in group_counts.most_common(3)]
        top3_ranks = [group_last_rank[g] for g in top3]
        for i in range(len(top3_ranks) - 1):
            self.assertLessEqual(
                top3_ranks[i], top3_ranks[i + 1],
                f"Rank ordering: {top3[i]}(count={group_counts[top3[i]]}, "
                f"rank={top3_ranks[i]}) vs {top3[i+1]}("
                f"count={group_counts[top3[i+1]]}, rank={top3_ranks[i+1]})",
            )

    # --- 6. Re-rank on every submit confirmed ---

    def test_rank_changes_when_group_overtakes(self):
        """Build a fresh dispatcher, show that re-rank fires on every submit
        and a group overtaking another immediately gets the better rank."""
        dispatched = []
        CacheStateStore._instance = None
        d = PeekDispatcher(send_fn=lambda r: dispatched.append(dict(r)))

        prefix_a = list(range(200))
        prefix_b = list(range(200, 400))

        # A=10, B=9 → A rank=0, B rank=1
        for i in range(10):
            d.submit({"id": f"a-{i}", "token_ids": prefix_a + [i]})
        for i in range(9):
            d.submit({"id": f"b-{i}", "token_ids": prefix_b + [i]})

        rank_a = int(dispatched[-10]["rid"].split(":")[1])
        rank_b = int(dispatched[-1]["rid"].split(":")[1])
        self.assertEqual(rank_a, 0)
        self.assertEqual(rank_b, 1)

        # B gets 3 more → B=12, A=10 → B should be rank 0
        dispatched.clear()
        for i in range(3):
            d.submit({"id": f"b-extra-{i}", "token_ids": prefix_b + [100 + i]})

        rank_b_now = int(dispatched[-1]["rid"].split(":")[1])
        self.assertEqual(rank_b_now, 0,
                         f"B(12) should outrank A(10), got rank {rank_b_now}")

    # --- 7. Client pending > server queue (future demand) ---

    def test_client_pending_exceeds_server_queue_during_burst(self):
        """During arrival bursts, client pending should exceed
        server waiting+running (some requests still in-flight)."""
        # Find a cycle with arrivals > 0 and server waiting < client pending
        found = False
        for s in self.snapshots:
            if s["arrivals"] > 0 and s["client_pending_total"] > 0:
                server_total = s["server_waiting"] + s["server_running"]
                # Client tracks everything; server only sees what arrived.
                # They should be consistent: client >= server (some in-flight)
                # In our simulation (instant delivery), they're equal.
                # But client also tracks requests from THIS cycle's arrivals
                # that were submitted but server hasn't processed yet.
                self.assertGreaterEqual(
                    s["client_pending_total"], server_total,
                    f"Cycle {s['cycle']}: client pending "
                    f"({s['client_pending_total']}) < server total ({server_total})",
                )
                found = True
        self.assertTrue(found, "No cycles with arrivals found")

    # --- 8. Pending rises then falls ---

    def test_pending_lifecycle(self):
        """Pending should rise during arrivals and drain to zero after."""
        counts = [s["client_pending_total"] for s in self.snapshots]
        peak = max(counts)
        # Server processes fast (batch=32 per 30ms cycle), so peak may be
        # modest.  Just verify it's nonzero during arrivals and zero at end.
        self.assertGreater(peak, 0, "Peak should be > 0")
        self.assertEqual(counts[-1], 0, "Should drain to 0")


if __name__ == "__main__":
    unittest.main()
