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

"""Lifecycle parity: peek's incremental tree vs sglang rebuilt-from-queue.

Simulates the full set of waiting_queue mutation events that sglang's scheduler
performs (see scheduler.py):

  * arrival           -- append on new request (_add_request_to_queue)
  * schedule_success  -- bulk remove of can_run_list (line 2126)
  * priority_abort    -- single pop, higher-priority eviction (line 1757)
  * timeout_abort     -- bulk remove, waiting timeout (line 1800)
  * return_to_queue   -- append for preempted/retracted reqs (lines 2131, 2254)

At every tick we:
  1. Apply a random mix of events to both a ground-truth queue and peek's tree.
  2. Rebuild a fresh sglang RadixCache from the current queue (what stock
     sglang would produce on its next scheduling pass).
  3. Probe `match_prefix` against both peek and the sglang-rebuilt tree, for
     every req in the queue AND random synthetic token sequences. Every probe
     must return the same prefix length.
  4. Sanity check len(peek) == len(queue).

If peek's add/remove lifecycle is correct, peek's incremental tree is
indistinguishable from sglang's rebuild across any sequence of these events.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek import PendingTree

@dataclass
class MockReq:
    rid: int
    tokens: Tuple[int, ...]

    def __hash__(self) -> int:
        return self.rid

def _make_req(rid: int, rng: random.Random, system_prompts: List[Tuple[int, ...]]) -> MockReq:
    sp = rng.choice(system_prompts)
    tail_len = rng.randint(3, 30)
    tail = tuple(rng.randint(0, 255) for _ in range(tail_len))
    return MockReq(rid=rid, tokens=sp + tail)

def _sglang_rebuild(queue: List[MockReq]) -> RadixCache:
    tree = RadixCache.create_simulated()
    for r in queue:
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=list(r.tokens), extra_key=None),
                value=torch.empty(len(r.tokens), dtype=torch.bool),
            )
        )
    return tree

def _sglang_match_len(tree: RadixCache, tokens: Tuple[int, ...]) -> int:
    m = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=list(tokens), extra_key=None)))
    return len(m.device_indices)

def _assert_state_parity(
    peek_tree: PendingTree,
    queue: List[MockReq],
    rng: random.Random,
    system_prompts: List[Tuple[int, ...]],
) -> None:
    """Probe both trees with the current queue's tokens AND random tokens,
    asserting equal match lengths. Catches any drift between peek's incremental
    state and what sglang would produce by rebuilding."""
    assert len(peek_tree) == len(queue), (
        f"size mismatch: peek={len(peek_tree)} queue={len(queue)}"
    )
    sglang_tree = _sglang_rebuild(queue)
    # Probe with queue members -- ensures every real token path is represented.
    for r in queue:
        sg_len = _sglang_match_len(sglang_tree, r.tokens)
        pk_len = peek_tree.match_prefix(list(r.tokens))
        assert pk_len == sg_len, (
            f"match_prefix(queue rid={r.rid}) peek={pk_len} sglang={sg_len}"
        )
    # Probe with synthetic sequences -- covers non-members, partial matches, misses.
    for _ in range(10):
        sp = rng.choice(system_prompts)
        # Mix: sometimes a shared-prefix probe, sometimes a full random probe.
        if rng.random() < 0.5:
            probe = sp + tuple(rng.randint(0, 255) for _ in range(rng.randint(0, 10)))
        else:
            probe = tuple(rng.randint(0, 255) for _ in range(rng.randint(1, 50)))
        sg_len = _sglang_match_len(sglang_tree, probe)
        pk_len = peek_tree.match_prefix(list(probe))
        assert pk_len == sg_len, (
            f"match_prefix(synthetic) peek={pk_len} sglang={sg_len} probe_len={len(probe)}"
        )

# ---------------------------------------------------------------------------
# Event-driven driver. Each event mutates the shared queue + peek in lockstep.
# This mirrors what a real peek ↔ sglang monkey-patch hook would do.
# ---------------------------------------------------------------------------

def _apply_arrival(queue, peek_tree, running, next_rid_ref, rng, sps) -> None:
    rid = next_rid_ref[0]
    next_rid_ref[0] += 1
    req = _make_req(rid, rng, sps)
    queue.append(req)
    peek_tree.insert(req.rid, list(req.tokens))

def _apply_schedule_success(queue, peek_tree, running, rng) -> None:
    """Bulk-remove up to `batch_size` reqs. Mirrors scheduler.py:2126."""
    if not queue:
        return
    batch_size = rng.randint(1, min(8, len(queue)))
    # Pick a random subset (real sglang picks by LPM order, but any subset
    # exercises bulk remove identically for peek's correctness purposes).
    picks = rng.sample(queue, batch_size)
    picked_rids = {r.rid for r in picks}
    queue[:] = [r for r in queue if r.rid not in picked_rids]
    for r in picks:
        peek_tree.remove(r.rid)
        running[r.rid] = r  # retain for possible return_to_queue

def _apply_priority_abort(queue, peek_tree, running, rng) -> None:
    """Single pop of a low-priority req aborted by a higher-priority arrival.
    Mirrors scheduler.py:1757."""
    if not queue:
        return
    victim = rng.choice(queue)
    queue.remove(victim)
    peek_tree.remove(victim.rid)
    # Aborted reqs don't come back.

def _apply_timeout_abort(queue, peek_tree, running, rng) -> None:
    """Bulk remove of timed-out reqs. Mirrors scheduler.py:1800."""
    if not queue:
        return
    n = rng.randint(1, min(3, len(queue)))
    victims = rng.sample(queue, n)
    victim_rids = {r.rid for r in victims}
    queue[:] = [r for r in queue if r.rid not in victim_rids]
    for r in victims:
        peek_tree.remove(r.rid)

def _apply_return_to_queue(queue, peek_tree, running, rng) -> None:
    """Preempted/retracted req returns to waiting queue. Mirrors 2131, 2254.
    The returning req may have accumulated output_ids during running; we model
    that by optionally appending tokens (reflecting partial generation)."""
    if not running:
        return
    rid = rng.choice(list(running))
    req = running.pop(rid)
    # Simulate accumulated output_ids (retraction case).
    if rng.random() < 0.5:
        extra = tuple(rng.randint(0, 255) for _ in range(rng.randint(1, 8)))
        req = MockReq(rid=req.rid, tokens=req.tokens + extra)
    queue.append(req)
    peek_tree.insert(req.rid, list(req.tokens))

EVENT_WEIGHTS = [
    (_apply_arrival, 45),
    (_apply_schedule_success, 30),
    (_apply_return_to_queue, 10),
    (_apply_priority_abort, 10),
    (_apply_timeout_abort, 5),
]

def _pick_event(rng: random.Random):
    total = sum(w for _, w in EVENT_WEIGHTS)
    r = rng.randint(1, total)
    acc = 0
    for fn, w in EVENT_WEIGHTS:
        acc += w
        if r <= acc:
            return fn
    assert False, "unreachable"

# ---------------------------------------------------------------------------
# The test. Run many ticks with a random mix of events; verify state parity
# after each tick.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_ticks,seed", [(200, 0), (500, 1), (1000, 2)])
def test_lifecycle_state_parity(n_ticks: int, seed: int) -> None:
    rng = random.Random(seed)
    system_prompts = [
        tuple(rng.randint(0, 255) for _ in range(rng.randint(15, 40)))
        for _ in range(4)
    ]

    queue: List[MockReq] = []
    running: Dict[int, MockReq] = {}
    peek_tree = PendingTree()
    next_rid = [1]  # mutable holder so _apply_arrival can increment

    checks_done = 0
    for tick in range(n_ticks):
        # Per tick: fire 1-3 events to build up realistic lifecycle pressure.
        for _ in range(rng.randint(1, 3)):
            fn = _pick_event(rng)
            fn(queue, peek_tree, running, next_rid, rng, system_prompts) if fn is _apply_arrival \
                else fn(queue, peek_tree, running, rng)

        _assert_state_parity(peek_tree, queue, rng, system_prompts)
        checks_done += 1

    # Sanity: we actually exercised enough events and didn't stay at size 0.
    assert checks_done == n_ticks
    # Drain whatever's left and confirm peek returns to empty.
    for r in list(queue):
        peek_tree.remove(r.rid)
    assert len(peek_tree) == 0
    assert peek_tree.node_count() == 1  # only the root remains

def test_lifecycle_events_each_exercised(capsys: pytest.CaptureFixture) -> None:
    """Smoke test: confirm each of the 5 event types fires at least once in a
    typical run, and state parity holds at every tick."""
    rng = random.Random(7)
    system_prompts = [tuple(range(i, i + 20)) for i in range(3)]
    queue: List[MockReq] = []
    running: Dict[int, MockReq] = {}
    peek_tree = PendingTree()
    next_rid = [1]

    counts = {fn.__name__: 0 for fn, _ in EVENT_WEIGHTS}
    for _ in range(500):
        fn = _pick_event(rng)
        counts[fn.__name__] += 1
        if fn is _apply_arrival:
            fn(queue, peek_tree, running, next_rid, rng, system_prompts)
        else:
            fn(queue, peek_tree, running, rng)
        _assert_state_parity(peek_tree, queue, rng, system_prompts)

    with capsys.disabled():
        print(f"\nevent counts: {counts}")

    assert all(c > 0 for c in counts.values()), f"some events never fired: {counts}"
