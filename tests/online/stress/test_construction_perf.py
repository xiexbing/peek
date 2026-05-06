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

"""Tree construction timing: sglang RadixCache vs peek PendingTree.

Measures the raw cost of *building* each tree from N pending requests. This is
the per-pass cost sglang pays today — it resets and rebuilds its in-batch
waiting_queue_radix_tree every scheduling pass (schedule_policy.py:190).

Two angles:

  1. `test_construction_one_shot` — build from zero at a range of queue sizes.
     Reports per-request and total build time for both sides.

  2. `test_construction_amortized_over_lifecycle` — simulate a realistic
     arrival/schedule lifecycle (M ticks, steady-state queue). Compare
     cumulative construction cost: sglang rebuilds on every tick; peek only
     pays the per-arrival insert and per-schedule remove.

Both tests print timing tables under `-s` and assert peek is faster.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek import PendingTree

@dataclass(frozen=True)
class Req:
    rid: int
    tokens: Tuple[int, ...]

def _make_workload(n: int, seed: int) -> List[Req]:
    rng = random.Random(seed)
    vocab = 256
    system_prompts = [
        tuple(rng.randint(0, vocab - 1) for _ in range(rng.randint(20, 60)))
        for _ in range(4)
    ]
    reqs: List[Req] = []
    for i in range(n):
        sp = rng.choice(system_prompts)
        tail = tuple(rng.randint(0, vocab - 1) for _ in range(rng.randint(5, 40)))
        reqs.append(Req(rid=i + 1, tokens=sp + tail))
    return reqs

def _build_sglang(reqs: List[Req]) -> float:
    t0 = time.perf_counter()
    tree = RadixCache.create_simulated()
    for r in reqs:
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=list(r.tokens), extra_key=None),
                value=torch.empty(len(r.tokens), dtype=torch.bool),
            )
        )
    return time.perf_counter() - t0

def _build_peek(reqs: List[Req]) -> float:
    t0 = time.perf_counter()
    tree = PendingTree()
    for r in reqs:
        tree.insert(r.rid, list(r.tokens))
    return time.perf_counter() - t0

def _repeat_min(fn, reqs, n_trials: int) -> float:
    """Min-of-N to filter out GC / noise spikes. Returns seconds."""
    return min(fn(reqs) for _ in range(n_trials))

@pytest.mark.parametrize("n", [100, 500, 1000, 2000])
def test_construction_one_shot(n: int, capsys: pytest.CaptureFixture) -> None:
    reqs = _make_workload(n, seed=n)
    n_trials = 5
    # Warm-up.
    _build_sglang(reqs)
    _build_peek(reqs)

    sg_s = _repeat_min(_build_sglang, reqs, n_trials)
    pk_s = _repeat_min(_build_peek, reqs, n_trials)
    speedup = sg_s / pk_s if pk_s > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\n  N={n:>5} | sglang: {sg_s * 1000:7.2f} ms  "
            f"({sg_s * 1e6 / n:6.1f} µs/req) | "
            f"peek: {pk_s * 1000:6.2f} ms  "
            f"({pk_s * 1e6 / n:5.1f} µs/req) | "
            f"{speedup:5.1f}× faster"
        )

    assert pk_s < sg_s, f"peek should be faster: peek={pk_s:.4f}s sglang={sg_s:.4f}s"

def test_construction_amortized_over_lifecycle(capsys: pytest.CaptureFixture) -> None:
    """Steady-state queue of ~N, over M scheduling ticks. Each tick:
    some arrivals, some scheduled (removed). Compare:
      * sglang: rebuild entire tree at each tick → M * N_current inserts.
      * peek:   insert on arrival, remove on schedule → 2 × (total arrivals).

    Peek's cumulative tree-construction cost should be far lower.
    """
    rng = random.Random(1234)
    vocab = 256
    system_prompts = [
        tuple(rng.randint(0, vocab - 1) for _ in range(rng.randint(20, 50)))
        for _ in range(4)
    ]

    target_size = 400
    n_ticks = 50
    rid_counter = 0

    # Pre-generate all events so both sides see identical workloads.
    queue_state_per_tick: List[List[Req]] = []
    arrivals_per_tick: List[List[Req]] = []
    schedules_per_tick: List[List[int]] = []
    current: List[Req] = []

    for _ in range(n_ticks):
        # Arrivals: fill toward target_size + some churn.
        n_arrivals = max(1, target_size - len(current) + rng.randint(10, 30))
        arrivals: List[Req] = []
        for _ in range(n_arrivals):
            rid_counter += 1
            sp = rng.choice(system_prompts)
            tail = tuple(rng.randint(0, vocab - 1) for _ in range(rng.randint(5, 40)))
            arrivals.append(Req(rid=rid_counter, tokens=sp + tail))
        arrivals_per_tick.append(arrivals)
        current.extend(arrivals)

        # Schedules: remove a batch of 8–32 at random.
        n_sched = min(rng.randint(8, 32), len(current))
        scheduled = rng.sample(current, n_sched)
        sched_rids = {r.rid for r in scheduled}
        schedules_per_tick.append(list(sched_rids))
        current = [r for r in current if r.rid not in sched_rids]
        queue_state_per_tick.append(list(current))

    # --- sglang: rebuild from scratch at each tick ---
    t0 = time.perf_counter()
    for state in queue_state_per_tick:
        tree = RadixCache.create_simulated()
        for r in state:
            tree.insert(
                InsertParams(
                    key=RadixKey(token_ids=list(r.tokens), extra_key=None),
                    value=torch.empty(len(r.tokens), dtype=torch.bool),
                )
            )
    sglang_s = time.perf_counter() - t0

    # --- peek: incremental insert on arrival, remove on schedule ---
    peek_tree = PendingTree()
    t0 = time.perf_counter()
    for arrivals, sched_rids in zip(arrivals_per_tick, schedules_per_tick):
        for r in arrivals:
            peek_tree.insert(r.rid, list(r.tokens))
        for rid in sched_rids:
            peek_tree.discard(rid)
    peek_s = time.perf_counter() - t0

    avg_queue = sum(len(s) for s in queue_state_per_tick) / n_ticks
    speedup = sglang_s / peek_s if peek_s > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\n  lifecycle: {n_ticks} ticks, avg queue ≈ {avg_queue:.0f}, "
            f"{rid_counter} total arrivals"
        )
        print(
            f"  sglang rebuild-per-tick: {sglang_s * 1000:8.2f} ms  "
            f"(~{sglang_s * 1000 / n_ticks:.2f} ms/tick)"
        )
        print(
            f"  peek    incremental   : {peek_s * 1000:8.2f} ms  "
            f"(~{peek_s * 1e6 / rid_counter:.2f} µs/op)"
        )
        print(f"  speedup: {speedup:.1f}×")

    assert peek_s < sglang_s, (
        f"peek lifecycle cost ({peek_s:.4f}s) should be less than "
        f"sglang rebuild-per-tick ({sglang_s:.4f}s)"
    )
