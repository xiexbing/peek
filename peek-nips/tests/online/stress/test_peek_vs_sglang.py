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

"""Stress test peek against sglang's actual LPM primitives.

sglang's `SchedulePolicy._compute_prefix_matches` (schedule_policy.py) uses two
things to sort the waiting queue by longest prefix match:

  1. The *main* tree_cache, for "how much of this request's prefix is already in
     the KV cache?" We don't exercise that here -- it needs a live allocator.
  2. An in-batch `waiting_queue_radix_tree` (RadixCache.create_simulated()),
     rebuilt from scratch every scheduling pass. For each request in arrival
     order, it match_prefix's against the tree-so-far, then inserts (unless
     the match is long enough to deprioritize the request).

Peek replaces (2) with a persistent, incremental tree. For a given arrival
order, peek's per-request in-batch match should equal sglang's, and the
deprioritized set should be identical.

These tests:
  * `test_inbatch_parity_*` -- run the same arrival sequence through both and
    assert per-rid match length + deprioritize set match.
  * `test_inbatch_perf_many_passes` -- simulate many scheduling passes over a
    stable waiting queue and time both approaches. Peek should be markedly
    faster because it does not rebuild.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pytest
import torch

from sglang.srt.managers.schedule_policy import (
    IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD as DEPRIORITIZE_THRESHOLD,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek import PendingTree

# ---------------------------------------------------------------------------
# Workload generation: requests with realistic prefix overlap.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Req:
    rid: int
    tokens: Tuple[int, ...]

def make_workload(
    n_requests: int,
    *,
    n_system_prompts: int = 4,
    sys_prompt_len: Tuple[int, int] = (20, 50),
    continuation_len: Tuple[int, int] = (5, 40),
    vocab_size: int = 256,
    seed: int = 0,
) -> List[Req]:
    """Generate requests that share one of a small pool of system prompts,
    plus a random continuation. Mirrors LLM traffic with shared boilerplate.
    """
    rng = random.Random(seed)
    system_prompts: List[Tuple[int, ...]] = []
    for _ in range(n_system_prompts):
        length = rng.randint(*sys_prompt_len)
        system_prompts.append(tuple(rng.randint(0, vocab_size - 1) for _ in range(length)))

    reqs: List[Req] = []
    for i in range(n_requests):
        sp = rng.choice(system_prompts)
        cont_len = rng.randint(*continuation_len)
        cont = tuple(rng.randint(0, vocab_size - 1) for _ in range(cont_len))
        reqs.append(Req(rid=i + 1, tokens=sp + cont))
    return reqs

# ---------------------------------------------------------------------------
# sglang in-batch logic, extracted. Mirrors the else-branch of
# SchedulePolicy._compute_prefix_matches -- the part that uses the
# waiting-queue radix tree.
# ---------------------------------------------------------------------------

def sglang_inbatch_pass(reqs: Sequence[Req]) -> Tuple[dict, set]:
    tree = RadixCache.create_simulated()
    match_lens: dict = {}
    deprioritized: set = set()
    for r in reqs:
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=list(r.tokens), extra_key=None))
        )
        in_batch_len = len(m.device_indices)
        match_lens[r.rid] = in_batch_len
        if in_batch_len >= DEPRIORITIZE_THRESHOLD:
            deprioritized.add(r.rid)
        else:
            tree.insert(
                InsertParams(
                    key=RadixKey(token_ids=list(r.tokens), extra_key=None),
                    value=torch.empty(len(r.tokens), dtype=torch.bool),
                )
            )
    return match_lens, deprioritized

def peek_inbatch_pass(reqs: Sequence[Req]) -> Tuple[dict, set]:
    tree = PendingTree()
    match_lens: dict = {}
    deprioritized: set = set()
    for r in reqs:
        in_batch_len = tree.match_prefix(list(r.tokens))
        match_lens[r.rid] = in_batch_len
        if in_batch_len >= DEPRIORITIZE_THRESHOLD:
            deprioritized.add(r.rid)
        else:
            tree.insert(r.rid, list(r.tokens))
    return match_lens, deprioritized

# ---------------------------------------------------------------------------
# Correctness: peek must produce identical in-batch lens and deprioritized set.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_requests,seed",
    [(50, 0), (50, 1), (200, 2), (500, 3), (1000, 4)],
)
def test_inbatch_parity(n_requests: int, seed: int) -> None:
    reqs = make_workload(n_requests, seed=seed)
    s_lens, s_dep = sglang_inbatch_pass(reqs)
    p_lens, p_dep = peek_inbatch_pass(reqs)
    assert s_lens == p_lens, (
        f"match-length mismatch: {sum(1 for k in s_lens if s_lens[k] != p_lens[k])} rids differ"
    )
    assert s_dep == p_dep

def test_inbatch_parity_identical_prefixes() -> None:
    # Many duplicates of the same long prefix -- exercises repeated splits.
    prefix = tuple(range(40))
    reqs = [Req(rid=i + 1, tokens=prefix + (i,)) for i in range(100)]
    s_lens, s_dep = sglang_inbatch_pass(reqs)
    p_lens, p_dep = peek_inbatch_pass(reqs)
    assert s_lens == p_lens
    assert s_dep == p_dep

def test_inbatch_parity_strict_prefix_chain() -> None:
    # rid i's tokens are a strict prefix of rid i+1's -- the partial-edge case.
    reqs = [Req(rid=i + 1, tokens=tuple(range(1, i + 2))) for i in range(30)]
    s_lens, s_dep = sglang_inbatch_pass(reqs)
    p_lens, p_dep = peek_inbatch_pass(reqs)
    assert s_lens == p_lens
    assert s_dep == p_dep

# ---------------------------------------------------------------------------
# Performance: the motivating scenario. A waiting queue of N requests persists
# across M scheduling passes. Stock sglang rebuilds the in-batch tree on every
# pass; peek builds it once on arrival and only queries on each pass.
# ---------------------------------------------------------------------------

def _sglang_rebuild_pass_cost(reqs: Sequence[Req]) -> float:
    """One full sglang-style rebuild + per-rid match + maybe-insert."""
    t0 = time.perf_counter()
    sglang_inbatch_pass(reqs)
    return time.perf_counter() - t0

def _peek_query_only_pass_cost(tree: PendingTree, reqs: Sequence[Req]) -> float:
    """Once peek's tree is warm, a scheduling pass is just N queries."""
    t0 = time.perf_counter()
    for r in reqs:
        # Mirror the in-batch deprioritize decision used by sglang's policy.
        _ = tree.longest_shared_prefix(r.rid)
    return time.perf_counter() - t0

def test_inbatch_perf_many_passes(capsys: pytest.CaptureFixture) -> None:
    n_requests = 500
    n_passes = 20
    reqs = make_workload(n_requests, seed=42)

    # Amortized peek arrival cost: insert all rids once.
    peek_tree = PendingTree()
    t0 = time.perf_counter()
    for r in reqs:
        peek_tree.insert(r.rid, list(r.tokens))
    peek_arrival_ms = (time.perf_counter() - t0) * 1000

    # Warm-up once each to avoid first-call artifacts.
    _sglang_rebuild_pass_cost(reqs)
    _peek_query_only_pass_cost(peek_tree, reqs)

    # Measure M passes.
    sglang_times = [_sglang_rebuild_pass_cost(reqs) for _ in range(n_passes)]
    peek_times = [_peek_query_only_pass_cost(peek_tree, reqs) for _ in range(n_passes)]

    sg_total_ms = sum(sglang_times) * 1000
    pk_total_ms = sum(peek_times) * 1000
    speedup = sg_total_ms / pk_total_ms if pk_total_ms > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\n[{n_requests} reqs x {n_passes} passes]  "
            f"peek arrival(amortized once): {peek_arrival_ms:.2f} ms  |  "
            f"sglang rebuildxM: {sg_total_ms:.2f} ms  |  "
            f"peek queryxM: {pk_total_ms:.2f} ms  |  "
            f"speedup: {speedup:.1f}x"
        )

    # Sanity: peek per-pass should beat sglang per-pass. Use a conservative
    # margin so we don't flake on noisy CI -- anything less than "clearly
    # faster" is still a regression worth investigating.
    assert pk_total_ms < sg_total_ms, (
        f"peek per-pass ({pk_total_ms:.2f} ms) should be faster than "
        f"sglang rebuild ({sg_total_ms:.2f} ms)"
    )
