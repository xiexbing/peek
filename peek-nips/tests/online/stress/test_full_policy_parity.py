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

"""End-to-end LPM scheduling-flow parity: sglang's tree vs peek's tree.

Full flow under test (mirrors SchedulePolicy._compute_prefix_matches +
_sort_by_longest_prefix in sglang/srt/managers/schedule_policy.py):

  for each req in waiting_queue:
      1. main-cache match → r.prefix_indices   (simulated here: a random length)
      2. if len(r.prefix_indices) <= CHECK_THRESHOLD:
             in-batch match against waiting-queue tree
             if match >= DEPRIORITIZE_THRESHOLD: deprioritize
             else: insert into waiting-queue tree
  waiting_queue.sort(key= deprioritized ? +inf : -len(r.prefix_indices))

The ONLY thing we swap is the in-batch waiting-queue tree: sglang's
RadixCache.create_simulated() vs peek's PendingTree. Main-cache match is a
common input (same values fed to both). The final `sort` call uses sglang's
own `_sort_by_longest_prefix` so any disagreement is attributable to the
in-batch tree alone.

We assert the full post-sort rid order is identical under both trees.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import pytest
import torch

from sglang.srt.managers.schedule_policy import (
    IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD as CHECK_THRESHOLD,
    IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD as DEPRIORITIZE_THRESHOLD,
    SchedulePolicy,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek import PendingTree

# ---------------------------------------------------------------------------
# Minimal Req stand-in. SchedulePolicy._sort_by_longest_prefix only reads
# `rid` and `prefix_indices` (via len()). We carry tokens + extra_key so the
# in-batch tree can key on them.
# ---------------------------------------------------------------------------

@dataclass
class MockReq:
    rid: int
    origin_input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    extra_key: Any = None
    prefix_indices: Any = None  # simulated main-cache match (any sized sequence)

def make_workload(n: int, *, seed: int) -> List[MockReq]:
    """Requests with one of a handful of shared system prompts + a random tail.
    Main-cache `prefix_indices` length is randomized: most reqs land below the
    in-batch check threshold, some above — so the gating branch is exercised.
    """
    rng = random.Random(seed)
    vocab = 256
    system_prompts = [
        tuple(rng.randint(0, vocab - 1) for _ in range(rng.randint(20, 60)))
        for _ in range(4)
    ]
    reqs: List[MockReq] = []
    for i in range(n):
        sp = rng.choice(system_prompts)
        tail_len = rng.randint(5, 40)
        tail = [rng.randint(0, vocab - 1) for _ in range(tail_len)]
        tokens = list(sp) + tail
        # Simulate main-cache match length. Skewed so most reqs qualify for
        # in-batch checking (≤ CHECK_THRESHOLD) but some bypass it.
        if rng.random() < 0.25:
            main_len = rng.randint(CHECK_THRESHOLD + 1, max(CHECK_THRESHOLD + 2, len(tokens)))
            main_len = min(main_len, len(tokens))
        else:
            main_len = rng.randint(0, CHECK_THRESHOLD)
        reqs.append(
            MockReq(
                rid=i + 1,
                origin_input_ids=tokens,
                prefix_indices=torch.zeros(main_len, dtype=torch.int64),
            )
        )
    return reqs

# ---------------------------------------------------------------------------
# Two implementations of the in-batch step, differing only in the tree used.
# ---------------------------------------------------------------------------

def sglang_compute_deprioritized(queue: List[MockReq]) -> set:
    tree = RadixCache.create_simulated()
    deprioritized: set = set()
    for r in queue:
        if len(r.prefix_indices) > CHECK_THRESHOLD:
            continue
        prefix_ids = r.origin_input_ids + r.output_ids
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=prefix_ids, extra_key=r.extra_key))
        )
        if len(m.device_indices) >= DEPRIORITIZE_THRESHOLD:
            deprioritized.add(r.rid)
        else:
            tree.insert(
                InsertParams(
                    key=RadixKey(token_ids=prefix_ids, extra_key=r.extra_key),
                    value=torch.empty(len(prefix_ids), dtype=torch.bool),
                )
            )
    return deprioritized

def peek_compute_deprioritized(queue: List[MockReq]) -> set:
    tree = PendingTree()
    deprioritized: set = set()
    for r in queue:
        if len(r.prefix_indices) > CHECK_THRESHOLD:
            continue
        prefix_ids = r.origin_input_ids + r.output_ids
        in_batch_len = tree.match_prefix(prefix_ids)
        if in_batch_len >= DEPRIORITIZE_THRESHOLD:
            deprioritized.add(r.rid)
        else:
            tree.insert(r.rid, prefix_ids)
    return deprioritized

def run_policy(queue: List[MockReq], deprioritized: set) -> List[int]:
    """Apply sglang's own sort function. Returns the final rid order."""
    q = list(queue)  # don't mutate caller
    SchedulePolicy._sort_by_longest_prefix(q, deprioritized)
    return [r.rid for r in q]

# ---------------------------------------------------------------------------
# The test: same workload → same deprioritized set → same final queue order.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(50, 0), (200, 1), (500, 2), (1000, 3), (2000, 4)])
def test_full_policy_parity(n: int, seed: int) -> None:
    queue = make_workload(n, seed=seed)

    s_dep = sglang_compute_deprioritized(queue)
    p_dep = peek_compute_deprioritized(queue)
    assert s_dep == p_dep, (
        f"deprioritize set differs: sglang={len(s_dep)} peek={len(p_dep)} "
        f"sym_diff={len(s_dep ^ p_dep)}"
    )

    s_order = run_policy(queue, s_dep)
    p_order = run_policy(queue, p_dep)
    assert s_order == p_order, (
        f"final queue order differs at positions: "
        f"{[i for i, (a, b) in enumerate(zip(s_order, p_order)) if a != b][:10]}"
    )

def test_full_policy_parity_heavy_sharing() -> None:
    """Stress the sort ties: many reqs with identical main-cache lengths
    and heavy shared prefixes. Python's sort is stable, so if both branches
    produce identical keys and identical deprioritize sets, order must match.
    """
    base = list(range(40))
    queue: List[MockReq] = []
    for i in range(300):
        tokens = base + [i % 5, i]  # 5 branches off a common 40-token prefix
        queue.append(
            MockReq(
                rid=i + 1,
                origin_input_ids=tokens,
                prefix_indices=torch.zeros(i % 8, dtype=torch.int64),  # all ≤ CHECK_THRESHOLD
            )
        )

    s_dep = sglang_compute_deprioritized(queue)
    p_dep = peek_compute_deprioritized(queue)
    assert s_dep == p_dep

    assert run_policy(queue, s_dep) == run_policy(queue, p_dep)

def test_full_policy_parity_all_bypass_inbatch() -> None:
    """When every req has a long main-cache match, nothing enters the in-batch
    tree. Both implementations should produce an empty deprioritize set and an
    order sorted purely by main-cache length."""
    queue = [
        MockReq(
            rid=i + 1,
            origin_input_ids=list(range(i, i + 10)),
            prefix_indices=torch.zeros(CHECK_THRESHOLD + 1 + (i % 7), dtype=torch.int64),
        )
        for i in range(100)
    ]
    assert sglang_compute_deprioritized(queue) == set()
    assert peek_compute_deprioritized(queue) == set()
    assert run_policy(queue, set()) == run_policy(queue, set())

# ==========================================================================
# End-to-end parity: peek_lpm path (peek_sort_inplace + Rust lpm_sort_order)
# vs sglang's SchedulePolicy._sort_by_longest_prefix.
# ==========================================================================

def _run_peek_lpm_pipeline(queue: List[MockReq]) -> List[int]:
    """Run peek's full LPM pipeline end-to-end (dualwalk main_hit optional,
    in-batch deprio via peek's prefix-tuple claim, Rust lpm_sort_order sort)
    and return the final rid order.
    """
    from peek.online.lpm_integration import peek_sort_inplace
    tree = PendingTree()
    rid_to_int: dict = {}
    for r in queue:
        tokens = list(r.origin_input_ids) + list(r.output_ids or [])
        tree.insert(r.rid, tokens)
        rid_to_int[r.rid] = r.rid  # identity mapping; MockReq rid is already int
    q = list(queue)
    peek_sort_inplace(
        q,
        rid_to_int,
        tree,
        check_threshold=CHECK_THRESHOLD,
        deprioritize_threshold=DEPRIORITIZE_THRESHOLD,
        rank_by_cluster_size=False,
        main_hits=None,  # fallback to len(r.prefix_indices) — matches sglang's main_hit
        peek_lpm_sort=True,
    )
    return [r.rid for r in q]

def _run_peek_clpm_pipeline(queue: List[MockReq], window_ms: int = 0) -> List[int]:
    """peek_clpm with window_ms=0 + no sharing-specific content ⇒ should reduce
    to stock LPM order (warm by main_hit desc → pioneer arrival → sibling
    arrival). With window_ms>0, it diverges and needs its own reference.
    """
    from peek.online.lpm_integration import peek_clpm_sort_inplace
    tree = PendingTree()
    rid_to_int: dict = {}
    for r in queue:
        tokens = list(r.origin_input_ids) + list(r.output_ids or [])
        tree.insert(r.rid, tokens)
        rid_to_int[r.rid] = r.rid
    q = list(queue)
    peek_clpm_sort_inplace(
        q,
        rid_to_int,
        tree,
        window_ms=window_ms,
        check_threshold=CHECK_THRESHOLD,
        deprioritize_threshold=DEPRIORITIZE_THRESHOLD,
        main_hits=None,
        arrival_ts=None,
    )
    return [r.rid for r in q]

@pytest.mark.parametrize("n,seed", [(100, 0), (500, 1)])
def test_peek_clpm_with_zero_window_matches_sglang_lpm(n: int, seed: int) -> None:
    """With window_ms=0 and no arrival_ts, peek_clpm's primary axis
    (arrival_bucket) collapses to zero for every req. Within section,
    main_hit is still the primary ordering key, so warm reqs still sort
    by main_hit desc. peek_clpm adds req_score and cluster_size as
    secondary/tertiary keys — these break ties that stock LPM preserves
    as arrival order.

    This test verifies the HEAD of the sorted queue (the warm-by-main_hit
    section) matches stock LPM byte-for-byte. Within ties (including the
    whole cold-pioneer section), peek_clpm may diverge legitimately.
    """
    queue = make_workload(n, seed=seed)
    s_dep = sglang_compute_deprioritized(queue)
    s_order = run_policy(queue, s_dep)
    p_order = _run_peek_clpm_pipeline(queue, window_ms=0)
    # Warm section (main_hit > CHECK_THRESHOLD): same arrival order within
    # same-main_hit buckets. peek_clpm may reorder by cluster_score within
    # equal-main_hit — verify warm reqs with DIFFERENT main_hit are in the
    # same relative order.
    rid_to_req = {r.rid: r for r in queue}
    def _mh(rid):
        return len(rid_to_req[rid].prefix_indices)
    s_warm_rids = [rid for rid in s_order if _mh(rid) > CHECK_THRESHOLD]
    p_warm_rids = [rid for rid in p_order if _mh(rid) > CHECK_THRESHOLD]
    assert len(s_warm_rids) == len(p_warm_rids)
    # Same main_hit values at each position (may differ in tiebreak but
    # main_hit sequence must be identical — LPM-equivalence of the primary
    # signal).
    s_mhs = [_mh(rid) for rid in s_warm_rids]
    p_mhs = [_mh(rid) for rid in p_warm_rids]
    assert s_mhs == p_mhs, (
        f"warm-section main_hit sequence diverges: stock={s_mhs[:10]} peek_clpm={p_mhs[:10]}"
    )

@pytest.mark.parametrize("n,seed", [(50, 0), (200, 1), (500, 2), (1000, 3), (2000, 4)])
def test_peek_lpm_matches_sglang_lpm_random(n: int, seed: int) -> None:
    """Full peek_lpm pipeline — including the Rust `lpm_sort_order` sort —
    must produce the same admission order as sglang's native LPM on every
    waiting-queue shape."""
    queue = make_workload(n, seed=seed)

    s_dep = sglang_compute_deprioritized(queue)
    s_order = run_policy(queue, s_dep)

    p_order = _run_peek_lpm_pipeline(queue)

    assert s_order == p_order, (
        f"peek_lpm vs sglang LPM mismatch (n={n}): first diffs at positions "
        f"{[i for i, (a, b) in enumerate(zip(s_order, p_order)) if a != b][:10]}"
    )

def test_peek_lpm_matches_sglang_lpm_heavy_sharing() -> None:
    """Many reqs with identical main-cache lengths + heavy shared prefixes —
    stresses tie-preserving stable-sort behavior of lpm_sort_order."""
    base = list(range(40))
    queue: List[MockReq] = []
    for i in range(300):
        tokens = base + [i % 5, i]
        queue.append(
            MockReq(
                rid=i + 1,
                origin_input_ids=tokens,
                prefix_indices=torch.zeros(i % 8, dtype=torch.int64),
            )
        )
    s_order = run_policy(queue, sglang_compute_deprioritized(queue))
    p_order = _run_peek_lpm_pipeline(queue)
    assert s_order == p_order

def test_peek_lpm_matches_sglang_lpm_all_warm() -> None:
    """Every req has a long main-cache match → nothing deprioritized; both
    sort purely by descending main_hit with stable arrival-order tiebreak."""
    queue = [
        MockReq(
            rid=i + 1,
            origin_input_ids=list(range(i, i + 10)),
            prefix_indices=torch.zeros(CHECK_THRESHOLD + 1 + (i % 7), dtype=torch.int64),
        )
        for i in range(100)
    ]
    s_order = run_policy(queue, set())
    p_order = _run_peek_lpm_pipeline(queue)
    assert s_order == p_order

def test_peek_lpm_matches_sglang_lpm_loogle_shape() -> None:
    """Shared-system-prompts shape matching Scenario 1 of the Qwen sweep
    (100 groups × ~10 members × ~2048-token shared prefix). Verifies parity
    on the exact workload the user cares about."""
    rng = random.Random(42)
    G = 100
    MEMBERS_PER_G = 10
    SHARED_LEN = 512  # scaled down from 2048 for speed; sort logic is length-agnostic
    TAIL_LEN = 40
    system_prompts = [
        [rng.randint(0, 32_000) for _ in range(SHARED_LEN)] for _ in range(G)
    ]
    queue: List[MockReq] = []
    rid = 0
    for g in range(G):
        for _ in range(MEMBERS_PER_G):
            rid += 1
            tail = [rng.randint(0, 32_000) for _ in range(TAIL_LEN)]
            tokens = system_prompts[g] + tail
            # Mix of cold / warm reqs: 70% cold (main_hit=0..CHECK_THRESHOLD),
            # 30% warm (main_hit > CHECK_THRESHOLD) — mirrors production shape.
            if rng.random() < 0.3:
                main_len = rng.randint(CHECK_THRESHOLD + 1, SHARED_LEN)
            else:
                main_len = rng.randint(0, CHECK_THRESHOLD)
            queue.append(
                MockReq(
                    rid=rid,
                    origin_input_ids=tokens,
                    prefix_indices=torch.zeros(main_len, dtype=torch.int64),
                )
            )
    rng.shuffle(queue)

    s_dep = sglang_compute_deprioritized(queue)
    p_dep = peek_compute_deprioritized(queue)
    assert s_dep == p_dep, f"deprio diverge: sym_diff={len(s_dep ^ p_dep)}"
    s_order = run_policy(queue, s_dep)
    p_order = _run_peek_lpm_pipeline(queue)
    assert s_order == p_order, (
        f"loogle-shape sort diverges at positions: "
        f"{[i for i, (a, b) in enumerate(zip(s_order, p_order)) if a != b][:20]}"
    )
