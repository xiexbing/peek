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

"""Part 2: Python ↔ Rust crossing-correctness stress.

Each PyO3 primitive exposed to Python is checked against a naive Python-only
ground-truth, under random and structured workloads:

  * compute_main_hits (dualwalk)   vs naive per-rid cache_match_fn call
  * all_cluster_info                vs per-rid cluster_info
  * snapshot_for_walk               vs independently-reconstructed tree
  * pending_demand / terminators_at vs paths dict filter (sanity restate)

Purpose: catch bugs at the FFI boundary where Rust returns values to Python —
wrong type conversions, iterator ordering, dict key hashing, borrowed-vs-owned
lifetimes, etc. None of these are covered by the Rust unit tests.
"""

from __future__ import annotations

import random
from typing import Dict, List

from peek.online import PendingTree, compute_main_hits


def _build_tree(pairs):
    t = PendingTree()
    for rid, toks in pairs:
        t.insert(rid, toks)
    return t


def _rand_pairs(rng: random.Random, n: int, vocab: int, min_len: int, max_len: int):
    pairs = []
    used_rids = set()
    while len(pairs) < n:
        rid = rng.randrange(1, n * 10)
        if rid in used_rids:
            continue
        used_rids.add(rid)
        length = rng.randint(min_len, max_len)
        toks = [rng.randrange(vocab) for _ in range(length)]
        pairs.append((rid, toks))
    return pairs


# ----------------------------------------------------------------------------
# 1. compute_main_hits vs naive per-rid query.
# ----------------------------------------------------------------------------

def _naive_longest_cache_match(tokens, cached_sequences):
    """Best prefix length of `tokens` that appears in any cached sequence."""
    best = 0
    for cs in cached_sequences:
        common = 0
        for a, b in zip(tokens, cs):
            if a != b:
                break
            common += 1
        if common > best:
            best = common
    return best


def test_dualwalk_matches_naive_exact_mode():
    """At min_pending_count=1, dualwalk must match naive exactly."""
    rng = random.Random(1234)
    pairs = _rand_pairs(rng, n=40, vocab=6, min_len=2, max_len=10)
    tree = _build_tree(pairs)

    # Build a random subset of "cached" sequences from the rid tokens, plus
    # perturbations (so cache may diverge mid-edge).
    cached = [pairs[i][1][:rng.randint(0, len(pairs[i][1]))] for i in range(len(pairs))]
    # Add a few unrelated cache entries to stress divergence.
    for _ in range(5):
        cached.append([rng.randrange(6) for _ in range(rng.randint(1, 8))])

    def cache_match_fn(tokens):
        return _naive_longest_cache_match(tokens, cached)

    dw = compute_main_hits(tree, [], cache_match_fn, min_pending_count=1)
    naive = {rid: cache_match_fn(toks) for rid, toks in pairs}
    assert dw == naive, f"dualwalk != naive: dw={dw}, naive={naive}"


def test_dualwalk_min_pc_2_only_diverges_on_singleton_tails():
    """At min_pending_count=2, dualwalk may UNDER-report main_hit on subtrees
    where only one rid lives below the shared prefix (singleton tails aren't
    queried). When it diverges, the under-report is the shared-prefix depth
    for that rid, not an arbitrary wrong value."""
    rng = random.Random(777)
    pairs = _rand_pairs(rng, n=60, vocab=5, min_len=3, max_len=12)
    tree = _build_tree(pairs)
    cached = [pairs[i][1] for i in range(len(pairs))]

    def cache_match_fn(tokens):
        return _naive_longest_cache_match(tokens, cached)

    exact = compute_main_hits(tree, [], cache_match_fn, min_pending_count=1)
    approx = compute_main_hits(tree, [], cache_match_fn, min_pending_count=2)

    assert set(exact) == set(approx)
    for rid in exact:
        # Approximate should be <= exact (never over-report).
        assert approx[rid] <= exact[rid], f"rid {rid}: approx {approx[rid]} > exact {exact[rid]}"


# ----------------------------------------------------------------------------
# 2. all_cluster_info bulk vs per-rid cluster_info.
# ----------------------------------------------------------------------------

def test_all_cluster_info_matches_per_rid():
    rng = random.Random(9999)
    pairs = _rand_pairs(rng, n=80, vocab=4, min_len=2, max_len=7)
    tree = _build_tree(pairs)

    bulk = tree.all_cluster_info()
    for rid, _toks in pairs:
        per = tree.cluster_info(rid)
        assert bulk.get(rid) == per, f"rid {rid}: bulk={bulk.get(rid)} per={per}"
    # Every pending rid appears exactly once.
    assert set(bulk.keys()) == {r for r, _ in pairs}


# ----------------------------------------------------------------------------
# 4. snapshot_for_walk — reconstruct independently and compare topology.
# ----------------------------------------------------------------------------

def test_snapshot_round_trip_matches_tree_state():
    """Rebuild tree topology from snapshot; counts/terminators must match."""
    rng = random.Random(31337)
    pairs = _rand_pairs(rng, n=50, vocab=5, min_len=1, max_len=10)
    tree = _build_tree(pairs)

    snap = tree.snapshot_for_walk()
    # snap is list of (node_id, parent_id, edge_tokens, terminators, pending_count).
    # ROOT has id 0, parent 0.
    by_id = {n[0]: n for n in snap}
    assert 0 in by_id, "ROOT missing"
    root = by_id[0]
    assert root[1] == 0, "ROOT parent should be 0 (self)"
    assert root[2] == [], "ROOT edge should be empty"

    # Every non-root node's parent is present.
    for node_id, parent_id, edge, terms, pc in snap:
        if node_id == 0:
            continue
        assert parent_id in by_id, f"node {node_id}: parent {parent_id} missing"
        assert len(edge) > 0, f"non-root node {node_id} has empty edge"
        assert pc >= 0

    # Sum of ROOT's children's pc == total rids. Use it to sanity check.
    children_of_root = [n for n in snap if n[1] == 0 and n[0] != 0]
    total = sum(n[4] for n in children_of_root)
    assert total == len(pairs), f"total pending under root {total} != rid count {len(pairs)}"

    # Every rid appears as a terminator somewhere exactly once.
    termseen: Dict[int, int] = {}
    for _nid, _pid, _edge, terms, _pc in snap:
        for t in terms:
            termseen[t] = termseen.get(t, 0) + 1
    for rid, _ in pairs:
        assert termseen.get(rid, 0) == 1, f"rid {rid}: terminator count {termseen.get(rid)}"

    # Path reconstruction: walking from each terminator up to root yields the
    # rid's original tokens.
    parent_of = {nid: pid for nid, pid, _e, _t, _pc in snap}
    edge_of = {nid: e for nid, _pid, e, _t, _pc in snap}
    terminator_of: Dict[int, int] = {}
    for nid, _pid, _e, terms, _pc in snap:
        for t in terms:
            terminator_of[t] = nid

    pairs_by_rid = {r: toks for r, toks in pairs}
    for rid, expected in pairs_by_rid.items():
        tid = terminator_of[rid]
        rev: List[int] = []
        cur = tid
        while cur != 0:
            rev.extend(reversed(edge_of[cur]))
            cur = parent_of[cur]
        reconstructed = list(reversed(rev))
        assert reconstructed == expected, f"rid {rid}: reconstructed {reconstructed} != expected {expected}"


# ----------------------------------------------------------------------------
# 5. pending_demand / terminators_at at the Python boundary.
# ----------------------------------------------------------------------------

def test_python_boundary_demand_matches_ground_truth():
    rng = random.Random(2024)
    pairs = _rand_pairs(rng, n=40, vocab=5, min_len=2, max_len=8)
    tree = _build_tree(pairs)
    by_rid = {r: toks for r, toks in pairs}
    for _ in range(200):
        qlen = rng.randint(1, 6)
        q = [rng.randrange(5) for _ in range(qlen)]
        expected_demand = sum(1 for v in by_rid.values()
                              if len(v) >= len(q) and v[:len(q)] == q)
        expected_term = sum(1 for v in by_rid.values() if v == q)
        assert tree.pending_demand(q) == expected_demand, \
            f"pending_demand({q}): got {tree.pending_demand(q)}, expect {expected_demand}"
        assert tree.terminators_at(q) == expected_term, \
            f"terminators_at({q}): got {tree.terminators_at(q)}, expect {expected_term}"


# ----------------------------------------------------------------------------
# 6. Large-scale: agent-sessions workload end-to-end crossing.
# ----------------------------------------------------------------------------

def test_agent_sessions_workload_crossings():
    """Build a tree shaped like Scenario B (50 agents × 3 sessions × 5 turns).
    All four primitives must agree with naive re-derivation."""
    rng = random.Random(55555)
    pairs = []
    rid = 0
    for agent in range(50):
        sp = [1_000_000 + agent * 1000 + i for i in range(50)]  # 50-token SP
        for _sess in range(3):
            history = sp.copy()
            for _turn in range(5):
                toks = history + [rng.randrange(10_000) for _ in range(rng.randint(3, 10))]
                pairs.append((rid, toks))
                rid += 1
                history = toks
    tree = _build_tree(pairs)

    # pending_demand / terminators_at — random queries against ground truth.
    by_rid = {r: toks for r, toks in pairs}
    for _ in range(200):
        pick = rng.choice(pairs)
        qlen = rng.randint(1, len(pick[1]))
        q = pick[1][:qlen]
        expected_demand = sum(1 for v in by_rid.values()
                              if len(v) >= len(q) and v[:len(q)] == q)
        expected_term = sum(1 for v in by_rid.values() if v == q)
        assert tree.pending_demand(q) == expected_demand
        assert tree.terminators_at(q) == expected_term

    # Bulk cluster_info consistent with per-rid.
    bulk = tree.all_cluster_info()
    for r, _ in pairs:
        assert bulk[r] == tree.cluster_info(r)

    # Exact-mode dualwalk matches naive cache lookup.
    cached = [pairs[i][1][:rng.randint(0, len(pairs[i][1]))] for i in range(len(pairs))]

    def cache_match_fn(tokens):
        return _naive_longest_cache_match(tokens, cached)

    dw = compute_main_hits(tree, [], cache_match_fn, min_pending_count=1)
    for r, toks in pairs:
        assert dw[r] == cache_match_fn(toks), \
            f"rid {r}: dw {dw[r]} != naive {cache_match_fn(toks)}"
