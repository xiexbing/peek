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

"""Unit tests for peek's scheduling integration."""

from dataclasses import dataclass, field
from typing import List, Optional

from peek import PendingTree
from peek.online.lpm_integration import compute_in_batch_deprioritize, peek_sort_inplace


@dataclass
class FakeReq:
    rid: str
    origin_input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    # sglang populates this as a list-like; we only read its length.
    prefix_indices: Optional[List[int]] = None


def _intern_map(rids):
    return {rid: i + 1 for i, rid in enumerate(rids)}


def test_singleton_never_deprioritized():
    """A solo req (no cluster siblings) must never be deprioritized."""
    tree = PendingTree()
    tree.insert(1, [10, 20, 30, 40, 50])
    reqs = [FakeReq(rid="a", origin_input_ids=[10, 20, 30, 40, 50], prefix_indices=[])]
    rid_map = _intern_map(["a"])
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    assert dep == set()


def test_cold_cluster_pioneer_kept_siblings_deprioritized():
    """A cold cluster with shared prefix >= 32 tokens: first req is pioneer,
    rest are deprioritized."""
    tree = PendingTree()
    shared = list(range(100, 140))  # 40 shared tokens
    for i, tail in enumerate([[1], [2], [3]]):
        tree.insert(i + 1, shared + tail)
    reqs = [
        FakeReq(rid=str(i + 1), origin_input_ids=shared + [i + 1], prefix_indices=[])
        for i in range(3)
    ]
    rid_map = _intern_map(["1", "2", "3"])
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    # First in queue order ("1") is the pioneer — not deprioritized.
    assert "1" not in dep
    # The other two are deprioritized.
    assert dep == {"2", "3"}


def test_warm_reqs_skip_in_batch_check():
    """Reqs whose main_hit > check_threshold are already warm — skip the
    in-batch deprioritize check even if the cluster is large."""
    tree = PendingTree()
    shared = list(range(100, 140))
    for i, tail in enumerate([[1], [2], [3]]):
        tree.insert(i + 1, shared + tail)
    # prefix_indices length = 33 → above the 32-token check threshold.
    reqs = [
        FakeReq(
            rid=str(i + 1),
            origin_input_ids=shared + [i + 1],
            prefix_indices=list(range(33)),
        )
        for i in range(3)
    ]
    rid_map = _intern_map(["1", "2", "3"])
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    assert dep == set()


def test_shallow_cluster_below_deprioritize_threshold():
    """A cluster with shared prefix depth < deprioritize_threshold does not
    trigger deprioritize — the sharing is too shallow to be worth it."""
    tree = PendingTree()
    shared = list(range(100, 110))  # only 10 shared tokens — below the 32 threshold
    for i, tail in enumerate([[1], [2], [3]]):
        tree.insert(i + 1, shared + tail)
    reqs = [
        FakeReq(rid=str(i + 1), origin_input_ids=shared + [i + 1], prefix_indices=[])
        for i in range(3)
    ]
    rid_map = _intern_map(["1", "2", "3"])
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    assert dep == set()


def test_multiple_clusters_independent_pioneers():
    """Each cluster's first req in queue order is the pioneer, independent
    of which cluster iterated first."""
    tree = PendingTree()
    # Cluster A — rids 1,2 share a 40-token prefix starting at token 100.
    for i, tail in enumerate([[1], [2]]):
        tree.insert(i + 1, list(range(100, 140)) + tail)
    # Cluster B — rids 3,4 share a different 40-token prefix starting at 200.
    for i, tail in enumerate([[3], [4]]):
        tree.insert(i + 3, list(range(200, 240)) + tail)

    # Interleave queue order: B3, A1, B4, A2
    reqs = [
        FakeReq(rid="3", origin_input_ids=list(range(200, 240)) + [3], prefix_indices=[]),
        FakeReq(rid="1", origin_input_ids=list(range(100, 140)) + [1], prefix_indices=[]),
        FakeReq(rid="4", origin_input_ids=list(range(200, 240)) + [4], prefix_indices=[]),
        FakeReq(rid="2", origin_input_ids=list(range(100, 140)) + [2], prefix_indices=[]),
    ]
    rid_map = _intern_map(["1", "2", "3", "4"])
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    # Pioneers are the first-seen members: "3" (cluster B) and "1" (cluster A).
    assert "3" not in dep
    assert "1" not in dep
    # Siblings are deprioritized.
    assert dep == {"4", "2"}


def test_unknown_rid_deprio_by_token_prefix():
    """LPM-byte-exact deprio is purely token-prefix based — rid_map membership
    doesn't gate it. Req 99 shares the 40-token prefix with 1 and 2, so it
    gets deprioritized just like 2. Matches sglang's aux-tree semantics."""
    tree = PendingTree()
    shared = list(range(100, 140))
    tree.insert(1, shared + [1])
    tree.insert(2, shared + [2])
    reqs = [
        FakeReq(rid="1", origin_input_ids=shared + [1], prefix_indices=[]),
        FakeReq(rid="2", origin_input_ids=shared + [2], prefix_indices=[]),
        FakeReq(rid="99", origin_input_ids=shared + [99], prefix_indices=[]),
    ]
    rid_map = _intern_map(["1", "2"])  # 99 absent
    dep = compute_in_batch_deprioritize(reqs, rid_map, tree)
    # Pioneer for the shared prefix is "1"; "2" and "99" share the same
    # first-32 tokens and get deprioritized.
    assert dep == {"2", "99"}


def test_threshold_disabled_returns_empty():
    """Passing a negative check or deprioritize threshold disables the logic
    entirely, matching sglang's IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD=-1
    sentinel."""
    tree = PendingTree()
    shared = list(range(100, 140))
    for i, tail in enumerate([[1], [2], [3]]):
        tree.insert(i + 1, shared + tail)
    reqs = [
        FakeReq(rid=str(i + 1), origin_input_ids=shared + [i + 1], prefix_indices=[])
        for i in range(3)
    ]
    rid_map = _intern_map(["1", "2", "3"])
    assert compute_in_batch_deprioritize(reqs, rid_map, tree, check_threshold=-1) == set()
    assert (
        compute_in_batch_deprioritize(reqs, rid_map, tree, deprioritize_threshold=-1)
        == set()
    )


# ---------------------------------------------------------------------------
# peek_sort_inplace — the full LPM-semantic sort with cluster-size tiebreak.
# ---------------------------------------------------------------------------


def test_sort_degrades_to_lpm_when_no_clusters():
    """A queue of singletons (no clusters) sorts exactly by LPM: descending
    main_hit, stable by rid."""
    tree = PendingTree()
    tree.insert(1, list(range(100, 160)))  # 60 unique tokens
    tree.insert(2, list(range(200, 260)))
    tree.insert(3, list(range(300, 360)))
    reqs = [
        FakeReq(rid="a", origin_input_ids=list(range(100, 160)), prefix_indices=list(range(50))),
        FakeReq(rid="b", origin_input_ids=list(range(200, 260)), prefix_indices=list(range(10))),
        FakeReq(rid="c", origin_input_ids=list(range(300, 360)), prefix_indices=list(range(30))),
    ]
    rid_map = {"a": 1, "b": 2, "c": 3}
    peek_sort_inplace(reqs, rid_map, tree)
    # Expect order by -main_hit: a(50) > c(30) > b(10)
    assert [r.rid for r in reqs] == ["a", "c", "b"]


def test_sort_groups_siblings_adjacent_to_pioneer():
    """Cluster-grouped sort: same-cluster members land contiguously in the
    queue. Within each cluster, pioneer (non-deprioritized) sorts before
    siblings (deprioritized). Inter-cluster order is by cluster_node_id
    (stable under tree structure) — NOT by cluster size, so small clusters
    aren't starved by bigger ones cutting in."""
    tree = PendingTree()
    big_pref = list(range(100, 140))
    for i in range(4):
        tree.insert(i + 1, big_pref + [i + 1])
    small_pref = list(range(200, 240))
    for i in range(2):
        tree.insert(i + 5, small_pref + [i + 5])
    tree.insert(7, list(range(300, 340)) + [7])

    reqs = [
        FakeReq(rid="7", origin_input_ids=list(range(300, 340)) + [7], prefix_indices=[]),
        FakeReq(rid="5", origin_input_ids=small_pref + [5], prefix_indices=[]),
        FakeReq(rid="1", origin_input_ids=big_pref + [1], prefix_indices=[]),
        FakeReq(rid="6", origin_input_ids=small_pref + [6], prefix_indices=[]),
        FakeReq(rid="2", origin_input_ids=big_pref + [2], prefix_indices=[]),
        FakeReq(rid="3", origin_input_ids=big_pref + [3], prefix_indices=[]),
        FakeReq(rid="4", origin_input_ids=big_pref + [4], prefix_indices=[]),
    ]
    rid_map = {str(i + 1): i + 1 for i in range(7)}
    peek_sort_inplace(reqs, rid_map, tree)
    order = [r.rid for r in reqs]

    # Cluster A members {"1","2","3","4"} contiguous; "1" first (pioneer).
    a_idx = [i for i, rid in enumerate(order) if rid in {"1", "2", "3", "4"}]
    assert a_idx == list(range(a_idx[0], a_idx[0] + 4)), \
        f"cluster A not contiguous: {order}"
    assert order[a_idx[0]] == "1", f"A pioneer should be first in its block: {order}"
    # Cluster B members {"5","6"} contiguous; "5" first (pioneer).
    b_idx = [i for i, rid in enumerate(order) if rid in {"5", "6"}]
    assert b_idx == list(range(b_idx[0], b_idx[0] + 2)), \
        f"cluster B not contiguous: {order}"
    assert order[b_idx[0]] == "5", f"B pioneer should be first in its block: {order}"
    # Singleton "7" appears somewhere; placement relative to clusters depends
    # on the sentinel node_id, but it must not split a cluster.
    assert "7" in order


def test_sort_rank_by_size_off_matches_sglang_lpm():
    """With rank_by_cluster_size=False, the sort is LPM-faithful: among ties,
    arrival order (rid) wins; no cluster-size ranking."""
    tree = PendingTree()
    big_pref = list(range(100, 140))
    for i in range(4):
        tree.insert(i + 1, big_pref + [i + 1])
    small_pref = list(range(200, 240))
    for i in range(2):
        tree.insert(i + 5, small_pref + [i + 5])
    reqs = [
        FakeReq(rid="5", origin_input_ids=small_pref + [5], prefix_indices=[]),  # arrives first
        FakeReq(rid="1", origin_input_ids=big_pref + [1], prefix_indices=[]),
    ]
    rid_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    peek_sort_inplace(reqs, rid_map, tree, rank_by_cluster_size=False)
    # With rank-by-size off, both pioneers tie at main_hit=0 and both are
    # not-deprioritized → rid tiebreak makes "1" come first.
    assert [r.rid for r in reqs] == ["1", "5"]


def test_sort_warm_singleton_beats_cold_pioneer():
    """A warm singleton (main_hit > threshold) beats a cold cluster pioneer
    because the main_hit primary key dominates the cluster-size tiebreak."""
    tree = PendingTree()
    big_pref = list(range(100, 140))
    for i in range(4):
        tree.insert(i + 1, big_pref + [i + 1])
    tree.insert(99, list(range(500, 600)))

    reqs = [
        FakeReq(rid="1", origin_input_ids=big_pref + [1], prefix_indices=[]),  # cold A-pioneer, size=4
        FakeReq(
            rid="99",
            origin_input_ids=list(range(500, 600)),
            prefix_indices=list(range(80)),  # warm singleton, main_hit=80
        ),
    ]
    rid_map = {"1": 1, "2": 2, "3": 3, "4": 4, "99": 99}
    peek_sort_inplace(reqs, rid_map, tree)
    # Warm singleton wins on main_hit primary.
    assert [r.rid for r in reqs] == ["99", "1"]


def test_sort_singleton_vs_pioneer_on_main_hit_tie():
    """Singletons and pioneers tied on main_hit: the clustered req (lower
    cluster_node_id) sorts first; the singleton (sentinel node_id=2^31)
    sorts last. When the singleton has higher main_hit, it wins outright."""
    tree = PendingTree()
    big_pref = list(range(100, 140))
    for i in range(3):
        tree.insert(i + 1, big_pref + [i + 1])
    tree.insert(99, list(range(500, 600)))

    reqs = [
        FakeReq(rid="99", origin_input_ids=list(range(500, 600)), prefix_indices=[]),
        FakeReq(rid="1", origin_input_ids=big_pref + [1], prefix_indices=[]),
    ]
    rid_map = {"1": 1, "2": 2, "3": 3, "99": 99}
    peek_sort_inplace(reqs, rid_map, tree)
    # Pioneer of real cluster comes first; singleton (sentinel node_id) last.
    assert [r.rid for r in reqs] == ["1", "99"]
