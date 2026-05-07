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

"""Tests for the dual-walk main-hit computation.

The walk queries the cache ONCE PER EDGE in peek's tree, never per rid. When
the cache diverges within an edge, every rid in that subtree is assigned the
same main_hit — no further cache calls for that branch.

Default mode skips subtrees with pending_count < 2 (singleton-tail subtrees)
to match the typical LLM serving assumption that per-request tails are
rarely cached. Pass `min_pending_count=1` for exact per-edge queries.
"""

from peek.online import PendingTree, compute_main_hits


def test_empty_tree():
    tree = PendingTree()
    assert compute_main_hits(tree, [], lambda _: 0) == {}


def test_solitary_req_still_queried_by_default():
    """A solitary req (whole path is pc=1) must still be queried once —
    otherwise we'd miss its main_hit entirely."""
    tree = PendingTree()
    tree.insert(1, [1, 2, 3])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        return 2
    hits = compute_main_hits(tree, [], fake_cache)
    assert hits == {1: 2}
    assert calls == [[1, 2, 3]]


def test_cluster_tails_skipped_after_establishment():
    """Once the shared prefix is queried, singleton tails are skipped."""
    tree = PendingTree()
    tree.insert(1, [1, 2, 3, 4, 5])
    tree.insert(2, [1, 2, 3, 4, 6])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        return 4  # shared edge fully matched
    hits = compute_main_hits(tree, [], fake_cache)
    # Both rids get shared main_hit; their tails are skipped.
    assert hits == {1: 4, 2: 4}
    # Only ONE call: the shared edge. Tails not queried.
    assert calls == [[1, 2, 3, 4]]


def test_singleton_queried_with_min_pc_1():
    tree = PendingTree()
    tree.insert(1, [1, 2, 3])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        return 2
    hits = compute_main_hits(tree, [], fake_cache, min_pending_count=1)
    assert hits == {1: 2}
    assert calls == [[1, 2, 3]]


def test_cold_cluster_diverges_in_shared_edge_one_query():
    tree = PendingTree()
    tree.insert(1, [1, 2, 3, 4, 5])
    tree.insert(2, [1, 2, 3, 4, 6])
    tree.insert(3, [1, 2, 3, 4, 7])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        return 2  # diverges in the shared edge
    hits = compute_main_hits(tree, [], fake_cache)
    assert hits == {1: 2, 2: 2, 3: 2}
    assert len(calls) == 1
    assert calls[0] == [1, 2, 3, 4]


def test_warm_cluster_walks_tails_with_min_pc_1():
    tree = PendingTree()
    tree.insert(1, [1, 2, 3, 4, 5])
    tree.insert(2, [1, 2, 3, 4, 6])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        t = list(tokens)
        if t == [1, 2, 3, 4]:
            return 4
        if t == [1, 2, 3, 4, 5]:
            return 5
        if t == [1, 2, 3, 4, 6]:
            return 4
        return 0
    hits = compute_main_hits(tree, [], fake_cache, min_pending_count=1)
    assert hits == {1: 5, 2: 4}
    assert sorted(calls) == [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]


def test_deep_sharing_collapses_subtree_on_divergence():
    tree = PendingTree()
    for i in range(10):
        tree.insert(i + 1, [1, 2, 3, 4, 5, 6, 7, 100 + i])
    calls = []
    def fake_cache(tokens):
        calls.append(list(tokens))
        return 3  # diverges early
    hits = compute_main_hits(tree, [], fake_cache)
    assert len(hits) == 10
    assert all(v == 3 for v in hits.values())
    assert len(calls) == 1
