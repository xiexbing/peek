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

"""Query-count comparison: per-rid vs dual-walk cache matching.

On heavy-sharing LLM workloads, the dual-walk visits each edge in peek's tree
exactly once -- independent of how many rids share that edge. A per-rid
approach visits the shared edge once per rid.

This test measures the concrete query-count savings as sharing density grows.
"""

from __future__ import annotations

import pytest
pytest.importorskip("sglang")

pytestmark = pytest.mark.engine

import random
from typing import List, Tuple

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from peek.online import PendingTree, compute_main_hits

def _make_cluster_workload(
    *,
    n_clusters: int,
    members_per_cluster: int,
    shared_prefix_len: int,
    tail_len: int,
    seed: int,
) -> List[Tuple[int, List[int]]]:
    rng = random.Random(seed)
    reqs: List[Tuple[int, List[int]]] = []
    rid = 0
    for cl in range(n_clusters):
        shared = [rng.randint(0, 999) for _ in range(shared_prefix_len)]
        for _ in range(members_per_cluster):
            rid += 1
            tail = [rng.randint(0, 999) for _ in range(tail_len)]
            reqs.append((rid, shared + tail))
    return reqs

def _build(reqs):
    tree = PendingTree()
    for rid, tokens in reqs:
        tree.insert(rid, tokens)
    return tree

def _run_counts(tree: PendingTree, cache_match_fn):
    calls = [0]
    def counting(tokens):
        calls[0] += 1
        return cache_match_fn(tokens)
    hits = compute_main_hits(tree, [], counting)
    return len(hits), calls[0]

@pytest.mark.parametrize("members", [5, 20, 50, 100])
def test_dualwalk_query_count_scales_with_edges_not_rids(
    members: int, capsys: pytest.CaptureFixture
) -> None:
    """Shared-prefix cluster: cache diverges inside the shared edge. One edge
    -> one query, regardless of cluster size."""
    reqs = _make_cluster_workload(
        n_clusters=1,
        members_per_cluster=members,
        shared_prefix_len=50,
        tail_len=10,
        seed=0,
    )
    tree = _build(reqs)

    # Cache with nothing in it -> diverges at depth 0 on the first edge.
    def empty_cache(_tokens):
        return 0

    n_hits, n_calls = _run_counts(tree, empty_cache)
    per_rid_calls = members  # baseline: would be one per rid

    with capsys.disabled():
        print(
            f"\n  members={members:>3}  peek dualwalk calls={n_calls:>2}  "
            f"per-rid baseline={per_rid_calls:>3}  savings={per_rid_calls - n_calls}x"
        )
    assert n_hits == members
    # Dual-walk: the shared edge diverges at the first token. Count is bounded
    # by peek's edge count at the boundary -- just 1 edge visited.
    assert n_calls == 1

def test_dualwalk_vs_per_rid_realistic(capsys: pytest.CaptureFixture) -> None:
    """Realistic mix: several shared-prefix clusters, warm cache for some, cold
    for others. Compare total cache calls between dual-walk and per-rid."""
    reqs = _make_cluster_workload(
        n_clusters=5,
        members_per_cluster=20,
        shared_prefix_len=40,
        tail_len=15,
        seed=42,
    )
    tree = _build(reqs)

    # Real cache: has the first cluster's shared prefix cached (warm), others cold.
    cache = RadixCache.create_simulated()
    first_cluster_shared = reqs[0][1][:40]
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=first_cluster_shared, extra_key=None),
            value=torch.empty(40, dtype=torch.bool),
        )
    )

    def real_cache_match(tokens):
        m = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=list(tokens), extra_key=None))
        )
        return len(m.device_indices)

    # peek dual-walk
    dw_calls = [0]
    def dw_fn(tokens):
        dw_calls[0] += 1
        return real_cache_match(tokens)
    dw_hits = compute_main_hits(tree, [], dw_fn)

    # per-rid baseline
    pr_calls = 0
    for rid, tokens in reqs:
        _ = real_cache_match(tokens)
        pr_calls += 1

    with capsys.disabled():
        print(
            f"\n  5 clusters x 20 members (1 warm, 4 cold): "
            f"peek dualwalk={dw_calls[0]} calls  |  "
            f"per-rid={pr_calls} calls  |  "
            f"{pr_calls / dw_calls[0]:.1f}x fewer"
        )

    assert len(dw_hits) == len(reqs)
    # Dual-walk should be significantly fewer calls. For 4 cold clusters (1 call
    # each) + 1 warm cluster (1 shared edge + 20 tail edges) ≈ 25 calls.
    # Per-rid would be 100. Expect at least 3x savings.
    assert dw_calls[0] * 3 <= pr_calls
