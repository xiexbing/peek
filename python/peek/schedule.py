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

"""Engine-agnostic Cluster-LPM (cLPM) scheduling order.

``clpm_order`` computes a scheduling permutation of the waiting queue from
peek's pending radix tree, decoupled from any engine's request object. SGLang
and vLLM both call it: they supply, per waiting request, a compact tuple
``(rid, main_hit, prefix_key, arrival_ns)`` plus the shared ``PendingTree``, and
get back an order (a list of indices into the input) to reorder the queue by.

Cluster-LPM (PEEK paper, Xie et al., 2026, section 3.2) refines LPM with signals
LPM alone cannot cheaply produce, read off peek's incrementally-maintained tree:

  * ``req_score``    = Sum over ancestors of pending_count(v) * |edge(v)| along
                       the request's path -- peek-native dense-subtree weight.
  * ``cluster_size`` = pending_count at the request's deepest >=2 ancestor.

Requests split into sections (0 = cache-warm, 1 = cold pioneer, 2 = cold
sibling) so a shared prefix is warmed once before its siblings admit. Two lanes
are stride-interleaved: Lane A favors cache-warm / dense-cluster requests; Lane B
is oldest-first fairness so small clusters are never starved.
"""

from __future__ import annotations

from typing import Hashable, List, Optional, Sequence, Tuple

# One waiting request as seen by the scheduler:
#   rid         -- interned integer id, matching what was inserted into the tree
#   main_hit    -- KV-cache prefix-match length in tokens (LPM primary signal)
#   prefix_key  -- hashable claim key for in-batch pioneer/sibling detection
#                  (e.g. the first-K prompt tokens), or None if too short
#   arrival_ns  -- arrival timestamp in ns for fairness ordering (0 if unknown)
ClpmItem = Tuple[int, int, Optional[Hashable], int]


def clpm_order(
    items: Sequence[ClpmItem],
    tree,
    *,
    big_lane_share: float = 0.7,
    check_threshold: int = 32,
) -> List[int]:
    """Return a permutation of ``range(len(items))`` -- the cLPM schedule order.

    ``tree`` is a ``peek.PendingTree`` already synced to the waiting queue.
    ``big_lane_share`` in [0, 1] is the fraction of admissions drawn from the
    cache/cluster-preferring lane (Lane A); the rest come from the oldest-first
    fairness lane (Lane B). 1.0 = Lane A only, 0.0 = Lane B only.
    """
    n = len(items)
    if n <= 1:
        return list(range(n))

    # Two Rust calls over the whole tree, amortized across the queue.
    req_scores = tree.compute_req_scores()
    all_info = tree.all_cluster_info()

    seen_prefixes: set = set()
    lane_a: List[tuple] = []
    lane_b: List[tuple] = []
    for rid, main_hit, prefix_key, arrival_ns in items:
        # Section: 0 = warm (already cached), 1 = cold pioneer, 2 = cold sibling.
        # A shared prefix's first-seen (in arrival order) request is the pioneer;
        # later requests sharing its claim key are siblings (deprioritized to the
        # section-2 tail so the pioneer warms the prefix first).
        if main_hit > check_threshold:
            section = 0
        elif prefix_key is None:
            section = 1
        elif prefix_key in seen_prefixes:
            section = 2
        else:
            seen_prefixes.add(prefix_key)
            section = 1

        req_score = req_scores.get(rid, 0)
        info = all_info.get(rid)
        cluster_size = info[2] if info is not None else 0

        # Lane A: cache-warm + dense-subtree preferring.
        lane_a.append(
            (section, -int(main_hit), -int(req_score), -int(cluster_size), arrival_ns)
        )
        # Lane B: oldest-first fairness within section.
        lane_b.append((section, arrival_ns, -int(main_hit)))

    idx = list(range(n))
    a_sorted = sorted(idx, key=lambda i: lane_a[i])
    if big_lane_share >= 1.0 - 1e-9:
        return a_sorted
    b_sorted = sorted(idx, key=lambda i: lane_b[i])
    if big_lane_share <= 1e-9:
        return b_sorted

    # Two-lane stride scheduling: each lane advances a virtual clock by 1/share
    # per pick; the lane with the smaller clock picks next. Over many picks the
    # admission ratio converges to big_lane_share : (1 - big_lane_share).
    a_stride = 1.0 / big_lane_share
    b_stride = 1.0 / (1.0 - big_lane_share)
    a_vclock = a_stride
    b_vclock = b_stride
    a_ptr = b_ptr = 0
    picked: set = set()
    order: List[int] = []
    while len(order) < n:
        if a_vclock <= b_vclock:
            while a_ptr < n and a_sorted[a_ptr] in picked:
                a_ptr += 1
            if a_ptr < n:
                order.append(a_sorted[a_ptr])
                picked.add(a_sorted[a_ptr])
                a_ptr += 1
                a_vclock += a_stride
            else:
                a_vclock = float("inf")
        else:
            while b_ptr < n and b_sorted[b_ptr] in picked:
                b_ptr += 1
            if b_ptr < n:
                order.append(b_sorted[b_ptr])
                picked.add(b_sorted[b_ptr])
                b_ptr += 1
                b_vclock += b_stride
            else:
                b_vclock = float("inf")
    return order
