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

"""Engine-agnostic cluster-LPM (cLPM) scheduling order — the PEEK scheduler.

``clpm_order`` computes a scheduling permutation of the waiting queue from
peek's pending radix tree, decoupled from any engine's request object. SGLang
and vLLM both call it: they supply, per waiting request, a compact tuple
``(rid, main_hit, prefix_key, arrival_ns)`` plus the shared ``PendingTree``, and
get back an order (a list of indices into the input) to reorder the queue by.

The full PEEK scheduler (paper §3.2, "cLPM+GM+DL") composes three parts, all
read off peek's incrementally-maintained tree:

  * cLPM  -- requests split into sections (0 = cache-warm, 1 = cold pioneer,
             2 = cold sibling) so a shared prefix is warmed once before its
             siblings admit; a cache/cluster-preferring lane (Lane A) is
             stride-interleaved with an oldest-first fairness lane (Lane B).
             Lane A ranks by ``req_score`` (Σ pending_count(v)·|edge(v)| along
             the request's path) and ``cluster_size`` (pending_count at the
             deepest ≥2 ancestor).
  * GM    -- group-major: within a section, a cluster's members are emitted
             contiguously (clusters ranked by depth × size) so the cluster
             admits as one batch.
  * DL    -- dynamic-lane: the Lane-B fairness share is recomputed each call
             from the queue's singleton fraction and oldest-singleton wait,
             EMA-smoothed, so batching never starves singletons.
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

# Persistent Lane-B share for the dynamic-lane (DL) controller, keyed by tree
# identity so multiple tree instances don't cross-contaminate.
_dyn_lane_state: dict = {}


def clpm_order(
    items: Sequence[ClpmItem],
    tree,
    *,
    big_lane_share: float = 0.7,
    check_threshold: int = 32,
    group_major: bool = False,
    dynamic_lane: bool = False,
    slo_budget_s: float = 2.0,
) -> List[int]:
    """Return a permutation of ``range(len(items))`` -- the cLPM schedule order.

    ``tree`` is a ``peek.PendingTree`` already synced to the waiting queue.

    The full PEEK scheduler (``clpm_gm_dl``) sets ``group_major=True`` and
    ``dynamic_lane=True``. With both off this is plain cLPM with a static lane
    split:

    * ``group_major`` (GM) -- within a section, emit each cluster's members
      contiguously (clusters ranked by depth × size).
    * ``dynamic_lane`` (DL) -- recompute the Lane-B fairness share each call from
      the queue's singleton fraction and oldest-singleton wait (EMA-smoothed),
      overriding ``big_lane_share``. ``slo_budget_s`` scales the wait pressure.

    ``big_lane_share`` in [0, 1] is the static fraction of admissions from the
    cache/cluster-preferring lane (Lane A) when ``dynamic_lane`` is off; 1.0 =
    Lane A only, 0.0 = Lane B only.
    """
    n = len(items)
    if n <= 1:
        return list(range(n))

    # Two Rust calls over the whole tree, amortized across the queue.
    req_scores = tree.compute_req_scores()
    all_info = tree.all_cluster_info()

    # DL: recompute the fairness-lane share from queue composition + singleton
    # wait time, EMA-smoothed. Overrides big_lane_share for this call.
    if dynamic_lane:
        import time as _time

        now = _time.time()
        n_single = 0
        oldest_single_age = 0.0
        for rid, _mh, _pk, arrival_ns in items:
            info = all_info.get(rid)
            if info is None or info[2] < 2:  # singleton (not in a ≥2 cluster)
                n_single += 1
                if arrival_ns:
                    age = now - arrival_ns / 1e9
                    if age > oldest_single_age:
                        oldest_single_age = age
        singleton_frac = n_single / n
        age_pressure = (
            min(1.0, oldest_single_age / slo_budget_s) if slo_budget_s > 0 else 0.0
        )
        raw_b = max(
            0.10, min(0.60, 0.15 + 0.50 * singleton_frac + 0.30 * age_pressure)
        )
        key = id(tree)
        prev_b = _dyn_lane_state.get(key, 0.3)
        smoothed_b = 0.7 * prev_b + 0.3 * raw_b
        _dyn_lane_state[key] = smoothed_b
        big_lane_share = 1.0 - smoothed_b

    seen_prefixes: set = set()
    lane_a: List[tuple] = []
    lane_b: List[tuple] = []
    # For GM: (section, group_key, group_score, main_hit, arrival_ns) per item.
    gm_meta: List[tuple] = []
    for rid, main_hit, prefix_key, arrival_ns in items:
        # Section: 0 = warm (already cached), 1 = cold pioneer, 2 = cold sibling.
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

        # GM group: deepest ≥2 cluster, else the request is its own singleton
        # group of score 0.
        if info is not None and info[2] >= 2:
            cluster_node, depth, size = info
            gkey: Hashable = ("G", cluster_node)
            gscore = int(depth) * int(size)
        else:
            gkey = ("S", rid)
            gscore = 0
        gm_meta.append((section, gkey, gscore, int(main_hit), arrival_ns))

    idx = list(range(n))

    def _lane_a_order() -> List[int]:
        if not group_major:
            return sorted(idx, key=lambda i: lane_a[i])
        # GM: bucket by (section, group). Order groups by section, then
        # (-depth·size, earliest arrival); members by (-main_hit, arrival).
        groups: dict = {}
        group_score: dict = {}
        group_min_arr: dict = {}
        for i, (section, gkey, gscore, _mh, arr_ns) in enumerate(gm_meta):
            bk = (section, gkey)
            groups.setdefault(bk, []).append(i)
            group_score[bk] = gscore
            prev_min = group_min_arr.get(bk)
            if prev_min is None or arr_ns < prev_min:
                group_min_arr[bk] = arr_ns
        group_order = sorted(
            groups.keys(),
            key=lambda bk: (bk[0], -group_score[bk], group_min_arr[bk]),
        )
        result: List[int] = []
        for bk in group_order:
            members = groups[bk]
            members.sort(key=lambda i: (-gm_meta[i][3], gm_meta[i][4]))
            result.extend(members)
        return result

    a_sorted = _lane_a_order()
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
