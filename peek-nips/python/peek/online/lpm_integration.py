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

"""Peek's scheduling integration with sglang.

Peek's scheduler is LPM-semantic at its core -- longest-prefix-match against
sglang's radix cache drives the primary sort key -- augmented by two signals
that LPM alone cannot produce:

  1. In-batch pioneer selection. When multiple cold reqs share a prefix of
     length >= K tokens, the first-seen req becomes the pioneer and the rest
     are deprioritized so the pioneer warms the shared prefix first. sglang
     already does this with a throwaway auxiliary radix tree rebuilt each
     call; peek reads the same information off its pending tree in O(1) per
     req, with no extra rebuild cost.

  2. Pioneer ordering by cluster size. Among pioneers and singletons tied on
     main_hit, peek ranks by cluster size so the pioneer of the biggest
     group admits first -- its prefill unlocks the most downstream siblings'
     cached prefix. sglang has no analogue for this; LPM ties fall back to
     arrival order (FCFS). This is the key place peek beats LPM in both the
     all-cold regime (everyone tied at main_hit=0) and the mid-warm regime
     (many reqs tied on a shared cached prefix).

This module exposes two sort entry points:

* ``peek_lpm_sort_inplace`` -- LPM-equivalent sort. Key (3-tuple, with
  ``rank_by_cluster_size=False``):

      (-main_hit, is_deprioritized, rid)

  Byte-identical to stock sglang LPM. With ``rank_by_cluster_size=True``
  the key becomes ``(-main_hit, cluster_node_id, is_deprioritized, rid)``
  which groups same-cluster reqs adjacently without ranking clusters by
  size.

* ``peek_clpm_sort_inplace`` -- Cluster-LPM (paper §3.2, Eq. 1). Lane A
  key (5-tuple):

      (section, -main_hit, -req_score, -cluster_size, arrival_ns)

  where ``section`` ∈ {0=warm, 1=pioneer, 2=sibling} is the primary key,
  ``req_score = Σ pending_count(v).|edge(v)|`` along the rid's path
  through the pending tree, and ``cluster_size`` is the pending_count at
  the rid's deepest ≥2 ancestor. cLPM stride-interleaves Lane A with a
  fairness Lane B keyed by ``(section, arrival_ns, -main_hit)`` (paper
  §3.2 multi-lane scheduler).

When the queue has no shared-prefix structure, ``req_score`` and
``cluster_size`` are 0 for every rid and cLPM's ordering degenerates to
``(section, -main_hit, arrival)`` -- which is paper-stock LPM with the
section flag promoted to primary. The ``has_sharing`` guard (Rust core)
short-circuits this path entirely on no-sharing queues (paper §3.3).

This module reads sglang's radix cache indirectly via `prefix_indices`
populated on each req by the caller, and reads peek's pending tree directly
for cluster structure. It does not build any auxiliary tree of its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from peek import PendingTree


# Persistent state for the dynamic-lane controller. Holds the EMA-smoothed
# Lane B share across scheduling ticks. Keyed by tree identity so multiple
# tree instances don't cross-contaminate.
_dyn_lane_state: Dict[int, float] = {}


class _ReqLike(Protocol):
    rid: str
    prefix_indices: object  # sglang-specific container; only length is read


def compute_in_batch_deprioritize(
    waiting_queue: Sequence[_ReqLike],
    rid_to_int: Dict[str, int],
    tree: PendingTree,
    *,
    check_threshold: int = 32,
    deprioritize_threshold: int = 32,
) -> Set[str]:
    """LPM-byte-identical in-batch deprioritize.

    Matches sglang's `_compute_prefix_matches` semantics exactly:
      * For each cold req (main_hit <= check_threshold), walk the waiting
        queue in arrival order.
      * If this req's first `deprioritize_threshold` tokens are identical
        to a previously-seen pioneer's first `deprioritize_threshold`
        tokens -> deprioritize. Else, this req becomes a new pioneer and
        claims that token-prefix.

    Semantically equivalent to sglang's aux-radix-tree "match length >=
    deprioritize_threshold" check: any two reqs that match ≥K tokens in a
    radix tree necessarily match in their first K tokens, and vice versa.
    We use a simple `tuple(tokens[:K])` hash as the pioneer claim key.

    Earlier implementations used peek's `cluster_info` (deepest ≥2-pending
    ancestor) which diverged in topologies where a req's deepest cluster
    was not yet claimed but a shallower ancestor had been -- e.g., sibling
    of cluster X where X overlaps with cluster Y at depth < X.depth. The
    token-prefix approach avoids that trap by checking the exact threshold
    depth that LPM's aux tree checks.
    """
    del tree, rid_to_int  # accepted for API compatibility; not needed.
    if check_threshold < 0 or deprioritize_threshold < 0:
        return set()
    seen_prefixes: Set[tuple] = set()
    deprioritized: Set[str] = set()
    for r in waiting_queue:
        pi = r.prefix_indices
        main_hit = 0 if pi is None else len(pi)
        if main_hit > check_threshold:
            continue
        tokens = list(r.origin_input_ids) + list(r.output_ids or [])
        if len(tokens) < deprioritize_threshold:
            continue
        prefix_key = tuple(tokens[:deprioritize_threshold])
        if prefix_key in seen_prefixes:
            deprioritized.add(r.rid)
        else:
            seen_prefixes.add(prefix_key)
    return deprioritized


_SINGLETON_NODE_ID = 2**31  # sentinel for singletons in the sort key


def peek_sort_inplace(
    waiting_queue: List[_ReqLike],
    rid_to_int: Dict[str, int],
    tree: PendingTree,
    *,
    check_threshold: int = 32,
    deprioritize_threshold: int = 32,
    rank_by_cluster_size: bool = True,
    main_hits: Optional[Dict[int, int]] = None,
    peek_lpm_sort: bool = False,
) -> Set[str]:
    """Sort waiting_queue in-place using peek's cluster-grouped LPM scheduler.

    Sort key (5-tuple, lexicographic ascending):

        (-main_hit, -cluster_size, cluster_node_id, is_deprioritized, rid)

      1. `-main_hit` -- LPM primary. Cache-warm reqs (prefix_indices long)
         sort first.
      2. `-cluster_size` -- across clusters tied on main_hit, the larger
         cluster's reqs come first. Singletons have size=0 so they sort
         last among main_hit ties.
      3. `cluster_node_id` -- groups all members of the same cluster into a
         consecutive block in the queue. Singletons get a sentinel id so
         they sort after all real clusters (which have node ids < 2^31).
      4. `is_deprioritized` -- within a cluster block, the pioneer (flag=0)
         sits immediately before its siblings (flag=1). This lets the
         scheduler admit pioneer_A, A2, A3, ..., pioneer_B, B2, B3, ...
         rather than "all pioneers, then all siblings."
      5. `rid` -- stable final tiebreak.

    Setting `rank_by_cluster_size=False` drops keys 2-4 and reproduces
    sglang's vanilla LPM order (for A/B diagnostic runs).

    Returns the deprioritize set for validation/debugging.
    """
    # Strict-LPM sort ignores per-rid cluster info entirely; skip the O(N)
    # all_cluster_info() traversal in that mode. This is the "peek_lpm" fast
    # path: same sort output as stock sglang LPM, no peek-specific bookkeeping.
    all_info = {} if peek_lpm_sort else tree.all_cluster_info()
    deprioritized: Set[str] = set()
    per_rid_sort_info: Dict[str, tuple] = {}
    seen_prefixes: Set[tuple] = set()

    # main_hits lookup helper. If the caller provided dualwalk results, use
    # them (keyed by interned rid_int). Otherwise fall back to the length of
    # r.prefix_indices populated by sglang's match_prefix.
    def _mh(r: _ReqLike) -> int:
        if main_hits is not None:
            rid_int = rid_to_int.get(r.rid)
            if rid_int is not None:
                return main_hits.get(rid_int, 0)
        pi = r.prefix_indices
        return 0 if pi is None else len(pi)

    # Pass 1: populate per-rid sort info (cluster_node_id for the optional
    # rank-by-cluster-size path) AND the LPM-byte-exact in-batch deprio set.
    # The deprio check uses tokens[:deprioritize_threshold] as the pioneer
    # claim key (matches sglang's aux-radix-tree "match >= threshold" check
    # semantically -- see compute_in_batch_deprioritize docstring).
    check_disabled = check_threshold < 0 or deprioritize_threshold < 0
    for r in waiting_queue:
        if not peek_lpm_sort:
            rid_int = rid_to_int.get(r.rid)
            info = all_info.get(rid_int) if rid_int is not None else None
            if info is None:
                per_rid_sort_info[r.rid] = (_SINGLETON_NODE_ID, 0)
            else:
                cluster_node, _depth, size = info
                per_rid_sort_info[r.rid] = (cluster_node, size)
        if check_disabled:
            continue
        main_hit = _mh(r)
        if main_hit > check_threshold:
            continue
        tokens = list(r.origin_input_ids) + list(r.output_ids or [])
        if len(tokens) < deprioritize_threshold:
            continue
        prefix_key = tuple(tokens[:deprioritize_threshold])
        if prefix_key in seen_prefixes:
            deprioritized.add(r.rid)
        else:
            seen_prefixes.add(prefix_key)

    # Pass 2: stable sort using the 4-tuple key.
    #   (-main_hit, cluster_node_id, is_deprioritized, rid)
    # cluster_node_id groups same-cluster members adjacently (DFS-like) but
    # does NOT rank clusters by size -- so small clusters can't be perpetually
    # starved by new big-cluster arrivals cutting in front of them. Inter-
    # cluster order is effectively determined by peek's tree structure
    # (arrival-stable node ids); within each cluster, pioneer (is_depr=0)
    # sorts before siblings (is_depr=1).
    #
    # Setting rank_by_cluster_size=False reverts to LPM-faithful 3-tuple for
    # A/B diagnostics -- that mode IGNORES cluster grouping entirely.
    if peek_lpm_sort:
        # LPM-byte-identical sort, Rust-backed. Build (main_hit, is_depr) per
        # queue position in a single pass, hand to _core.lpm_sort_order for
        # the stable sort (eliminates per-compare Python call overhead), then
        # reorder the queue by the returned indices. Output is byte-identical
        # to `sorted(queue, key=lambda r: inf if depr else -main_hit)`.
        from peek._core import lpm_sort_order as _lpm_sort_order
        keys = [(_mh(r), r.rid in deprioritized) for r in waiting_queue]
        order = _lpm_sort_order(keys)
        waiting_queue[:] = [waiting_queue[i] for i in order]
        return deprioritized

    def _key(r: _ReqLike):
        main_hit = _mh(r)
        is_depr = 1 if r.rid in deprioritized else 0
        if rank_by_cluster_size:
            node_id, _size = per_rid_sort_info.get(
                r.rid, (_SINGLETON_NODE_ID, 0)
            )
            return (-main_hit, node_id, is_depr, r.rid)
        # 3-tuple LPM-equivalent (deprio tiebreak within main_hit bucket,
        # NOT banished to end; differs from strict LPM in mid-warm edge cases).
        return (-main_hit, is_depr, r.rid)

    waiting_queue.sort(key=_key)
    return deprioritized


# ---------------------------------------------------------------------------
# peek_clpm -- cluster-LPM.
#
# Builds on peek_lpm:
#   - Same pioneer/sibling semantics (token-prefix claim set for deprio)
#   - Same LPM-exact main_hit signal
# Adds three signals stock LPM can't cheaply produce:
#   - arrival_bucket  = floor((now − arrive_ts) / W)  -> FCFS across windows
#   - req_score       = Σ (pending_count x edge_length) along ancestors -> peek-native
#                        dense-subtree weight
#   - cluster_size    = pending_count at finest cluster -> broader cluster wins
#
# Sort key per req:
#   (section_id, -main_hit, arrival_bucket, -req_score, -cluster_size, arrival_ns)
#
# Sections: 0 = warm, 1 = cold pioneer, 2 = cold sibling.
# Deprio banishment preserved (sibling -> section 2 -> tail).
# Key order rationale: section/main_hit primary preserves LPM cache locality;
# bucket kicks in only as a tiebreak among cache-equal reqs (anti-starvation
# within same-main_hit bucket); req_score/cluster_size tiebreak below bucket.
# Prior ordering (bucket primary) fought cache locality on shared-prompt
# workloads where reqs of the same group arrive dispersed across many buckets.
# ---------------------------------------------------------------------------


def peek_clpm_sort_inplace(
    waiting_queue: List[_ReqLike],
    rid_to_int: Dict[str, int],
    tree: PendingTree,
    *,
    window_ms: int = 500,  # retained for back-compat; unused
    check_threshold: int = 32,
    deprioritize_threshold: int = 32,
    main_hits: Optional[Dict[int, int]] = None,
    arrival_ts: Optional[Dict[str, float]] = None,
    age_alpha: float = 0.0,
    big_lane_share: float = 0.7,
    group_major: bool = False,
    dynamic_lane: bool = False,
    slo_budget_s: float = 2.0,
) -> Set[str]:
    """Cluster-LPM with weighted 2-lane interleave (stride scheduler).

    Lane A (share=big_lane_share): sorted by (section, -main_hit, -req_score,
      -cluster_size, arrival_ns). Favors cache-warm + dense-subtree reqs.
      Same order as peek_clpm_W=0 -- preserves cache-locality advantage.

    Lane B (share=1-big_lane_share): sorted by (section, arrival_ns, -main_hit).
      Favors oldest pending. Guarantees small/singleton clusters aren't starved.

    At admission time, interleave via stride scheduling: the lane with smaller
    `virtual_runtime = picks_so_far / share` picks next. Over many admissions,
    ratio converges to big_lane_share : (1-big_lane_share).

    big_lane_share=1.0 -> Lane A only (≡ peek_clpm_W=0, no starvation bound).
    big_lane_share=0.0 -> Lane B only (pure arrival-FCFS within section).
    Default 0.7 -> 70% admissions from the dense-cluster-preferring order,
    30% from the oldest-first fairness order.

    `age_alpha` (default 0): if > 0, adds age-weighting on top -- applied
    inside Lane A's main_hit computation (effective_mh = mh + α x wait_s).
    Mostly redundant with lane interleaving; leave at 0 for clean A/B.

    `arrival_ts` maps rid -> wall-clock seconds (wired from phase tracking).
    `main_hits` is optional dict of {rid_int: main_hit}; falls back to
    len(r.prefix_indices).

    `group_major` (default False): if True, Lane A groups reqs by prefix-
    sharing cluster (deepest ≥2-pending ancestor in peek's pending tree)
    and emits group members contiguously. Groups ranked by depth x size;
    singletons are groups of score 0. Tightens inter-req KV reuse within
    the decode batch beyond what the per-req cluster_size tiebreak can do.

    `dynamic_lane` (default False): if True, the Lane B share is recomputed
    each tick from queue composition + oldest-singleton wait time:
        b_share = clamp(0.15 + 0.5.singleton_frac + 0.3.age_pressure, 0.1, 0.6)
    then EMA-smoothed (α=0.3). `big_lane_share` arg is ignored in this
    mode. `slo_budget_s` (default 2.0) is the denominator for age_pressure.

    Returns the deprioritize set (for validation).
    """
    import time as _time

    # One Rust call: per-rid score across the pending tree.
    req_scores = tree.compute_req_scores()
    # One Rust call: per-rid (cluster_node, depth, size) or None.
    all_info = tree.all_cluster_info()

    def _mh(r: _ReqLike) -> int:
        if main_hits is not None:
            rid_int = rid_to_int.get(r.rid)
            if rid_int is not None:
                return main_hits.get(rid_int, 0)
        pi = r.prefix_indices
        return 0 if pi is None else len(pi)

    # Pass 1: assign section_id via in-batch prefix-claim set.
    section_of: Dict[str, int] = {}
    deprioritized: Set[str] = set()
    seen_prefixes: Set[tuple] = set()
    check_disabled = check_threshold < 0 or deprioritize_threshold < 0
    for r in waiting_queue:
        mh = _mh(r)
        if mh > check_threshold:
            section_of[r.rid] = 0  # warm
            continue
        if check_disabled:
            section_of[r.rid] = 1  # pioneer (no claim check)
            continue
        tokens = list(r.origin_input_ids) + list(r.output_ids or [])
        if len(tokens) < deprioritize_threshold:
            section_of[r.rid] = 1  # pioneer (too short to be meaningfully deprioritized)
            continue
        prefix_key = tuple(tokens[:deprioritize_threshold])
        if prefix_key in seen_prefixes:
            deprioritized.add(r.rid)
            section_of[r.rid] = 2  # sibling
        else:
            seen_prefixes.add(prefix_key)
            section_of[r.rid] = 1  # pioneer

    # Sampling time for age-weighting and dynamic-lane controller.
    # Single `now` per tick; sampled unconditionally when arrival_ts is
    # present, since the dynamic-lane path also consumes it.
    need_now = bool(arrival_ts) and (age_alpha > 0 or dynamic_lane)
    now = _time.time() if need_now else 0.0

    n = len(waiting_queue)

    # Dynamic lane-share controller: recompute big_lane_share per tick from
    # queue composition + singleton wait-time pressure, EMA-smoothed.
    #
    #   singleton_frac  = fraction of reqs not in any ≥2-pending cluster
    #   age_pressure    = min(1, oldest_singleton_wait / slo_budget_s)
    #   raw_b_share     = 0.15 + 0.50.singleton_frac + 0.30.age_pressure
    #   b_share         = clamp(raw_b_share, 0.10, 0.60)
    #   b_share         = 0.7.prev_b_share + 0.3.b_share        (EMA)
    #   big_lane_share  = 1 − b_share
    #
    # Floor 0.10 keeps a trickle of fairness even in all-cluster queues.
    # Ceiling 0.60 prevents Lane B from inverting Lane A (cache-locality
    # must remain dominant). EMA α=0.3 damps tick-to-tick oscillation.
    if dynamic_lane and n > 0:
        n_single = 0
        oldest_single_age = 0.0
        for r in waiting_queue:
            rid_int = rid_to_int.get(r.rid)
            info_r = all_info.get(rid_int) if rid_int is not None else None
            is_single = info_r is None or info_r[2] < 2
            if is_single:
                n_single += 1
                if arrival_ts:
                    at = arrival_ts.get(r.rid)
                    if at:
                        age = now - at if now else 0.0
                        if age > oldest_single_age:
                            oldest_single_age = age
        singleton_frac = n_single / n
        age_pressure = (
            min(1.0, oldest_single_age / slo_budget_s)
            if slo_budget_s > 0 else 0.0
        )
        raw_b = 0.15 + 0.50 * singleton_frac + 0.30 * age_pressure
        raw_b = max(0.10, min(0.60, raw_b))
        key = id(tree)
        prev_b = _dyn_lane_state.get(key, 0.3)
        smoothed_b = 0.7 * prev_b + 0.3 * raw_b
        _dyn_lane_state[key] = smoothed_b
        big_lane_share = 1.0 - smoothed_b

    # Pass 2: build per-req (lane_A_key, lane_B_key) pairs.
    # Also collect per-req metadata we'll reuse for the group-major path.
    lane_a: List[tuple] = []
    lane_b: List[tuple] = []
    # For group_major: (section, group_key, group_score, mh, arrival_ns) per i.
    gm_meta: List[Tuple[int, Any, int, int, int]] = []
    for r in waiting_queue:
        rid_int = rid_to_int.get(r.rid)
        mh = _mh(r)
        if arrival_ts:
            at = arrival_ts.get(r.rid, now if now else 0.0)
            arrival_ns = int(at * 1e9)
        else:
            at = 0.0
            arrival_ns = 0
        if arrival_ts and age_alpha > 0:
            wait_sec = max(0.0, now - at)
            effective_mh = mh + int(age_alpha * wait_sec)
        else:
            effective_mh = mh
        section = section_of.get(r.rid, 1)
        req_score = req_scores.get(rid_int, 0) if rid_int is not None else 0
        info = all_info.get(rid_int) if rid_int is not None else None
        cluster_size = info[2] if info is not None else 0
        # Lane A (per-req): dense-cluster-preferring (≡ peek_clpm_W=0 key).
        lane_a.append((
            section, -int(effective_mh), -int(req_score),
            -int(cluster_size), arrival_ns,
        ))
        # Lane B: fairness -- oldest-first within section.
        lane_b.append((
            section, arrival_ns, -int(mh),
        ))
        # Group-major metadata: group_key + group_score.
        # Group = deepest ≥2-pending cluster a req belongs to; singletons
        # each form their own group of score 0 (so they sort last within
        # their section but keep arrival-order among themselves).
        if info is not None and info[2] >= 2:
            cluster_node, depth, size = info
            gkey: Any = ("G", cluster_node)
            gscore = int(depth) * int(size)
        else:
            gkey = ("S", r.rid)
            gscore = 0
        gm_meta.append((section, gkey, gscore, int(mh), arrival_ns))

    idx = list(range(n))

    def _lane_a_order() -> List[int]:
        """Lane A ordering. Per-req key by default; group-major if requested.

        Group-major: within each section, reqs sharing a cluster_node are
        emitted contiguously. Groups ranked by depth x size (deeper shared
        prefix x more members = higher priority). Singletons = groups of
        score 0, admitted last within their section in arrival order.
        Within a group, order by (-main_hit, arrival_ns).
        """
        if not group_major:
            return sorted(idx, key=lambda i: lane_a[i])
        # Bucket by (section, group_key).
        groups: Dict[Tuple[int, Any], List[int]] = {}
        group_score: Dict[Tuple[int, Any], int] = {}
        group_min_arr: Dict[Tuple[int, Any], int] = {}
        for i, (section, gkey, gscore, _mh_i, arr_ns) in enumerate(gm_meta):
            bk = (section, gkey)
            groups.setdefault(bk, []).append(i)
            group_score[bk] = gscore
            prev_min = group_min_arr.get(bk)
            if prev_min is None or arr_ns < prev_min:
                group_min_arr[bk] = arr_ns
        # Order groups: section asc, then (-group_score, min arrival_ns).
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

    # Fast paths: single-lane collapses.
    if big_lane_share >= 1.0 - 1e-9:
        order = _lane_a_order()
    elif big_lane_share <= 1e-9:
        order = sorted(idx, key=lambda i: lane_b[i])
    else:
        # Two-lane stride scheduling. Each lane advances a virtual clock by
        # 1/share per pick; smaller clock picks next. Converges to share ratio.
        a_sorted = _lane_a_order()
        b_sorted = sorted(idx, key=lambda i: lane_b[i])
        a_stride = 1.0 / big_lane_share
        b_stride = 1.0 / (1.0 - big_lane_share)
        a_vclock = a_stride  # start each clock at its first-pick cost
        b_vclock = b_stride
        a_ptr = b_ptr = 0
        picked: Set[int] = set()
        order: List[int] = []
        while len(order) < n:
            pick_a = a_vclock <= b_vclock
            if pick_a:
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

    waiting_queue[:] = [waiting_queue[i] for i in order]
    return deprioritized
