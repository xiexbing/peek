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

"""PeekEngine — queue-depth-aware scheduling + eviction for online serving.

Two modes:

**Online (client pre-sorted)**:
  Requests arrive in DFS group order from PeekDispatcher.  The engine
  detects group boundaries via an O(N) scan, matches ONE representative
  per group against the radix tree (O(G × tree_depth)), copies results
  to all members, and applies queue-aware protection.  No reordering —
  the client's DFS order is trusted.  Total: O(G × tree_depth + N).

**Fallback (unsorted queue)**:
  Full pipeline: per-request prefix matching + queue-depth-aware
  reorder + selective protection.  Used when the client doesn't
  pre-sort (e.g. direct HTTP, no PeekDispatcher).

The key insight: when the client pre-sorts, the server drops from
O(N × tree_depth) to O(G × tree_depth) per scheduling cycle.
"""
from __future__ import annotations

import time as _time
from collections import defaultdict
from typing import Any

from peek.offline.scheduler import detect_sharing_sglang


class CacheStateStore:
    """Bidirectional shared store for server↔client state exchange.

    **Server → Client** (cache fractions):
      The server writes per-group cache fractions after each scheduling
      cycle.  The client reads them to inform DFS ordering.

    **Client → Server** (pending counts):
      The client writes per-group pending request counts on every
      submit/remove.  The server reads the latest snapshot at the
      start of each scheduling cycle to incorporate client-side demand
      into its scoring formula.

      The client's pending count includes requests that are in-flight
      (submitted but not yet completed) — a superset of what the server
      sees in its waiting queue.  This gives the server visibility into
      future demand: if a group has 100 client-side pending but only 20
      in the server's waiting queue, 80 more are coming.

    Thread-safe: server writes/reads from the scheduler thread, client
    writes/reads from the request-submission thread.
    """

    _instance: "CacheStateStore | None" = None

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        # Server → Client: group_key_hash → cache_frac
        self._group_cache_frac: dict[int, float] = {}
        # Client → Server: group_key_hash → pending_count
        self._client_pending: dict[int, int] = {}
        # Server → Client: request IDs admitted this scheduling cycle
        self._scheduled_rids: list[str] = []

    @classmethod
    def get(cls) -> "CacheStateStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- Server → Client: cache fractions ---

    def update(self, group_fracs: dict[int, float]) -> None:
        """Server pushes cache fractions for all groups seen this cycle."""
        with self._lock:
            self._group_cache_frac.update(group_fracs)

    def get_cache_frac(self, group_key_hash: int) -> float:
        """Client reads cache fraction for a group. Returns -1 if unknown."""
        with self._lock:
            return self._group_cache_frac.get(group_key_hash, -1.0)

    def snapshot(self) -> dict[int, float]:
        """Client reads full snapshot."""
        with self._lock:
            return dict(self._group_cache_frac)

    # --- Client → Server: pending counts ---

    def update_client_pending(self, pending: dict[int, int]) -> None:
        """Client pushes current pending counts for all groups."""
        with self._lock:
            self._client_pending = dict(pending)

    def get_client_pending(self) -> dict[int, int]:
        """Server reads client's pending counts at scheduling cycle start."""
        with self._lock:
            return dict(self._client_pending)

    # --- Server → Client: scheduled request IDs (legacy) ---
    # Kept for backward compatibility with tests.  In production,
    # PeekEngine calls dispatcher.remove() directly (zero delay).

    def push_scheduled(self, rids: list[str]) -> None:
        with self._lock:
            self._scheduled_rids = list(rids)

    def pop_scheduled(self) -> list[str]:
        with self._lock:
            rids = self._scheduled_rids
            self._scheduled_rids = []
            return rids


class PeekEngine:
    """Queue-depth-aware scheduling + eviction engine."""

    _KEY_LEN = 64

    def __init__(self, tree_cache: Any, schedule_policy: Any) -> None:
        self.tree_cache = tree_cache
        self.schedule_policy = schedule_policy
        self._profile_count = 0
        self._profile_total = 0.0
        self._target_nodes: list[Any] = []
        self._prev_group_keys: list[tuple] | None = None
        self._cache_state = CacheStateStore.get()
        self._dispatcher: Any = None  # set by PeekDispatcher for direct remove

    @staticmethod
    def is_enabled(tree_cache: Any) -> bool:
        import os
        policy = getattr(tree_cache, "eviction_policy", "")
        if policy in ("queue-aware", "lrfu"):
            return True
        return os.environ.get("PEEK_OFFLINE_ENABLE") == "1"

    def run(self, waiting_queue: list[Any], running_batch: Any = None) -> bool:
        _t0 = _time.perf_counter()

        if self._has_peek_tags(waiting_queue):
            n_groups = self._run_tagged(waiting_queue)
        else:
            n_groups = self._run_full(waiting_queue)

        self._log_profile(_t0, len(waiting_queue), n_groups)
        return True

    # ------------------------------------------------------------------
    # Online fast path: tag-based grouping, O(G × tree_depth + N)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_peek_tags(waiting_queue: list[Any]) -> bool:
        """Check if requests carry peek group tags in their rid."""
        if not waiting_queue:
            return False
        rid = getattr(waiting_queue[0], "rid", "")
        return isinstance(rid, str) and rid.startswith("peek:")

    @staticmethod
    def _parse_peek_rid(rid: str) -> tuple[int, int, str]:
        """Parse 'peek:<group_order>:<group_key>:<original_rid>'."""
        parts = rid.split(":", 3)
        if len(parts) == 4 and parts[0] == "peek":
            return int(parts[1]), int(parts[2]), parts[3]
        return 0, 0, rid

    def _run_tagged(self, waiting_queue: list[Any]) -> int:
        """Fast path: O(G) grouped matching + coordinated scoring.

        Client tags enable O(G) grouping instead of O(N) per-request
        matching.  Server scores groups using the offline-proven
        coordinated formula with fresh cache state:

        score = cache_frac × 10⁴         (exploit cached prefixes)
              - future_refs               (defer groups that can't be consumed this batch)
              + eviction_risk × 10²       (urgent: use at-risk prefixes before LRU evicts)
              + queue_count × 10⁻³        (tiebreaker: larger groups first)

        Steps:
        1. Reset previous protection — O(K × depth)
        2. Group by tag — O(N)
        3. Match ONE rep per group — O(G × tree_depth)
        4. Copy results to all members — O(N)
        5. Score groups using coordinated formula — O(G log G)
        6. Rebuild queue in scored order — O(N)
        7. Adaptive protection (3/4 cache budget) — O(G)
        8. Push cache state to client — O(G)
        """
        tc = self.tree_cache

        # Step 1: Targeted reset
        self._reset_protection(tc)

        # Step 2: Group by tag — O(N)
        tag_groups: dict[int, list[Any]] = {}  # group_key → [reqs]
        for r in waiting_queue:
            rid = getattr(r, "rid", "")
            if isinstance(rid, str) and rid.startswith("peek:"):
                _, group_key, _ = self._parse_peek_rid(rid)
                tag_groups.setdefault(group_key, []).append(r)

        # Step 3 & 4: Match ONE rep per group, copy to members — O(G × D + N)
        from sglang.srt.mem_cache.radix_cache import MatchPrefixParams, RadixKey

        # (group_key, members, queue_count, prefix_len, cache_frac, rep_node, eviction_risk)
        group_stats: list[tuple[int, list, int, int, float, Any, float]] = []

        for group_key, members in tag_groups.items():
            rep = members[0]
            prefix_ids = rep.origin_input_ids + rep.output_ids
            extra_key = rep.extra_key

            match_result = tc.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                )
            )

            # Copy match results to all members — O(group_size)
            for r in members:
                r.prefix_indices = match_result.device_indices
                r.last_node = match_result.last_device_node
                r.last_host_node = match_result.last_host_node
                r.host_hit_length = match_result.host_hit_length
                r._peek_matched = True

            prefix_len = len(rep.origin_input_ids)
            cached_len = len(match_result.device_indices)
            cache_frac = cached_len / max(prefix_len, 1)
            rep_node = match_result.last_device_node

            # Eviction risk: how stale is this cached prefix?
            # Higher = older = closer to LRU eviction.
            if rep_node is not None and cached_len > 0:
                last_access = getattr(rep_node, "last_access_time", 0.0)
                eviction_risk = max(0.0, _time.monotonic() - last_access)
            else:
                eviction_risk = 0.0

            group_stats.append((
                group_key, members, len(members),
                prefix_len, cache_frac, rep_node, eviction_risk,
            ))

        # Step 5: Coordinated scoring with anti-starvation + DFS locality.
        #
        # The score formula determines scheduling priority.  The client's
        # DFS rank (from its trie) determines locality: groups sharing a
        # prefix should be adjacent so the server's radix cache walk is
        # sequential.
        #
        # Sort key: (score descending, dfs_rank ascending).
        # Score is primary (cache-aware, server ground truth).
        # DFS rank is secondary (prefix locality from client trie).
        #
        # The client pushes {group_hash: pending_count} to the shared
        # CacheStateStore.  pending_count includes in-flight requests
        # (submitted but not yet completed) — a superset of what the
        # server sees in its waiting queue.  This lets the server
        # estimate future demand.
        _BATCH_EST = 32
        now = _time.perf_counter()
        client_pending = self._cache_state.get_client_pending()

        # Read DFS rank from client tags.  The rank in the FIRST member
        # of each group reflects the client's latest DFS ordering.
        # Groups sharing a prefix have adjacent ranks.
        group_dfs_rank: dict[int, int] = {}
        for group_key, members, *_ in group_stats:
            rep = members[0]
            rid = getattr(rep, "rid", "")
            if isinstance(rid, str) and rid.startswith("peek:"):
                dfs_rank, _, _ = self._parse_peek_rid(rid)
                group_dfs_rank[group_key] = dfs_rank

        scored: list[tuple[float, int, int, list, int, float, Any]] = []

        for group_key, members, queue_count, prefix_len, cache_frac, rep_node, eviction_risk in group_stats:
            total_pending = client_pending.get(group_key, queue_count)
            future_refs = max(0, total_pending - _BATCH_EST)

            max_wait = 0.0
            for r in members:
                ts = getattr(r, "time_stats", None)
                arr = getattr(ts, "wait_queue_entry_time", 0.0) if ts else 0.0
                if arr > 0:
                    max_wait = max(max_wait, now - arr)

            score = (
                cache_frac * 1e4       # exploit: cached groups first
                - future_refs          # defer: don't over-invest
                + eviction_risk * 1e2  # urgent: use before LRU evicts
                + max_wait * 1e2       # anti-starvation
                + queue_count * 1e-3   # tiebreaker: larger groups first
            )
            dfs_rank = group_dfs_rank.get(group_key, 0)
            scored.append((score, dfs_rank, group_key, members, prefix_len, cache_frac, rep_node))

        # Primary: score descending.  Secondary: DFS rank ascending
        # (adjacent DFS ranks = shared prefix = cache-sequential).
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Step 6: Rebuild queue in scored order — O(N)
        waiting_queue.clear()
        group_fracs: dict[int, float] = {}

        for _, _, group_key, members, _, cache_frac, _ in scored:
            waiting_queue.extend(members)
            group_fracs[group_key] = cache_frac

        # Step 7: Adaptive protection — protect top 3/4 of cache capacity.
        # Leave 25% for LRU rotation so new groups can enter cache.
        num_groups = len(scored)
        total_cached = 0
        try:
            total_cached = tc.total_size()
        except Exception:
            pass

        if num_groups > 0 and total_cached > 0:
            avg_prefix_len = max(1, sum(s[4] for s in scored) / num_groups)
            cache_capacity_groups = int(total_cached / avg_prefix_len)
        else:
            cache_capacity_groups = num_groups

        protect_k = min(num_groups, max(1, cache_capacity_groups * 3 // 4))

        protected_nodes: list[Any] = []
        for i, (_, _, _, members, _, cache_frac, rep_node) in enumerate(scored):
            if i >= protect_k:
                break
            if rep_node is not None and cache_frac > 0:
                tc.inc_queue_ref(rep_node)
                protected_nodes.append(rep_node)

        tc._peek_prev_protected = protected_nodes

        # Step 8: Push cache state to client
        self._cache_state.update(group_fracs)

        # Step 9: Remove scheduled requests from client trie.
        # After reordering, PrefillAdder will pick from the front of the
        # queue up to _BATCH_EST requests.  Remove them from the
        # dispatcher's trie directly — zero delay, same call stack.
        if self._dispatcher is not None:
            for r in waiting_queue[:_BATCH_EST]:
                rid = getattr(r, "rid", "")
                if isinstance(rid, str) and rid.startswith("peek:"):
                    _, _, orig_rid = self._parse_peek_rid(rid)
                    info = self._dispatcher._rid_to_remove_info.pop(orig_rid, None)
                    if info is not None:
                        self._dispatcher.remove(info[0], info[1])

        return num_groups

    # ------------------------------------------------------------------
    # Fallback: full pipeline for unsorted queues
    # ------------------------------------------------------------------

    def _run_full(self, waiting_queue: list[Any]) -> int:
        """Full pipeline: prefix match + reorder + protect.

        Used when requests are not client pre-sorted.
        """
        from sglang.srt.managers.schedule_policy import (
            CacheAwarePolicy,
            SchedulePolicy,
        )

        tc = self.tree_cache

        # Step 1: Targeted reset
        self._reset_protection(tc)

        # Step 2: Per-request prefix matching
        deprioritized = self.schedule_policy._compute_prefix_matches(
            waiting_queue, CacheAwarePolicy.LPM,
        )

        # Step 3: Queue-depth-aware reorder
        if len(waiting_queue) > 1 and detect_sharing_sglang(waiting_queue):
            n_groups = self._queue_depth_reorder(
                waiting_queue, deprioritized,
            )
        else:
            SchedulePolicy._sort_by_longest_prefix(
                waiting_queue, deprioritized,
            )
            n_groups = len(waiting_queue)
            self._target_nodes = []

        # Step 4: Selective protection
        protected_nodes: list[Any] = []
        for node in self._target_nodes:
            tc.inc_queue_ref(node)
            protected_nodes.append(node)
        tc._peek_prev_protected = protected_nodes

        # Step 5: Push cache state to client — O(N)
        kl = self._KEY_LEN
        group_fracs: dict[int, float] = {}
        for r in waiting_queue:
            ids = getattr(r, "origin_input_ids", None)
            if ids is None:
                continue
            key = tuple(ids[:kl]) if len(ids) >= kl else tuple(ids)
            group_key_hash = hash(key) & 0xFFFFFFFF
            if group_key_hash not in group_fracs:
                prefix_len = len(ids)
                cached_len = len(getattr(r, "prefix_indices", []))
                group_fracs[group_key_hash] = cached_len / max(prefix_len, 1)
        self._cache_state.update(group_fracs)

        return n_groups

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _reset_protection(self, tc: Any) -> None:
        """Targeted reset of previous cycle's protected nodes.
        O(K × depth) instead of walking the entire tree."""
        prev = getattr(tc, "_peek_prev_protected", None)
        if prev is not None:
            for node in prev:
                walk = node
                while walk is not None and walk != tc.root_node:
                    walk.queue_ref_count = 0
                    walk = walk.parent
        else:
            tc.reset_all_queue_refs()

    def _queue_depth_reorder(
        self,
        waiting_queue: list[Any],
        deprioritized: set,
    ) -> int:
        """Queue-depth-aware reorder: invest in what's needed, not just cached.

        Groups requests by prefix, computes a target cache set based on
        investment value (queue_count × prefix_len), then schedules:
          1. Cached target groups first (exploit)
          2. Uncached target groups next (invest — one request populates cache)
          3. Non-target groups last (opportunistic)
        """
        kl = self._KEY_LEN

        # Step 1: Group by prefix key — O(N)
        groups: dict[tuple, list[Any]] = defaultdict(list)
        for r in waiting_queue:
            ids = getattr(r, "origin_input_ids", None)
            if ids is not None and len(ids) >= kl:
                key = tuple(ids[:kl])
            elif ids is not None and len(ids) > 0:
                key = tuple(ids)
            else:
                key = ()
            groups[key].append(r)

        # Step 2: Per-group stats — O(N)
        group_stats: list[tuple[tuple, list, int, int, float, Any]] = []
        deprioritized_reqs: list[Any] = []

        for key, members in groups.items():
            normal = []
            for r in members:
                if hasattr(r, "rid") and r.rid in deprioritized:
                    deprioritized_reqs.append(r)
                else:
                    normal.append(r)
            if not normal:
                continue

            queue_count = len(normal)
            prefix_len = len(getattr(normal[0], "origin_input_ids", []))
            total_cached = sum(
                len(getattr(r, "prefix_indices", []))
                for r in normal
            )
            cache_frac = total_cached / max(prefix_len * queue_count, 1)
            rep_node = getattr(normal[0], "last_node", None)

            group_stats.append(
                (key, normal, queue_count, prefix_len, cache_frac, rep_node)
            )

        # Step 3: Compute target cache set — O(G log G)
        group_stats.sort(
            key=lambda g: g[2] * g[3],  # queue_count × prefix_len
            reverse=True,
        )

        cache_budget = 0
        try:
            cache_budget = int(self.tree_cache.total_size() * 0.75)
        except Exception:
            pass

        target_keys: set[tuple] = set()
        budget_used = 0
        if cache_budget > 0:
            for key, _, _, prefix_len, _, _ in group_stats:
                if budget_used + prefix_len > cache_budget and target_keys:
                    break
                target_keys.add(key)
                budget_used += prefix_len
        else:
            for key, _, _, _, _, _ in group_stats:
                target_keys.add(key)

        # Step 4: Score for scheduling order — O(G log G)
        scored: list[tuple[float, list[Any]]] = []
        target_nodes: list[Any] = []

        for key, normal, queue_count, prefix_len, cache_frac, rep_node in group_stats:
            is_target = key in target_keys
            target_bonus = 10000 if is_target else 0
            score = target_bonus + cache_frac * 1000 + queue_count
            scored.append((score, normal))

            if is_target and rep_node is not None:
                if len(getattr(normal[0], "prefix_indices", [])) > 0:
                    target_nodes.append(rep_node)

        self._target_nodes = target_nodes

        # Step 5: Sort and rebuild queue
        scored.sort(key=lambda x: x[0], reverse=True)

        waiting_queue.clear()
        for _, members in scored:
            waiting_queue.extend(members)
        waiting_queue.extend(deprioritized_reqs)

        return len(scored)

    def _log_profile(
        self, t0: float, queue_len: int, num_groups: int,
    ) -> None:
        elapsed = _time.perf_counter() - t0
        self._profile_count += 1
        self._profile_total += elapsed
        if self._profile_count % 100 == 0:
            avg = self._profile_total / self._profile_count * 1000
            print(
                f"  [peek-engine] calls={self._profile_count} "
                f"avg={avg:.2f}ms last={elapsed*1000:.2f}ms "
                f"N={queue_len} G={num_groups}",
                flush=True,
            )
