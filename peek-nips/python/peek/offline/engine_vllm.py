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

"""VllmPeekEngine -- coordinated scheduling + eviction for vLLM.

Mirrors :class:`PeekEngine` (``engine.py``) for SGLang but uses vLLM's
block-hash based prefix caching instead of a radix tree.

Owns the full loop:
  1. Detect prefix sharing among waiting requests
  2. Group requests by shared block-hash prefix
  3. Score groups against block-pool cache state (cache hit, eviction risk)
  4. Reorder waiting queue by score (ready groups first, buffering last)
  5. Protect top-K groups from eviction via queue ref counting

Works identically for online (Poisson arrivals) and offline (full batch)
modes -- it operates on whatever is in the waiting queue at each cycle.
"""
from __future__ import annotations

import time as _time
from collections import defaultdict
from typing import Any

from peek.offline.scheduler import (
    detect_sharing,
    vllm_cached_prefix_length,
)


class VllmPeekEngine:
    """Coordinated scheduling + eviction engine for vLLM.

    Instantiated once per vLLM scheduler when ``PEEK_OFFLINE_SERVER_REORDER=1``
    and queue-aware eviction is active.  Called from :func:`vllm_on_schedule`
    via a lazy-init hook.
    """

    _KEY_LEN = 8         # block-hash grouping key length (~64 tokens)
    _BATCH_EST = 32      # rough batch-size estimate for future_refs

    def __init__(self, block_pool: Any, group_ids: list[int]) -> None:
        self.block_pool = block_pool
        self.group_ids = group_ids
        self._prev_protected: list[tuple[Any, list[int]]] | None = None
        self._profile_count = 0
        self._profile_total = 0.0
        # group key -> monotonic timestamp when group first appeared
        self._group_first_seen: dict[tuple, float] = {}
        # group key -> monotonic timestamp when group was last in the queue
        self._group_last_seen: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, waiting: list[Any]) -> None:
        """Full Peek scheduling cycle.  Mutates *waiting* in-place."""
        _t0 = _time.perf_counter()

        # Partition: new (uncomputed) vs in-progress
        new_reqs = [r for r in waiting
                    if r.num_computed_tokens == 0
                    and len(r.block_hashes) > 0]
        if len(new_reqs) <= 1:
            return

        # Detect sharing via first block hash
        first_hashes = [r.block_hashes[0] for r in new_reqs]
        if not detect_sharing(first_hashes):
            return

        other = [r for r in waiting
                 if r.num_computed_tokens != 0
                 or len(r.block_hashes) == 0]

        # --- Peek pipeline ---
        self._reset_queue_refs()

        groups = self._group_requests(new_reqs)
        scored = self._match_and_score_groups(groups)
        reordered = self._reorder_by_score(scored, groups, len(waiting))
        self._protect_top_groups(scored)

        waiting.clear()
        waiting.extend(reordered)
        waiting.extend(other)

        self._log_profile(_t0, len(waiting), len(groups))

    # ------------------------------------------------------------------
    # Scheduling pipeline
    # ------------------------------------------------------------------

    def _group_requests(
        self, new_reqs: list[Any],
    ) -> dict[tuple, list[Any]]:
        """Hash first ``_KEY_LEN`` block hashes to partition requests."""
        kl = self._KEY_LEN
        groups: dict[tuple, list[Any]] = defaultdict(list)
        for r in new_reqs:
            bh = r.block_hashes
            if len(bh) >= kl:
                key = tuple(bh[:kl])
            elif len(bh) > 0:
                key = tuple(bh)
            else:
                key = ()
            groups[key].append(r)
        return groups

    def _match_and_score_groups(
        self,
        groups: dict[tuple, list[Any]],
    ) -> list[tuple[float, tuple, list[Any], int]]:
        """Score each group by tokens_saved (cache_frac x total_blocks x group_size).

        Returns list of (score, key, members, cached_blocks).
        """
        now = _time.monotonic()
        bp = self.block_pool
        gids = self.group_ids

        group_data: list[tuple[float, tuple, list[Any], int]] = []

        for key, members in groups.items():
            rep = members[0]
            cached_blocks = vllm_cached_prefix_length(
                bp, list(rep.block_hashes), gids,
            )
            total_blocks = len(rep.block_hashes)
            group_size = len(members)

            cache_frac = cached_blocks / max(total_blocks, 1)
            future_refs = max(0, group_size - self._BATCH_EST)

            score = (
                cache_frac * 1e4
                - future_refs
                + group_size * 1e-3
            )
            group_data.append((score, key, members, cached_blocks))

            # Update last-seen
            self._group_last_seen[key] = now

        return group_data

    def _reorder_by_score(
        self,
        scored: list[tuple[float, tuple, list[Any], int]],
        groups: dict[tuple, list[Any]],
        queue_depth: int = 0,
    ) -> list[Any]:
        """Reorder: ready groups first (by score), buffering groups last.

        Buffering thresholds adapt to queue depth:
          - Shallow (< 8):  no buffering -- schedule immediately
          - Medium  (8-32): light buffering (min_group=2, wait=10ms)
          - Deep    (> 32): aggressive buffering (min_group=4, wait=30ms)
        """
        now = _time.monotonic()

        # Adaptive thresholds based on queue depth
        if queue_depth < 8:
            threshold = 1
            max_wait_s = 0.0
        elif queue_depth <= 32:
            threshold = 2
            max_wait_s = 0.010
        else:
            threshold = 4
            max_wait_s = 0.030

        # Update first-seen timestamps; clean up groups no longer in queue
        active_keys = set(groups.keys())
        for key in list(self._group_first_seen):
            if key not in active_keys:
                del self._group_first_seen[key]
        for key in active_keys:
            if key not in self._group_first_seen:
                self._group_first_seen[key] = now

        # Partition into ready vs buffering
        ready: list[tuple[float, tuple, list[Any], int]] = []
        buffering: list[tuple[float, tuple, list[Any], int]] = []
        for item in scored:
            score, key, members, cached_blocks = item
            age = now - self._group_first_seen.get(key, now)
            if len(members) >= threshold or age >= max_wait_s:
                ready.append(item)
            else:
                buffering.append(item)

        # If nothing is ready, promote the highest-scoring buffering group
        # to avoid stalling the scheduler entirely
        if not ready and buffering:
            buffering.sort(key=lambda x: x[0], reverse=True)
            ready.append(buffering.pop(0))

        ready.sort(key=lambda x: x[0], reverse=True)
        buffering.sort(key=lambda x: x[0], reverse=True)

        reordered: list[Any] = []
        for _, _, members, _ in ready:
            # Within each group, shortest sequence first (better packing)
            members.sort(key=lambda r: len(r.block_hashes))
            reordered.extend(members)
        for _, _, members, _ in buffering:
            members.sort(key=lambda r: len(r.block_hashes))
            reordered.extend(members)

        return reordered

    # ------------------------------------------------------------------
    # Eviction coordination
    # ------------------------------------------------------------------

    def _reset_queue_refs(self) -> None:
        """Targeted reset of previously protected blocks."""
        bp = self.block_pool
        if self._prev_protected is not None:
            for block_hash, gids in self._prev_protected:
                blocks = bp.get_cached_block(block_hash, gids)
                if blocks:
                    for b in blocks:
                        b.queue_ref_count = 0
        else:
            bp.reset_queue_ref_counts()

    def _protect_top_groups(
        self, scored: list[tuple[float, tuple, list[Any], int]],
    ) -> None:
        """Adaptive eviction protection via inc_queue_ref for top-K groups.

        Limits protection to 50% of cache capacity and skips stale
        singleton groups to prevent ossification.
        """
        bp = self.block_pool
        gids = self.group_ids
        now = _time.monotonic()

        # Estimate cache capacity in prefix-sized slots
        num_groups = len(scored)
        total_pool_blocks = len(getattr(bp, "blocks", []))
        if num_groups > 0 and total_pool_blocks > 0:
            avg_prefix_blocks = max(1, sum(
                len(members[0].block_hashes) for _, _, members, _ in scored
            ) // num_groups)
            capacity_groups = total_pool_blocks // avg_prefix_blocks
        else:
            capacity_groups = num_groups

        # Protect at most 2/3 of cache capacity -- leave 1/3 for rotation
        protect_k = min(num_groups, max(1, capacity_groups * 2 // 3))

        # Recency gate: only protect groups seen recently
        recency_cutoff = 0.5  # seconds

        protected: list[tuple[Any, list[int]]] = []
        for i, (_, key, members, cached_blocks) in enumerate(scored):
            if i >= protect_k:
                break
            if cached_blocks == 0:
                continue
            # Skip protecting stale singleton groups
            first_seen = self._group_first_seen.get(key, now)
            if now - first_seen > recency_cutoff and len(members) <= 1:
                continue
            rep = members[0]
            for j, bh in enumerate(rep.block_hashes):
                if j >= cached_blocks:
                    break
                blocks = bp.get_cached_block(bh, gids)
                if blocks:
                    bp.inc_queue_ref_count(blocks)
                    protected.append((bh, gids))

        self._prev_protected = protected

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def _log_profile(
        self, t0: float, queue_len: int, num_groups: int,
    ) -> None:
        elapsed = _time.perf_counter() - t0
        self._profile_count += 1
        self._profile_total += elapsed
        if self._profile_count % 100 == 0:
            avg = self._profile_total / self._profile_count * 1000
            print(
                f"  [peek-vllm-engine] calls={self._profile_count} "
                f"avg={avg:.2f}ms last={elapsed*1000:.2f}ms "
                f"N={queue_len} G={num_groups}",
                flush=True,
            )
