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

"""vllm integration: inject peek's pending-tree and (optionally) its LPM/CLPM
scheduler into the vllm v1 scheduler.

Gated by the same env vars as the sglang integration so benchmark drivers can
flip engines with a single export:

    PEEK_ONLINE_ENABLED=1   python -m vllm.entrypoints.openai.api_server ...   # eviction+sched
    PEEK_ONLINE_SCHEDULER=1 python -m vllm.entrypoints.openai.api_server ...   # sched only
    PEEK_ONLINE_CLPM=1      python -m vllm.entrypoints.openai.api_server ...   # cluster-LPM
    # no flags:      runs vanilla vllm

Integration surface (vllm 0.19.1, v1 engine):

    vllm.v1.core.sched.scheduler.Scheduler.schedule
        -> our patch syncs peek's PendingTree against self.waiting, optionally
           reorders self.waiting via peek's LPM/CLPM policy, then delegates.
    vllm.v1.core.sched.scheduler.Scheduler.add_request
        -> sync-on-arrival fast path (peek tree gains the rid immediately).
    vllm.v1.core.sched.scheduler.Scheduler.finish_requests
        -> drop completed/aborted rids from peek tree.

We hook by monkey-patching the Scheduler class at module import time. Call
sites:
    import peek.online.engines.vllm.patch_hook         # once per scheduler process
Or set PYTHONSTARTUP / sitecustomize.py so the import fires before vllm
instantiates Scheduler.

Eviction axis (PEEK_ONLINE_EVICTION=1) is wired by monkey-patching
`BlockPool.get_new_blocks`. vllm v1's stock victim selection is
`free_block_queue.popleft_n(N)` (oldest free blocks). Peek replaces that
with a demand-aware pick: scan the free queue and prefer blocks whose
hash has zero pending-request demand. Only when fewer than N demand-free
blocks are available within the scan cap do we fall back to lowest-demand
blocks (and beyond that, to stock LRU). The inverted index
`block_hash -> demand` is maintained by the same sync loop that maintains
the PendingTree, so eviction overhead is amortized.

Caveats:
  - Only the `plain` eviction mode is supported on vllm. The
    demand_recency / demand_cluster / demand_decay variants in peek's
    sglang path rely on per-node last_access_time and ancestor walks
    that don't translate to vllm's hash-based block model.
  - `PEEK_ONLINE_EVICTION_SCAN_CAP` (default 4xN, min 256) bounds the per-call
    free-queue scan to keep overhead bounded on large pools.
"""

from __future__ import annotations

import logging
import os
import time as _time


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


# Legacy back-compat: PEEK_ONLINE_ENABLED=1 -> scheduler + (tried) eviction,
# PEEK_ONLINE_ENABLED=full -> same (no distinction on vllm since we have no LPM-full
# variant beyond peek_lpm).
_LEGACY = os.environ.get("PEEK_ONLINE_ENABLED", "").lower()
_LEGACY_ANY = _LEGACY in ("1", "true", "yes", "on", "full")

_EVICTION = _flag("PEEK_ONLINE_EVICTION") or _LEGACY_ANY
_SCHEDULER = (
    _flag("PEEK_ONLINE_SCHEDULER") or _LEGACY_ANY
    or _flag("PEEK_ONLINE_LPM") or _flag("PEEK_ONLINE_CLPM")
)
_PEEK_LPM = _flag("PEEK_ONLINE_LPM")
_PEEK_CLPM = _flag("PEEK_ONLINE_CLPM")
_PEEK_CLPM_WINDOW_MS = int(os.environ.get("PEEK_ONLINE_CLPM_WINDOW_MS", "500"))
_PEEK_CLPM_AGE_ALPHA = float(os.environ.get("PEEK_ONLINE_CLPM_AGE_ALPHA", "0"))
_PEEK_CLPM_BIGLANE_SHARE = float(os.environ.get("PEEK_ONLINE_CLPM_BIGLANE_SHARE", "0.7"))
_PEEK_CLPM_GROUP_MAJOR = _flag("PEEK_ONLINE_CLPM_GROUP_MAJOR")
_PEEK_CLPM_DYNAMIC_LANE = _flag("PEEK_ONLINE_CLPM_DYNAMIC_LANE")
_PEEK_CLPM_SLO_BUDGET_S = float(os.environ.get("PEEK_ONLINE_CLPM_SLO_BUDGET_S", "2.0"))

# Mirror sglang's LPM>N FCFS fallback. Stock vllm FCFS is already arrival-
# ordered in the deque, so "fallback" just means we skip reordering above the
# threshold to keep scheduler overhead bounded.
_LPM_FB_THRESHOLD = int(os.environ.get("PEEK_ONLINE_LPM_FALLBACK_THRESHOLD", "128"))

# In-batch pioneer/sibling thresholds. vllm has no equivalent constant; use
# sglang defaults so A/B runs compare apples-to-apples.
_CHECK_THRESHOLD = int(os.environ.get("PEEK_ONLINE_IN_BATCH_CHECK", "32"))
_DEPRIO_THRESHOLD = int(os.environ.get("PEEK_ONLINE_IN_BATCH_DEPRIO", "32"))

_PHASE_TRACKING = _flag("PEEK_ONLINE_PHASE_TRACKING") or _SCHEDULER
_PHASE_DUMP_PATH_TEMPLATE = os.environ.get(
    "PEEK_ONLINE_PHASE_DUMP_PATH", "/tmp/peek_phases_{pid}.json"
)
_PROFILE = _flag("PEEK_ONLINE_PROFILE")
_VALIDATE = _flag("PEEK_ONLINE_VALIDATE")

# Per-call cap on free-queue scan when picking eviction victims. None ->
# default to max(4*N, _MIN_SCAN_CAP). Caps keep overhead bounded even when
# the free queue is enormous.
_EVICTION_SCAN_CAP = os.environ.get("PEEK_ONLINE_EVICTION_SCAN_CAP")
_EVICTION_SCAN_CAP = int(_EVICTION_SCAN_CAP) if _EVICTION_SCAN_CAP else 0
_EVICTION_MIN_SCAN_CAP = 256

# Eviction priority formula. Mirrors sglang's PEEK_ONLINE_EVICTION_MODE knob:
#   plain    -- priority = demand                                 (binary protect)
#   cluster  -- priority = demand x (1 + max_chain_depth)         (depth-of-sharing)
#   recency  -- priority = demand − W x age_seconds                (linear aging)
#   decay    -- priority = demand x exp(−age_seconds / τ)          (exponential aging)
#
# `cluster` needs `_block_hash_max_depth[h]` (deepest position in any pending
# rid's prefix chain) -- maintained on _bump_rid, cleared on demand-zero.
# `recency` and `decay` need a per-block last-access timestamp that vllm
# doesn't track natively. Peek fabricates it by patching `BlockPool.touch`
# (cache-hit stamps) and `BlockPool.cache_full_blocks` (initial-cache stamps).
_EVICTION_MODE = os.environ.get("PEEK_ONLINE_EVICTION_MODE", "plain").lower()
if _EVICTION_MODE not in ("plain", "cluster", "recency", "decay"):
    import warnings
    warnings.warn(
        f"peek.vllm: PEEK_ONLINE_EVICTION_MODE={_EVICTION_MODE!r} not recognized. "
        "Falling back to 'plain'.",
        stacklevel=2,
    )
    _EVICTION_MODE = "plain"
# recency: tokens-at-risk subtracted per second of age. 100 ≈ sglang default.
_EVICTION_RECENCY_W = float(os.environ.get("PEEK_ONLINE_EVICTION_RECENCY_W", "100"))
# decay: exponential half-life parameter (seconds).
_EVICTION_DECAY_TAU = float(os.environ.get("PEEK_ONLINE_EVICTION_DECAY_TAU", "30"))
# True iff the mode needs per-block last_access_time (drives BlockPool patches).
_EVICTION_NEEDS_LAST_ACCESS = _EVICTION_MODE in ("recency", "decay")
_VALIDATE_PATH = os.environ.get(
    "PEEK_ONLINE_VALIDATE_PATH", "/tmp/peek_validate_{pid}.json"
)

_ENABLED = _SCHEDULER or _PHASE_TRACKING or _EVICTION
_log = logging.getLogger("peek.vllm")

_vstats: dict = {
    "sync_checks": 0,
    "sync_extra_in_peek": 0,
    "sync_missing_in_peek": 0,
    "sync_token_mismatch": 0,
    "first_violation_logged": False,
}
_prof: dict = {
    "sync_calls": 0, "sync_skipped": 0, "sync_ns": 0,
    "insert_calls": 0, "insert_ns": 0,
    "discard_calls": 0, "discard_ns": 0,
    "calc_priority_calls": 0, "calc_priority_fallbacks": 0, "calc_priority_ns": 0,
    "evict_calls": 0, "evict_demand_aware": 0, "evict_fallback": 0,
    "evict_ns": 0, "evict_scanned": 0, "evict_zero_demand_picked": 0,
}


def _install() -> None:
    """Install monkey-patches on vllm v1 scheduler. Idempotent per-process:
    guards with a flag on the Scheduler class so a second import is a no-op."""
    from peek import PendingTree

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.request_queue import (
        FCFSRequestQueue,
        SchedulingPolicy,
    )
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_block_hash

    if getattr(Scheduler, "_peek_installed", False):
        return
    Scheduler._peek_installed = True

    peek_tree = PendingTree()

    # rid (vllm uses str) -> u64 interned rid for peek.
    rid_interner: dict[str, int] = {}
    _counter = [0]

    def _intern(rid_str: str) -> int:
        rid_int = rid_interner.get(rid_str)
        if rid_int is None:
            _counter[0] += 1
            rid_int = _counter[0]
            rid_interner[rid_str] = rid_int
        return rid_int

    _seen_this_session: set[int] = set()
    _last_rids: set = set()
    # Per-rid phase timings keyed by str rid; same schema as sglang hook.
    _phase_timings: dict = {}
    # Eviction inverted index: cached block hash -> number of pending rids
    # whose prefix passes through that block. Maintained by sync alongside
    # the PendingTree. Also a per-rid hash list so discard can reverse it.
    # Keyed by raw BlockHash (bytes), not BlockHashWithGroupId -- that's what
    # Request.block_hashes contains and what we'd recover from a cached
    # block via get_block_hash(blk.block_hash).
    _block_hash_demand: dict = {}
    # Cluster mode: max position-in-prefix-chain (1-indexed) at which any
    # pending rid currently has this hash. Stale-high tolerated; cleared
    # when demand[h] hits 0.
    _block_hash_max_depth: dict = {}
    # Recency / decay modes: per-block-hash last-access wall-clock seconds.
    # Stamped by patched BlockPool.touch + cache_full_blocks. Stale entries
    # (hash no longer cached anywhere) are harmless dict bloat; they only
    # matter if a future block re-uses the same hash, in which case the
    # stale timestamp is *older* than the new block, so we'd over-evict
    # the new block -- bounded harm (still a valid LRU-ish choice).
    _block_hash_last_access: dict = {}
    _rid_to_hashes: dict[int, list] = {}

    def _bump_rid(req, rid_int: int) -> None:
        """Snapshot req.block_hashes and increment demand counts. In cluster
        mode also bump per-hash max-chain-depth."""
        bh = getattr(req, "block_hashes", None)
        if not bh:
            _rid_to_hashes[rid_int] = []
            return
        snap = list(bh)
        for i, h in enumerate(snap):
            _block_hash_demand[h] = _block_hash_demand.get(h, 0) + 1
            if _EVICTION_MODE == "cluster":
                d = i + 1
                prev = _block_hash_max_depth.get(h, 0)
                if d > prev:
                    _block_hash_max_depth[h] = d
        _rid_to_hashes[rid_int] = snap

    def _drop_rid(rid_int: int) -> None:
        """Decrement demand counts for an rid leaving the waiting set.
        In cluster mode also drop the depth entry when demand hits 0
        (lazy invalidation -- depth can be stale-high while demand > 0)."""
        snap = _rid_to_hashes.pop(rid_int, None)
        if not snap:
            return
        for h in snap:
            n = _block_hash_demand.get(h, 0) - 1
            if n <= 0:
                _block_hash_demand.pop(h, None)
                if _EVICTION_MODE == "cluster":
                    _block_hash_max_depth.pop(h, None)
            else:
                _block_hash_demand[h] = n

    def _req_tokens(req) -> list[int]:
        """Get a request's full token sequence (prompt + output so far)."""
        prompt = list(req.prompt_token_ids) if req.prompt_token_ids else []
        output = list(req.output_token_ids) if req.output_token_ids else []
        return prompt + output

    def _sync(waiting_iter) -> None:
        """Diff peek tree against the scheduler's waiting set. Fast path:
        skip if the rid set hasn't changed since last sync."""
        t0 = _time.perf_counter_ns() if _PROFILE else 0
        wq_list = list(waiting_iter)
        if len(wq_list) == len(_last_rids):
            current_strs = {r.request_id for r in wq_list}
            if current_strs == _last_rids:
                if _PROFILE:
                    _prof["sync_calls"] += 1
                    _prof["sync_skipped"] += 1
                    _prof["sync_ns"] += _time.perf_counter_ns() - t0
                return
        else:
            current_strs = {r.request_id for r in wq_list}

        now_wall = _time.time() if _PHASE_TRACKING else 0.0
        current_ints: set[int] = set()
        for req in wq_list:
            rid_int = _intern(req.request_id)
            current_ints.add(rid_int)
            if rid_int not in _seen_this_session:
                tokens = _req_tokens(req)
                if _PROFILE:
                    t1 = _time.perf_counter_ns()
                try:
                    peek_tree.insert(rid_int, tokens)
                except ValueError:
                    peek_tree.discard(rid_int)
                    _drop_rid(rid_int)
                    peek_tree.insert(rid_int, tokens)
                if _PROFILE:
                    _prof["insert_calls"] += 1
                    _prof["insert_ns"] += _time.perf_counter_ns() - t1
                _seen_this_session.add(rid_int)
                if _EVICTION:
                    _bump_rid(req, rid_int)
                if _PHASE_TRACKING:
                    _phase_timings[req.request_id] = {"arrive_ts": now_wall}
        for rid_int in list(_seen_this_session):
            if rid_int not in current_ints:
                if _PROFILE:
                    t1 = _time.perf_counter_ns()
                peek_tree.discard(rid_int)
                if _EVICTION:
                    _drop_rid(rid_int)
                if _PROFILE:
                    _prof["discard_calls"] += 1
                    _prof["discard_ns"] += _time.perf_counter_ns() - t1
                _seen_this_session.discard(rid_int)

        if _PHASE_TRACKING:
            for rid_str in _last_rids - current_strs:
                slot = _phase_timings.get(rid_str)
                if slot is not None and "pick_ts" not in slot:
                    slot["pick_ts"] = now_wall

        _last_rids.clear()
        _last_rids.update(current_strs)
        if _PROFILE:
            _prof["sync_calls"] += 1
            _prof["sync_ns"] += _time.perf_counter_ns() - t0

    def _validate_sync(wq_list) -> None:
        wq_ints = set()
        missing = 0
        token_mm = 0
        for req in wq_list:
            rid_int = rid_interner.get(req.request_id)
            if rid_int is None:
                missing += 1
                continue
            wq_ints.add(rid_int)
            expected = _req_tokens(req)
            stored = peek_tree.tokens(rid_int)
            if stored is None:
                missing += 1
            elif stored != expected:
                token_mm += 1
        extra = sum(1 for r in _seen_this_session if r not in wq_ints)
        _vstats["sync_checks"] += 1
        if missing or extra or token_mm:
            _vstats["sync_missing_in_peek"] += missing
            _vstats["sync_extra_in_peek"] += extra
            _vstats["sync_token_mismatch"] += token_mm
            if not _vstats["first_violation_logged"]:
                _vstats["first_violation_logged"] = True
                _log.warning(
                    "peek VALIDATE: first sync violation -- missing=%d extra=%d "
                    "token_mm=%d wq_size=%d peek_tracked=%d",
                    missing, extra, token_mm, len(wq_list),
                    len(_seen_this_session),
                )

    # --- Main-hit from vllm's KV cache manager ------------------------------
    # vllm's equivalent of sglang RadixCache.match_prefix is the coordinator's
    # find_longest_cache_hit(block_hashes, max_cache_hit_length), which returns
    # (blocks, num_new_computed_tokens). We call it once per waiting req with
    # prefix caching enabled; result is the main_hit in token units.

    def _compute_main_hits(scheduler, wq_list) -> dict[str, int]:
        kvm = getattr(scheduler, "kv_cache_manager", None)
        if kvm is None or not getattr(kvm, "enable_caching", False):
            return {}
        out: dict[str, int] = {}
        for r in wq_list:
            if getattr(r, "skip_reading_prefix_cache", False):
                out[r.request_id] = 0
                continue
            bh = getattr(r, "block_hashes", None)
            if not bh:
                out[r.request_id] = 0
                continue
            try:
                max_len = max(0, r.num_tokens - 1)
                _blocks, num_hit = kvm.coordinator.find_longest_cache_hit(
                    bh, max_len,
                )
                out[r.request_id] = int(num_hit)
            except Exception as e:
                _log.debug("peek: find_longest_cache_hit failed for %s: %s",
                           r.request_id, e)
                out[r.request_id] = 0
        return out

    # --- Reorder waiting queue via peek LPM/CLPM ---------------------------

    def _reorder_waiting(scheduler) -> None:
        tcp = _time.perf_counter_ns() if _PROFILE else 0
        if not _SCHEDULER:
            return
        wq = scheduler.waiting
        # Only FCFS queue is a plain deque we can reorder. PriorityRequestQueue
        # carries user-set priority semantics we must not break -- skip.
        if not isinstance(wq, FCFSRequestQueue):
            if _PROFILE:
                _prof["calc_priority_calls"] += 1
                _prof["calc_priority_fallbacks"] += 1
                _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
            return
        n = len(wq)
        if n <= 1:
            if _PROFILE:
                _prof["calc_priority_calls"] += 1
                _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
            return
        # Mirror sglang LPM's long-queue FCFS fallback.
        if _PEEK_LPM and n > _LPM_FB_THRESHOLD:
            if _PROFILE:
                _prof["calc_priority_calls"] += 1
                _prof["calc_priority_fallbacks"] += 1
                _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
            return

        wq_list = list(wq)
        # Give each req a peek-compatible `prefix_indices` shim so
        # lpm_integration's fallback path (len(pi)) still works if our
        # main_hits dict is missing an entry. We attach a property-like
        # dynamic attribute only when necessary; lpm_integration reads
        # r.prefix_indices -- a bare list of length=main_hit is enough.
        main_hits_by_str = _compute_main_hits(scheduler, wq_list)
        rid_to_int = {r.request_id: _intern(r.request_id) for r in wq_list}
        # Convert str-keyed dict to int-keyed for lpm_integration.
        main_hits_by_int = {
            rid_to_int[rs]: v for rs, v in main_hits_by_str.items()
        }

        # Shims: lpm_integration expects attrs rid / prefix_indices /
        # origin_input_ids / output_ids. vllm's Request has request_id /
        # prompt_token_ids / output_token_ids. Wrap each req in a cheap
        # adapter that exposes the sglang names without copying tokens.
        adapters = [_ReqAdapter(r, main_hits_by_str.get(r.request_id, 0))
                    for r in wq_list]

        from peek.online.lpm_integration import (
            peek_sort_inplace,
            peek_clpm_sort_inplace,
        )

        if _PEEK_CLPM:
            arr_ts = {
                rid_str: slot.get("arrive_ts", 0.0)
                for rid_str, slot in _phase_timings.items()
                if "arrive_ts" in slot
            } if _PHASE_TRACKING else None
            peek_clpm_sort_inplace(
                adapters,
                rid_to_int,
                peek_tree,
                window_ms=_PEEK_CLPM_WINDOW_MS,
                check_threshold=_CHECK_THRESHOLD,
                deprioritize_threshold=_DEPRIO_THRESHOLD,
                main_hits=main_hits_by_int,
                arrival_ts=arr_ts,
                age_alpha=_PEEK_CLPM_AGE_ALPHA,
                big_lane_share=_PEEK_CLPM_BIGLANE_SHARE,
                group_major=_PEEK_CLPM_GROUP_MAJOR,
                dynamic_lane=_PEEK_CLPM_DYNAMIC_LANE,
                slo_budget_s=_PEEK_CLPM_SLO_BUDGET_S,
            )
        else:
            peek_sort_inplace(
                adapters,
                rid_to_int,
                peek_tree,
                check_threshold=_CHECK_THRESHOLD,
                deprioritize_threshold=_DEPRIO_THRESHOLD,
                rank_by_cluster_size=False,
                main_hits=main_hits_by_int,
                peek_lpm_sort=_PEEK_LPM,
            )

        # Write the reordered sequence back into the deque in place.
        wq.clear()
        for a in adapters:
            wq.append(a._req)
        if _PROFILE:
            _prof["calc_priority_calls"] += 1
            _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp

    # --- Scheduler.schedule hook ------------------------------------------

    _orig_schedule = Scheduler.schedule

    def _patched_schedule(self):
        wq_list = list(self.waiting)
        _sync(wq_list)
        if _VALIDATE:
            _validate_sync(wq_list)
        _reorder_waiting(self)
        return _orig_schedule(self)

    Scheduler.schedule = _patched_schedule

    # --- Scheduler.add_request fast path ----------------------------------
    # Also catch arrivals between schedule() calls so phase-arrival timings
    # are accurate even under low scheduling cadence.

    _orig_add_request = Scheduler.add_request

    def _patched_add_request(self, request):
        result = _orig_add_request(self, request)
        # Run the full sync against the new waiting set; it's cheap when
        # only one rid changed thanks to the _last_rids fast path.
        try:
            _sync(self.waiting)
        except Exception as e:
            _log.warning("peek: sync on add_request failed: %s", e)
        return result

    Scheduler.add_request = _patched_add_request

    # --- Scheduler.finish_requests drop path ------------------------------
    # finish_requests is called for completed/aborted rids. Drop them from
    # peek eagerly so the tree doesn't have to wait for the next schedule()
    # tick to notice they're gone.

    _orig_finish_requests = getattr(Scheduler, "finish_requests", None)
    if _orig_finish_requests is not None:
        def _patched_finish_requests(self, request_ids, *args, **kwargs):
            # Forward the call verbatim -- vllm's signature includes a
            # required `finished_status`, but other versions / patches may
            # add params; *args/**kwargs is the safe pass-through.
            result = _orig_finish_requests(
                self, request_ids, *args, **kwargs,
            )
            try:
                if isinstance(request_ids, str):
                    rids_iter = (request_ids,)
                else:
                    rids_iter = tuple(request_ids)
                for rid_str in rids_iter:
                    rid_int = rid_interner.get(rid_str)
                    if rid_int is not None and rid_int in _seen_this_session:
                        peek_tree.discard(rid_int)
                        if _EVICTION:
                            _drop_rid(rid_int)
                        _seen_this_session.discard(rid_int)
            except Exception as e:
                _log.warning("peek: finish_requests drop failed: %s", e)
            return result

        Scheduler.finish_requests = _patched_finish_requests

    # --- BlockPool.get_new_blocks: peek-aware victim selection -----------
    if _EVICTION:
        _orig_get_new_blocks = BlockPool.get_new_blocks

        def _peek_get_new_blocks(self, num_blocks):
            te = _time.perf_counter_ns() if _PROFILE else 0
            if num_blocks <= 0 or not self.enable_caching:
                if _PROFILE:
                    _prof["evict_calls"] += 1
                    _prof["evict_fallback"] += 1
                    _prof["evict_ns"] += _time.perf_counter_ns() - te
                return _orig_get_new_blocks(self, num_blocks)
            # Cheap availability check up front -- match stock behavior.
            if num_blocks > self.get_num_free_blocks():
                if _PROFILE:
                    _prof["evict_calls"] += 1
                    _prof["evict_fallback"] += 1
                    _prof["evict_ns"] += _time.perf_counter_ns() - te
                return _orig_get_new_blocks(self, num_blocks)
            # If we have no demand signal at all, defer to stock LRU. Pure
            # stock behavior with one extra branch -- preserves perf parity
            # when peek isn't holding any pending state yet.
            if not _block_hash_demand:
                if _PROFILE:
                    _prof["evict_calls"] += 1
                    _prof["evict_fallback"] += 1
                    _prof["evict_ns"] += _time.perf_counter_ns() - te
                return _orig_get_new_blocks(self, num_blocks)

            fq = self.free_block_queue
            cap = (
                _EVICTION_SCAN_CAP if _EVICTION_SCAN_CAP > 0
                else max(num_blocks * 4, _EVICTION_MIN_SCAN_CAP)
            )
            picked: list = []          # blocks to evict (zero-priority first)
            low_demand: list = []      # (priority, position, block) deferred
            cur = fq.fake_free_list_head.next_free_block
            tail = fq.fake_free_list_tail
            scanned = 0
            while (cur is not None and cur is not tail
                   and len(picked) < num_blocks and scanned < cap):
                bh = cur._block_hash
                if bh is None:
                    picked.append(cur)
                else:
                    raw = get_block_hash(bh)
                    d = _block_hash_demand.get(raw, 0)
                    if d == 0:
                        picked.append(cur)
                    else:
                        # Per-mode priority. Higher = more protected.
                        # The picker takes blocks of lowest priority first
                        # (after the demand=0 batch), so a bigger number
                        # here means harder to evict.
                        if _EVICTION_MODE == "cluster":
                            depth = _block_hash_max_depth.get(raw, 1)
                            priority = d * (1 + depth)
                        elif _EVICTION_MODE == "recency":
                            la = _block_hash_last_access.get(raw)
                            age = (_time.time() - la) if la else 0.0
                            # int() to keep priority an integer for ordering;
                            # negative values (very old, low demand) sort
                            # ahead of positive ones -- exactly what we want.
                            priority = int(d - _EVICTION_RECENCY_W * age)
                        elif _EVICTION_MODE == "decay":
                            la = _block_hash_last_access.get(raw)
                            age = (_time.time() - la) if la else 0.0
                            import math as _math
                            decay = (
                                _math.exp(-age / _EVICTION_DECAY_TAU)
                                if _EVICTION_DECAY_TAU > 0 else 1.0
                            )
                            # Multiply by 1000 so float->int sort is stable
                            # at small priority differences.
                            priority = int(d * decay * 1000)
                        else:  # plain
                            priority = d
                        low_demand.append((priority, scanned, cur))
                cur = cur.next_free_block
                scanned += 1

            if len(picked) < num_blocks:
                # Top up from lowest-priority blocks we deferred.
                low_demand.sort(key=lambda x: (x[0], x[1]))
                need = num_blocks - len(picked)
                picked.extend(b for _, _, b in low_demand[:need])

            if len(picked) < num_blocks:
                # Scan cap was too tight. Bail to stock behavior -- peek's
                # hint isn't binding when we can't fully cover the request.
                if _PROFILE:
                    _prof["evict_calls"] += 1
                    _prof["evict_fallback"] += 1
                    _prof["evict_scanned"] += scanned
                    _prof["evict_ns"] += _time.perf_counter_ns() - te
                return _orig_get_new_blocks(self, num_blocks)

            # Detach picked blocks from the free queue (each remove()
            # decrements num_free_blocks) and apply the same eviction +
            # ref bookkeeping stock get_new_blocks does.
            for blk in picked:
                fq.remove(blk)
            zero_demand = 0
            for blk in picked:
                if blk._block_hash is None:
                    zero_demand += 1
                self._maybe_evict_cached_block(blk)
                assert blk.ref_cnt == 0
                blk.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(blk)
            if _PROFILE:
                _prof["evict_calls"] += 1
                _prof["evict_demand_aware"] += 1
                _prof["evict_scanned"] += scanned
                _prof["evict_zero_demand_picked"] += zero_demand
                _prof["evict_ns"] += _time.perf_counter_ns() - te
            return picked

        # Expose internal dicts on the patched function for test/debug.
        # Production code should never read these -- they're install-scope
        # state, exposed only because there's no other way to introspect
        # peek's eviction-side state from outside the closure.
        _peek_get_new_blocks._block_hash_demand = _block_hash_demand
        _peek_get_new_blocks._block_hash_max_depth = _block_hash_max_depth
        _peek_get_new_blocks._block_hash_last_access = _block_hash_last_access
        BlockPool.get_new_blocks = _peek_get_new_blocks
        _log.warning(
            "peek: BlockPool.get_new_blocks patched (eviction mode=%s)",
            _EVICTION_MODE,
        )

        # Stamp last-access timestamps for recency / decay modes by patching
        # the two BlockPool entry points where a block transitions to "fresh":
        #   - touch(blocks): a previously-free cached block was hit (ref_cnt
        #     0->1). Real cache-hit event.
        #   - cache_full_blocks(...): a newly-filled block enters the cache
        #     for the first time. Counts as a fresh access.
        if _EVICTION_NEEDS_LAST_ACCESS:
            _orig_touch = BlockPool.touch
            _orig_cache_full = BlockPool.cache_full_blocks

            def _peek_touch(self, blocks):
                now = _time.time()
                for blk in blocks:
                    bh = blk._block_hash
                    if bh is not None:
                        _block_hash_last_access[get_block_hash(bh)] = now
                return _orig_touch(self, blocks)

            def _peek_cache_full(self, request, blocks, num_cached_blocks,
                                 num_full_blocks, block_size, kv_cache_group_id):
                # Run the original first so block.block_hash is populated on
                # the new full blocks; then stamp the access time.
                result = _orig_cache_full(
                    self, request, blocks, num_cached_blocks,
                    num_full_blocks, block_size, kv_cache_group_id,
                )
                now = _time.time()
                for blk in blocks[num_cached_blocks:num_full_blocks]:
                    bh = blk._block_hash
                    if bh is not None:
                        _block_hash_last_access[get_block_hash(bh)] = now
                return result

            BlockPool.touch = _peek_touch
            BlockPool.cache_full_blocks = _peek_cache_full
            _log.warning(
                "peek: BlockPool.touch + cache_full_blocks patched "
                "(last-access tracking for mode=%s, recency_W=%s, decay_tau=%s)",
                _EVICTION_MODE, _EVICTION_RECENCY_W, _EVICTION_DECAY_TAU,
            )

    _log.warning(
        "peek: vllm hooks installed (sched=%s, clpm=%s, peek_lpm=%s, "
        "evict=%s, check=%d, deprio=%d, fallback_threshold=%d)",
        _SCHEDULER, _PEEK_CLPM, _PEEK_LPM, _EVICTION,
        _CHECK_THRESHOLD, _DEPRIO_THRESHOLD, _LPM_FB_THRESHOLD,
    )

    # Periodic dumps (phase, validate, profile) -- same pattern as sglang hook.
    if _PHASE_TRACKING:
        import threading as _pthreading
        import json as _pjson

        _phase_path = _PHASE_DUMP_PATH_TEMPLATE.format(pid=os.getpid())

        def _phase_dump_loop():
            while True:
                if _phase_timings:
                    try:
                        with open(_phase_path, "w") as f:
                            _pjson.dump(_phase_timings, f)
                    except Exception:
                        pass
                _time.sleep(1.0)

        _pt = _pthreading.Thread(target=_phase_dump_loop, daemon=True)
        _pt.start()
        _log.warning("peek: phase-tracking active; dump -> %s", _phase_path)

    if _VALIDATE:
        import threading as _vthreading
        import json as _vjson

        _vpath = _VALIDATE_PATH.format(pid=os.getpid())

        def _vdump_loop():
            while True:
                try:
                    with open(_vpath, "w") as f:
                        _vjson.dump(_vstats, f)
                except Exception:
                    pass
                _time.sleep(2.0)

        _vt = _vthreading.Thread(target=_vdump_loop, daemon=True)
        _vt.start()
        _log.warning("peek: validation dump; path=%s", _vpath)

    if _PROFILE:
        import threading as _ppthreading
        import json as _ppjson

        _ppath = os.environ.get(
            "PEEK_ONLINE_PROFILE_PATH", "/tmp/peek_profile_{pid}.json",
        ).format(pid=os.getpid())

        def _prof_dump_loop():
            while True:
                try:
                    snap = dict(_prof)
                    with open(_ppath, "w") as f:
                        _ppjson.dump(snap, f)
                except Exception:
                    pass
                _time.sleep(2.0)

        _ppt = _ppthreading.Thread(target=_prof_dump_loop, daemon=True)
        _ppt.start()
        _log.warning("peek: profile dump; path=%s", _ppath)


class _ReqAdapter:
    """Presents a vllm Request as an sglang-style req to peek.lpm_integration.

    lpm_integration reads: rid, prefix_indices (length), origin_input_ids,
    output_ids. vllm's Request exposes request_id, prompt_token_ids,
    output_token_ids. We use __slots__ and hold a direct reference to the
    vllm req so the reorder step can write the original objects back into
    the deque without copying tokens."""

    __slots__ = ("_req", "rid", "prefix_indices", "origin_input_ids",
                 "output_ids")

    def __init__(self, req, main_hit: int) -> None:
        self._req = req
        self.rid = req.request_id
        # prefix_indices only needs len(); a list of that size is the
        # cheapest way to satisfy lpm_integration's fallback path. In the
        # common case main_hits is populated and this length isn't read,
        # so the allocation only bites on sparse/empty-hit queues.
        self.prefix_indices = [0] * main_hit if main_hit > 0 else []
        self.origin_input_ids = req.prompt_token_ids or []
        # vllm output_token_ids is a read-only view; materialize once.
        self.output_ids = list(req.output_token_ids) if req.output_token_ids else []


if _ENABLED:
    try:
        _install()
    except ModuleNotFoundError as e:
        # Expected when this hook is imported in an env that does not have
        # vllm installed (e.g. the sglang env, where peek_sitecustomize tries
        # both engines and lets the wrong one no-op). Log at debug only.
        if (e.name or "").split(".")[0] == "vllm":
            _log.debug("peek: vllm not importable (%s); skipping vllm hook", e)
        else:
            _log.exception("peek: vllm integration failed; running vanilla: %s", e)
    except Exception as e:
        _log.exception("peek: vllm integration failed; running vanilla: %s", e)
else:
    _log.debug("peek: no PEEK_* flag set; vllm runs vanilla")
