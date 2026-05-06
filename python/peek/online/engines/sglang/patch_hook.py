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

"""sglang integration: inject peek's demand-aware eviction strategy and keep
peek's pending tree in sync with the scheduler's waiting queue.

Gated by env var so the same binary can run vanilla (baseline) or peek-enabled
with no code changes:

    PEEK_ONLINE_ENABLED=1 python -m sglang.launch_server ...     # peek active
    python -m sglang.launch_server ...                    # vanilla baseline

Integration strategy (eviction-only — does not touch LPM scheduling):

1. Monkey-patch `RadixCache.__init__` to replace its eviction_strategy with
   PeekDemandStrategy referencing a singleton PendingTree.
2. Monkey-patch `Scheduler.process_input_requests` to sync the pending tree
   against `self.waiting_queue` on every scheduler iteration. This is a
   diff-based sync — it tolerates any sglang mutation path (append, pop,
   list-comprehension reassignment) without needing to know about each one.

The singleton PendingTree is fine because a single sglang scheduler process
owns one waiting_queue. For multi-replica deployments peek would need to be
scoped per scheduler, but that's out of scope for this benchmark.
"""

from __future__ import annotations

import atexit
import logging
import os
import time as _time

def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


# Legacy back-compat. PEEK_ONLINE_ENABLED=1 turns on eviction; PEEK_ONLINE_ENABLED=full
# additionally turns on peek's scheduling (now LPM-semantic, cluster-ranked).
_LEGACY = os.environ.get("PEEK_ONLINE_ENABLED", "").lower()
_LEGACY_ANY = _LEGACY in ("1", "true", "yes", "on", "full")
_LEGACY_FULL = _LEGACY == "full"

# Per-mechanism toggles:
#   PEEK_ONLINE_SCHEDULER=1        — peek's scheduler (LPM primary, in-batch pioneer/
#                             sibling split, cluster-size pioneer ordering,
#                             no >128 fallback)
#   PEEK_ONLINE_EVICTION=1         — peek's queue-aware RadixCache eviction
#   PEEK_ONLINE_PHASE_TRACKING=1   — dump per-rid {arrive_ts, pick_ts} to
#                             /tmp/peek_phases.json. No scheduler/eviction
#                             override; works for baseline LPM too so the
#                             client can decouple queue_wait vs prefill+decode.
# Reserved for future mechanisms: PEEK_ONLINE_HANDOFF, PEEK_ONLINE_SESSION_HORIZON.
_EVICTION = _flag("PEEK_ONLINE_EVICTION") or _LEGACY_ANY
_SCHEDULER = (
    _flag("PEEK_ONLINE_SCHEDULER") or _LEGACY_FULL
    or _flag("PEEK_ONLINE_LPM") or _flag("PEEK_ONLINE_CLPM")
)
# Note: _PEEK_LPM is defined below; this ordering forward-references it via
# os.environ so _SCHEDULER picks up PEEK_ONLINE_LPM=1 before _PEEK_LPM is read.
# Handoff reverted: testing on shared-system-prompts with decode=8 tokens at
# queue<128 regime showed handoff adds ≤0.25pp cache hit and at decode=8
# rate=15 actively harmed throughput. The natural main_hit resort already
# promotes siblings during pioneer's decode window. Code kept out of the
# scheduler path; Rust primitive `longest_match_along` remains for future
# experiments. Re-enable via git if/when we build a different mechanism.

# PEEK_ONLINE_KV_BUDGET=1 — admission control using peek's running-batch visibility.
# Computes the KV budget the running batch has COMMITTED to exhaust (Σ of
# per-req prefill_len + remaining_decode) and, when sorting the waiting
# queue, defers pending rids whose admission would push commitment past
# max_total_tokens. Prevents mid-decode retractions; most valuable in
# long-decode regimes where each admission pins KV for thousands of tokens.
# Deferred rids remain in the queue; they naturally re-qualify next tick
# when running reqs progress / complete and free commitment budget.
_KV_BUDGET = _flag("PEEK_ONLINE_KV_BUDGET")
# Safety margin: don't spend the last X% of the budget. Leaves room for
# running reqs whose max_new_tokens under-estimates actual decode length.
_KV_BUDGET_MARGIN = float(os.environ.get("PEEK_ONLINE_KV_BUDGET_MARGIN", "0.05"))
# PEEK_ONLINE_PREDICT_DECODE=1 — use peek's per-cluster EWMA of completed reqs'
# output length to estimate decode commitment, rather than the (loose)
# `max_new_tokens` ceiling. Defaults on when KV-budget is on, since the
# whole point of KV-budget is moot without a tighter estimate. Falls back
# to max_new_tokens when no prediction is available (cold-start).
_PREDICT_DECODE = _flag("PEEK_ONLINE_PREDICT_DECODE") or (
    _KV_BUDGET and os.environ.get("PEEK_ONLINE_PREDICT_DECODE", "1").lower()
    not in ("0", "false", "no", "off")
)
# Min completed samples per cluster before its EWMA is trusted for
# prediction. Lower = adapts faster, higher = more stable. 3 is a good
# tradeoff for our workloads (covers warm-up of ~30 reqs over 100 groups).
_PREDICT_MIN_SAMPLES = int(os.environ.get("PEEK_ONLINE_PREDICT_MIN_SAMPLES", "3"))
# Safety multiplier on predicted decode length: actual outputs vary even
# within a cluster, so reserve 1.5× the EWMA to avoid under-reservation.
_PREDICT_SAFETY = float(os.environ.get("PEEK_ONLINE_PREDICT_SAFETY", "1.5"))
_PHASE_TRACKING = (
    _flag("PEEK_ONLINE_PHASE_TRACKING") or _EVICTION or _SCHEDULER
)
# Phase dumps are per-process because sglang runs several Python processes
# (tokenizer manager, scheduler, detokenizer, etc.) that all import this
# module. Only the scheduler process populates _phase_timings from its sync
# hook; other processes would overwrite the single file with empty dicts
# otherwise. Client reads all matching PID-suffixed files and merges.
_PHASE_DUMP_PATH_TEMPLATE = os.environ.get(
    "PEEK_ONLINE_PHASE_DUMP_PATH", "/tmp/peek_phases_{pid}.json"
)

# Default on: rank pioneers by cluster size (peek's enhancement over sglang).
# Set PEEK_ONLINE_RANK_BY_SIZE=0 to reproduce sglang's LPM order exactly for A/B.
_RANK_BY_SIZE = os.environ.get("PEEK_ONLINE_RANK_BY_SIZE", "1").lower() not in (
    "0", "false", "no", "off",
)
# PEEK_ONLINE_VALIDATE_LPM_ORDER=1 — on every scheduler tick where peek_lpm runs,
# independently recompute the sort using sglang's own _sort_by_longest_prefix
# on the same (queue state, deprio set) and record any mismatches to
# /tmp/peek_lpm_order_diffs_{pid}.json. Ground-truth request-by-request sort
# parity check. Adds one extra O(N log N) sort per tick — leave off for perf
# runs, turn on for correctness validation.
_VALIDATE_LPM_ORDER = _flag("PEEK_ONLINE_VALIDATE_LPM_ORDER")
_LPM_ORDER_DIFF_PATH = os.environ.get(
    "PEEK_ONLINE_LPM_ORDER_DIFF_PATH", "/tmp/peek_lpm_order_diffs_{pid}.json"
)

# PEEK_ONLINE_LPM=1 — pure LPM-semantic scheduler backed by peek primitives.
#
# Produces byte-identical sort output to stock sglang LPM (single-scalar
# `-main_hit`, deprioritized reqs banished to the end) but sources
# everything from peek — no call into sglang's LPM functions:
#   - main_hit per rid comes from one dualwalk of the pending tree (instead
#     of N separate radix match_prefix calls like stock LPM)
#   - in-batch deprio uses peek's exact-threshold tokens[:K] claim set
#     (same semantic as sglang's aux-tree check, no aux tree built)
#   - the sort itself runs in Rust (`_core.lpm_sort_order`) — no per-compare
#     Python callback overhead
#   - cluster_info / cluster-size ranking is NOT consulted
#
# Implies PEEK_ONLINE_SCHEDULER=1. Use as an A/B baseline against stock LPM to
# isolate peek's scheduling-data-plane overhead from its ordering-policy
# changes. If peek_lpm beats stock LPM, it's because dualwalk is cheaper
# than N match_prefix calls; if peek beats peek_lpm, it's because of
# cluster-size ordering.
_PEEK_LPM = _flag("PEEK_ONLINE_LPM")
# PEEK_ONLINE_CLPM=1 — cluster-LPM. Keeps peek_lpm's LPM-exact main_hit
# primary ordering, then applies a 4-key tiebreak ladder within each
# arrival-time window bucket:
#   (arrival_bucket, section_id, -main_hit, -req_score, -cluster_size, arrival_ns)
# where section_id ∈ {0=warm, 1=pioneer, 2=sibling} and
#   req_score = Σ (pending_count × edge_length) along rid's ancestor chain
#              in peek's pending tree.
# Implies PEEK_ONLINE_SCHEDULER=1. `PEEK_ONLINE_CLPM_WINDOW_MS` tunes the window (default
# 500 ms). If `PEEK_ONLINE_CLPM_WINDOW_MS=0`, bucketing is disabled and the policy
# degrades to peek_lpm with the extra tiebreaks (still ≥ peek_lpm).
_PEEK_CLPM = _flag("PEEK_ONLINE_CLPM")
_PEEK_CLPM_WINDOW_MS = int(os.environ.get("PEEK_ONLINE_CLPM_WINDOW_MS", "500"))
# Age-weighted main_hit alpha: virtual main_hit tokens added per second of
# wait. Default 400 → 5s-waited cold req ties a warm req at main_hit=2000.
# Set 0 to disable aging (falls back to raw main_hit + req_score tiebreaks).
_PEEK_CLPM_AGE_ALPHA = float(os.environ.get("PEEK_ONLINE_CLPM_AGE_ALPHA", "0"))
# Lane A (dense-cluster-preferring) share of admissions. 1.0 → pure W=0
# ordering (cache-locality only); 0.0 → pure FCFS fairness; 0.7 default
# gives big lanes 70% of admissions, small-cluster fairness lane 30%.
_PEEK_CLPM_BIGLANE_SHARE = float(os.environ.get("PEEK_ONLINE_CLPM_BIGLANE_SHARE", "0.7"))
# PEEK_ONLINE_CLPM_GROUP_MAJOR=1 → Lane A sorts by group (cluster_node) rather than
# per-req; within each section, all members of the top-scoring group admit
# back-to-back. Group score = depth × size. Singletons are groups of score 0
# (admitted last within their section, arrival-ordered).
_PEEK_CLPM_GROUP_MAJOR = _flag("PEEK_ONLINE_CLPM_GROUP_MAJOR")
# PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 → recompute Lane B share per tick from queue
# composition (singleton fraction) + oldest-singleton age, EMA-smoothed.
# Overrides PEEK_ONLINE_CLPM_BIGLANE_SHARE.
_PEEK_CLPM_DYNAMIC_LANE = _flag("PEEK_ONLINE_CLPM_DYNAMIC_LANE")
# SLO budget (seconds) for the dynamic-lane age-pressure denominator.
_PEEK_CLPM_SLO_BUDGET_S = float(os.environ.get("PEEK_ONLINE_CLPM_SLO_BUDGET_S", "2.0"))
# PEEK_ONLINE_LAZY_MATCH_PREFIX=1 turns on the lazy variant: calc_priority uses
# dualwalk for the sort key, and match_prefix (populating prefix_indices)
# runs only for the top K reqs in the sorted queue. Reqs past K this tick
# stay unpopulated; they'll be populated next tick when they rise to the top.
# Savings matter only when queue size > K (the current scenario has
# queue ~30-45 which is always < K, so expect no benefit there — lazy's
# big win is at queue > 128, where LPM falls back to FCFS but peek keeps
# sorting).
_LAZY_MATCH_PREFIX = _flag("PEEK_ONLINE_LAZY_MATCH_PREFIX")
_LAZY_K = int(os.environ.get("PEEK_ONLINE_LAZY_K", "128"))

# Any peek machinery active → need sync and install (phase tracking is
# cheap enough that we install the sync hook whenever it's requested, even
# if no scheduler/eviction override is on — baseline LPM runs get phase
# data this way).
_ENABLED = _EVICTION or _SCHEDULER or _PHASE_TRACKING
# Any mechanism that patches the scheduler → need the scheduling hook.
_SCHEDULE = _SCHEDULER

_PROFILE = _flag("PEEK_ONLINE_PROFILE")
_VALIDATE = _flag("PEEK_ONLINE_VALIDATE")

# PEEK_ONLINE_DECODE_AWARE was consumed by the legacy scoring path's decode-budget
# admission logic. Scoring is gone; the flag is a no-op now. Warn rather than
# silently accept so old sweep scripts surface the deprecation.
if _flag("PEEK_ONLINE_DECODE_AWARE"):
    import warnings
    warnings.warn(
        "peek: PEEK_ONLINE_DECODE_AWARE=1 is ignored — decode-budget admission was "
        "part of the removed scoring path. Remove from launch env.",
        stacklevel=2,
    )
_VALIDATE_PATH = os.environ.get(
    "PEEK_ONLINE_VALIDATE_PATH", "/tmp/peek_validate_{pid}.json"
)

# Validation counters — populated only when PEEK_ONLINE_VALIDATE=1. Each key tracks a
# distinct correctness violation category. Zero across the board = peek's view
# of the waiting queue and cache matches sglang's actual state.
_vstats: dict = {
    "sync_checks": 0,
    "sync_extra_in_peek": 0,      # rid in peek but not in waiting_queue
    "sync_missing_in_peek": 0,    # rid in waiting_queue but not in peek
    "sync_token_mismatch": 0,     # rid's peek tokens != req.origin_input_ids
    "first_violation_logged": False,
}

_log = logging.getLogger("peek.sglang")

# Per-section cumulative timing (populated only when PEEK_ONLINE_PROFILE=1).
_prof: dict = {
    "sync_calls": 0, "sync_skipped": 0, "sync_ns": 0,
    "insert_calls": 0, "insert_ns": 0,
    "discard_calls": 0, "discard_ns": 0,
    "calc_priority_calls": 0, "calc_priority_fallbacks": 0, "calc_priority_ns": 0,
}


def _install() -> None:
    """Install monkey-patches. Called lazily only when PEEK_ONLINE_ENABLED is set so
    importing this module in a vanilla run is a no-op beyond the env check."""
    from peek import PendingTree, PeekDemandStrategy

    peek_tree = PendingTree()

    # Per-rid phase timings. Keyed by rid_str (same as server's req.rid, which
    # the client sees in the OpenAI response `id` field → direct join).
    # arrive_ts: wall-clock when the rid first enters the waiting queue.
    # pick_ts:   wall-clock when the scheduler pulls it out of waiting
    #            (waiting_queue diff transitions rid from present → absent).
    # Using time.time() so the client can match against its own wall-clock.
    _phase_timings: dict = {}
    # string rid → u64 interned rid (peek speaks u64, sglang uses strings)
    rid_interner: dict[str, int] = {}
    _counter = [0]

    def _intern(rid_str: str) -> int:
        rid_int = rid_interner.get(rid_str)
        if rid_int is None:
            _counter[0] += 1
            rid_int = _counter[0]
            rid_interner[rid_str] = rid_int
        return rid_int

    # --- 1. RadixCache eviction strategy injection ---------------------------
    # Only install when PEEK_ONLINE_EVICTION is on. Scheduling-only modes leave the
    # stock LRU strategy in place so the eviction axis can be A/B-tested
    # independently of the scheduling axis.

    if _EVICTION:
        from sglang.srt.mem_cache.radix_cache import RadixCache

        _orig_radix_init = RadixCache.__init__
        _patched_caches: list = []

        def _patched_radix_init(self, params):
            _orig_radix_init(self, params)
            self.eviction_strategy = PeekDemandStrategy(peek_tree)
            _patched_caches.append(self)
            _log.warning("peek: injected PeekDemandStrategy into RadixCache")

        RadixCache.__init__ = _patched_radix_init

        # Per-victim eviction trace: log first N evictions' chosen victim with
        # its (demand×depth, path_len, access_time). Lets us verify that heap
        # pops actually follow the policy (low priority first). Gated by
        # PEEK_ONLINE_EVICTION_DEBUG=1 so default runs don't pay the overhead.
        if _flag("PEEK_ONLINE_EVICTION_DEBUG"):
            import heapq
            from peek.online.eviction import _max_ancestor_demand
            _orig_evict = RadixCache.evict
            _trace_path = "/tmp/peek_eviction_trace_{pid}.jsonl".format(pid=os.getpid())
            _trace_n = [0]
            _trace_max = 2000  # log up to 2000 victim events per process
            _trace_fh = [None]

            def _open_trace():
                if _trace_fh[0] is None:
                    try:
                        _trace_fh[0] = open(_trace_path, "w")
                    except Exception:
                        _trace_fh[0] = False
                return _trace_fh[0]

            def _patched_evict(self, params):
                # We can't cheaply intercept each heappop without rewriting
                # evict(). Instead, sample the state at function entry: log
                # priority of the minimum and the 90th-percentile of the
                # heap, to confirm the policy imposes a real ordering. Also
                # log a post-hoc summary of what was popped by comparing
                # leaf membership before/after.
                fh = _open_trace()
                leaves_before = list(self.evictable_leaves) if fh else []
                result = _orig_evict(self, params)
                if fh and fh is not False and _trace_n[0] < _trace_max:
                    try:
                        leaves_after = set(self.evictable_leaves)
                        evicted_this_call = [n for n in leaves_before if n not in leaves_after]
                        # Sample at most 8 evicted victims to keep trace readable.
                        for v in evicted_this_call[:8]:
                            if _trace_n[0] >= _trace_max:
                                break
                            try:
                                val, plen, lvls = _max_ancestor_demand(v, peek_tree)
                                rec = {
                                    "ev_idx": _trace_n[0],
                                    "priority_value": int(val),
                                    "path_len": int(plen),
                                    "levels": int(lvls),
                                    "last_access_time": float(
                                        getattr(v, "last_access_time", 0.0)
                                    ),
                                    "leaf_key_len": int(len(getattr(v.key, "token_ids", [])) if getattr(v, "key", None) else 0),
                                    "num_evicted_this_call": len(evicted_this_call),
                                    "heap_size_at_entry": len(leaves_before),
                                }
                                import json as _jj
                                fh.write(_jj.dumps(rec) + "\n")
                                fh.flush()
                                _trace_n[0] += 1
                            except Exception:
                                pass
                    except Exception:
                        pass
                return result

            RadixCache.evict = _patched_evict
            _log.warning(
                "peek: RadixCache.evict patched for per-victim trace → %s (up to %d events)",
                _trace_path, _trace_max,
            )

    # --- 2. Scheduler waiting-queue sync -------------------------------------

    from sglang.srt.managers.scheduler import Scheduler

    _orig_process_input = Scheduler.process_input_requests

    # Cache of last-seen {rid_str} to short-circuit when the queue hasn't changed.
    _last_rids: set = set()

    def _sync(scheduler) -> None:
        """Diff peek tree against scheduler.waiting_queue and bring it in sync.

        Fast path: if the waiting_queue's rid set hasn't changed since last
        sync, skip all work. Common case in a busy event loop where peek's
        scheduling hook runs on every iteration but the queue only changes
        when arrivals/picks happen."""
        t0 = _time.perf_counter_ns() if _PROFILE else 0
        wq = scheduler.waiting_queue
        # Fast path: identical set of rids → nothing changed.
        if len(wq) == len(_last_rids):
            current_strs = {req.rid for req in wq}
            if current_strs == _last_rids:
                if _PROFILE:
                    _prof["sync_calls"] += 1
                    _prof["sync_skipped"] += 1
                    _prof["sync_ns"] += _time.perf_counter_ns() - t0
                return
        else:
            current_strs = {req.rid for req in wq}

        # Slow path: rebuild sets and diff.
        now_wall = _time.time() if _PHASE_TRACKING else 0.0
        current_ints: set[int] = set()
        for req in wq:
            rid_int = _intern(req.rid)
            current_ints.add(rid_int)
            if rid_int not in _seen_this_session:
                tokens = list(req.origin_input_ids) + list(req.output_ids or [])
                if _PROFILE:
                    t1 = _time.perf_counter_ns()
                try:
                    peek_tree.insert(rid_int, tokens)
                except ValueError:
                    peek_tree.discard(rid_int)
                    peek_tree.insert(rid_int, tokens)
                if _PROFILE:
                    _prof["insert_calls"] += 1
                    _prof["insert_ns"] += _time.perf_counter_ns() - t1
                _seen_this_session.add(rid_int)
                # Phase: new rid → record arrive_ts (first time observed in
                # the waiting queue via this scheduler's sync).
                if _PHASE_TRACKING:
                    _phase_timings[req.rid] = {"arrive_ts": now_wall}
        for rid_int in list(_seen_this_session):
            if rid_int not in current_ints:
                if _PROFILE:
                    t1 = _time.perf_counter_ns()
                peek_tree.discard(rid_int)
                if _PROFILE:
                    _prof["discard_calls"] += 1
                    _prof["discard_ns"] += _time.perf_counter_ns() - t1
                _seen_this_session.discard(rid_int)
        # Phase: rids that were in last_rids but are no longer in wq were
        # picked by the scheduler (pulled from waiting → running batch) or
        # aborted. Record pick_ts. Works off the string rid set directly.
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

    # "seen this session" = rids currently tracked by peek. Separate from
    # rid_interner so we can distinguish "first time seen" from "previously
    # saw, removed, now back" (retraction case).
    _seen_this_session: set[int] = set()

    def _validate_sync(scheduler) -> None:
        """After _sync, verify the invariant: peek.tree tracks EXACTLY the rids
        in scheduler.waiting_queue, with matching tokens. Counts mismatches by
        category and logs the first occurrence. Cheap — one pass over queue,
        plus one peek.tokens() lookup per rid."""
        wq = scheduler.waiting_queue
        wq_ints = set()
        # Build wq_ints and check every waiting rid is in peek with correct tokens.
        token_mismatches = 0
        missing = 0
        for req in wq:
            rid_int = rid_interner.get(req.rid)
            if rid_int is None:
                missing += 1
                continue
            wq_ints.add(rid_int)
            # peek's stored tokens = origin_input_ids + output_ids at insert time.
            expected = list(req.origin_input_ids) + list(req.output_ids or [])
            stored = peek_tree.tokens(rid_int)
            if stored is None:
                missing += 1
                continue
            if stored != expected:
                token_mismatches += 1
        # Every peek-tracked rid must also be in waiting_queue.
        extra = 0
        for rid_int in _seen_this_session:
            if rid_int not in wq_ints:
                extra += 1
        _vstats["sync_checks"] += 1
        if missing or extra or token_mismatches:
            _vstats["sync_missing_in_peek"] += missing
            _vstats["sync_extra_in_peek"] += extra
            _vstats["sync_token_mismatch"] += token_mismatches
            if not _vstats["first_violation_logged"]:
                _vstats["first_violation_logged"] = True
                _log.warning(
                    "peek VALIDATE: first sync violation — missing=%d extra=%d token_mm=%d "
                    "wq_size=%d peek_tracked=%d",
                    missing, extra, token_mismatches, len(wq), len(_seen_this_session),
                )

    def _patched_process_input(self, recv_reqs):
        result = _orig_process_input(self, recv_reqs)
        _sync(self)
        if _VALIDATE:
            _validate_sync(self)
        return result

    Scheduler.process_input_requests = _patched_process_input

    # --- 3. Also sync right before get_next_batch_to_run ---------------------
    # Arrivals happen in process_input_requests; departures happen inside
    # get_next_batch_to_run and other post-batch cleanup paths. Hook before
    # the batch forms so peek tree reflects pre-pick state (useful if eviction
    # fires during the pick).

    _orig_get_next_batch = Scheduler.get_next_batch_to_run

    def _patched_get_next_batch(self):
        _sync(self)
        if _VALIDATE:
            _validate_sync(self)
        return _orig_get_next_batch(self)

    Scheduler.get_next_batch_to_run = _patched_get_next_batch

    _log.warning(
        "peek: installed hooks on RadixCache.__init__, "
        "Scheduler.process_input_requests, Scheduler.get_next_batch_to_run"
    )

    # --- 4. Scheduler policy ------------------------------------------------
    # PEEK_ONLINE_SCHEDULER=1 installs peek's scheduler: LPM-semantic primary sort by
    # longest prefix against sglang's radix cache, in-batch pioneer/sibling
    # split via peek's pending tree (replaces sglang's throwaway aux tree),
    # cluster-size pioneer ordering, no >128 FCFS fallback.

    if _SCHEDULER:
        from sglang.srt.managers.schedule_policy import (
            SchedulePolicy,
            CacheAwarePolicy,
            IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
            IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
        )
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from peek.online.lpm_integration import peek_sort_inplace

        _orig_calc_priority = SchedulePolicy.calc_priority
        _peek_call_count = [0]
        # Decode-prediction observation state: snapshot of last tick's
        # running-batch reqs as {rid_str: (tokens_tuple, last_output_len)}.
        # When a rid disappears between ticks it has finished (or been
        # retracted; rare enough not to matter for EWMA). Record into the
        # pending tree so predict_decode can use it for future arrivals.
        _running_seen: dict = {}
        _decode_obs_count = [0]

        def _patched_calc_priority(self, waiting_queue, running_batch=None):
            tcp = _time.perf_counter_ns() if _PROFILE else 0
            # --- Decode-length observation -----------------------------------
            # Update _running_seen with current running reqs; for any rid that
            # vanished, push (input_tokens, output_len) into peek's tree.
            if _PREDICT_DECODE and running_batch is not None:
                running_reqs = getattr(running_batch, "reqs", None) or []
                cur_rids = set()
                for rr in running_reqs:
                    cur_rids.add(rr.rid)
                    out_len = len(rr.output_ids or [])
                    if rr.rid in _running_seen:
                        _, _ = _running_seen[rr.rid]
                        _running_seen[rr.rid] = (
                            _running_seen[rr.rid][0],
                            out_len,
                        )
                    else:
                        _running_seen[rr.rid] = (
                            tuple(rr.origin_input_ids),
                            out_len,
                        )
                # Reqs that disappeared from the running batch → finished.
                for rid_str in list(_running_seen.keys()):
                    if rid_str not in cur_rids:
                        tokens_tuple, last_out = _running_seen.pop(rid_str)
                        if last_out > 0:
                            try:
                                peek_tree.record_decode(list(tokens_tuple), int(last_out))
                                _decode_obs_count[0] += 1
                            except Exception as e:
                                _log.warning("peek: record_decode failed: %s", e)
            # Only override when LPM is the configured policy. Other policies
            # (FCFS, DFS_WEIGHT, LOF, RANDOM) delegate to sglang's original.
            if self.policy != CacheAwarePolicy.LPM:
                if _PROFILE:
                    _prof["calc_priority_calls"] += 1
                    _prof["calc_priority_fallbacks"] += 1
                    _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
                return _orig_calc_priority(self, waiting_queue, running_batch)

            # Stock sglang LPM bypasses to FCFS when the queue is long — both
            # because scheduler overhead grows super-linearly AND because LPM's
            # signal quality degrades at long queues (most reqs tied at
            # main_hit=0, cache churn outpaces admission decisions). peek_lpm
            # mirrors this fallback to preserve apples-to-apples perf parity
            # with stock LPM. Env-tunable so we can later probe how far peek's
            # cheaper pending-tree maintenance lets us push the threshold.
            _lpm_fb_threshold = int(
                os.environ.get("PEEK_ONLINE_LPM_FALLBACK_THRESHOLD", "128")
            )
            if _PEEK_LPM and len(waiting_queue) > _lpm_fb_threshold:
                if _PROFILE:
                    _prof["calc_priority_calls"] += 1
                    _prof["calc_priority_fallbacks"] += 1
                    _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
                # FCFS: waiting_queue already arrives in arrival order; a
                # stable noop-sort is a safe no-op equivalent to FCFS.
                return

            # match_prefix population strategy:
            #   - Lazy mode (PEEK_ONLINE_LAZY_MATCH_PREFIX=1): skip here; we'll
            #     populate only the top-K reqs AFTER sorting. Saves work
            #     when queue > K; the sort key comes from the dualwalk below
            #     which doesn't need prefix_indices.
            #   - Default (eager): populate all reqs up front. Required if
            #     downstream considers reqs past K and expects populated
            #     prefix_indices.
            if not _LAZY_MATCH_PREFIX:
                for r in waiting_queue:
                    prefix_ids = list(r.origin_input_ids) + list(r.output_ids or [])
                    mr = self.tree_cache.match_prefix(
                        MatchPrefixParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=r.extra_key),
                        )
                    )
                    (
                        r.prefix_indices,
                        r.last_node,
                        r.last_host_node,
                        r.host_hit_length,
                    ) = (
                        mr.device_indices,
                        mr.last_device_node,
                        mr.last_host_node,
                        mr.host_hit_length,
                    )

            # Dualwalk over peek's pending tree alongside sglang's radix
            # cache: one pass that computes main_hit per rid. Replaces LPM's
            # N per-req match_prefix walks for the sort-key signal. We still
            # rely on the match_prefix calls above to populate prefix_indices
            # (downstream admission needs device indices, not just lengths),
            # but the dualwalk result is the primary signal for sorting.
            #
            # PEEK_ONLINE_LPM_SKIP_DUALWALK=1 — ablation flag: skip the dualwalk and
            # fall back to len(r.prefix_indices) as the sort key (same signal
            # stock LPM uses). Diagnoses whether dualwalk's pure-Python
            # _cache_match callback is the overhead culprit.
            cache_root = getattr(self.tree_cache, "root_node", None)
            rid_to_int = {r.rid: _intern(r.rid) for r in waiting_queue}
            dualwalk_hits = None
            _skip_dualwalk = _flag("PEEK_ONLINE_LPM_SKIP_DUALWALK")
            if cache_root is not None and not _skip_dualwalk:
                def _cache_match(tokens):
                    cur = cache_root
                    consumed = 0
                    while consumed < len(tokens):
                        first = tokens[consumed]
                        children = getattr(cur, "children", {})
                        child = children.get(first)
                        if child is None:
                            return consumed
                        edge = list(child.key.token_ids)
                        rem = tokens[consumed:]
                        common = 0
                        for a, b in zip(edge, rem):
                            if a != b:
                                break
                            common += 1
                        consumed += common
                        if common < len(edge):
                            return consumed
                        cur = child
                    return consumed
                try:
                    # min_pending_count=1 walks every edge exactly (LPM-faithful
                    # — every req's prefix length is computed as if queried
                    # individually, no approximation).
                    dualwalk_hits = peek_tree.compute_main_hits(_cache_match, 1)
                except Exception as _e:
                    _log.warning("peek: dualwalk failed (%s); falling back to "
                                 "prefix_indices-based sort key", _e)

            # Peek-native LPM sort: LPM-faithful order (-main_hit, is_depr,
            # rid) driven by dualwalk; in-batch deprio via peek cluster_info.
            # With PEEK_ONLINE_LPM=1, use single-scalar key that's byte-identical
            # to sglang's _sort_by_longest_prefix — the whole sort runs in
            # Rust via _core.lpm_sort_order.
            if _VALIDATE_LPM_ORDER and _PEEK_LPM:
                _validation_queue_snapshot = list(waiting_queue)
            if _PEEK_CLPM:
                # Cluster-LPM: arrival-bucket + section + tiebreak ladder.
                from peek.online.lpm_integration import peek_clpm_sort_inplace
                arr_ts_map = {
                    rid_str: slot.get("arrive_ts", 0.0)
                    for rid_str, slot in _phase_timings.items()
                    if "arrive_ts" in slot
                } if _PHASE_TRACKING else None
                _peek_depr = peek_clpm_sort_inplace(
                    waiting_queue,
                    rid_to_int,
                    peek_tree,
                    window_ms=_PEEK_CLPM_WINDOW_MS,
                    check_threshold=IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
                    deprioritize_threshold=IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
                    main_hits=dualwalk_hits,
                    arrival_ts=arr_ts_map,
                    age_alpha=_PEEK_CLPM_AGE_ALPHA,
                    big_lane_share=_PEEK_CLPM_BIGLANE_SHARE,
                    group_major=_PEEK_CLPM_GROUP_MAJOR,
                    dynamic_lane=_PEEK_CLPM_DYNAMIC_LANE,
                    slo_budget_s=_PEEK_CLPM_SLO_BUDGET_S,
                )
            else:
                _peek_depr = peek_sort_inplace(
                    waiting_queue,
                    rid_to_int,
                    peek_tree,
                    check_threshold=IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
                    deprioritize_threshold=IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
                    rank_by_cluster_size=False,  # plain LPM order; no cluster grouping
                    main_hits=dualwalk_hits,
                    peek_lpm_sort=_PEEK_LPM,
                )
            if _VALIDATE_LPM_ORDER and _PEEK_LPM:
                # Re-sort a COPY of the pre-sort queue using sglang's own
                # _sort_by_longest_prefix with peek's deprio set. If the
                # peek_lpm pipeline is LPM-byte-identical, the resulting
                # rid orderings must match exactly.
                _ref_q = list(_validation_queue_snapshot)
                try:
                    from sglang.srt.managers.schedule_policy import (
                        SchedulePolicy as _SP,
                    )
                    _SP._sort_by_longest_prefix(_ref_q, _peek_depr)
                    _peek_order = [r.rid for r in waiting_queue]
                    _ref_order = [r.rid for r in _ref_q]
                    if _peek_order != _ref_order:
                        _first_diff = next(
                            (i for i, (a, b) in enumerate(zip(_peek_order, _ref_order))
                             if a != b),
                            None,
                        )
                        _mismatch = {
                            "tick_ts": _time.time(),
                            "queue_len": len(_peek_order),
                            "first_diff_at": _first_diff,
                            "peek_head": _peek_order[:20],
                            "ref_head": _ref_order[:20],
                            "peek_tail": _peek_order[-10:],
                            "ref_tail": _ref_order[-10:],
                        }
                        _log.warning(
                            "peek_lpm ORDER MISMATCH at tick: queue_len=%d "
                            "first_diff_at=%s", len(_peek_order), _first_diff,
                        )
                        import json as _json
                        _pth = _LPM_ORDER_DIFF_PATH.format(pid=os.getpid())
                        try:
                            with open(_pth, "a") as _f:
                                _f.write(_json.dumps(_mismatch) + "\n")
                        except Exception:
                            pass
                except Exception as _e:
                    _log.warning("peek_lpm validation failed: %s", _e)

            # Lazy match_prefix: after sorting, populate prefix_indices only
            # for the top-K reqs most likely to admit this tick. Reqs beyond
            # K stay with their default (empty/None) prefix_indices; if
            # sglang's admission path happens to consider them, they get a
            # cold prefill for this tick (they'll rise to top next tick and
            # be populated then). Savings are O(N-K) per call when queue > K.
            if _LAZY_MATCH_PREFIX:
                k = min(len(waiting_queue), _LAZY_K)
                for i in range(k):
                    r = waiting_queue[i]
                    # Skip if already populated by a prior call this tick.
                    pi = getattr(r, "prefix_indices", None)
                    if pi is not None and len(pi) > 0:
                        continue
                    prefix_ids = list(r.origin_input_ids) + list(r.output_ids or [])
                    mr = self.tree_cache.match_prefix(
                        MatchPrefixParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=r.extra_key),
                        )
                    )
                    (
                        r.prefix_indices,
                        r.last_node,
                        r.last_host_node,
                        r.host_hit_length,
                    ) = (
                        mr.device_indices,
                        mr.last_device_node,
                        mr.last_host_node,
                        mr.host_hit_length,
                    )

            # --- Mechanism A: KV-budget-aware admission control ---
            # Compute the budget the running batch has committed to exhaust
            # (each running req's remaining prefill + max_new_tokens decode).
            # Walk the sorted queue greedily: admit while cumulative commit
            # stays under budget; defer the rest. Deferred rids stay in
            # waiting_queue (we move them to the back) and re-qualify next
            # tick as running reqs progress.
            if _KV_BUDGET and running_batch is not None:
                # Helper: estimate remaining decode tokens for one req.
                # When PREDICT_DECODE is on, query peek's per-cluster EWMA
                # (× safety) and clamp by max_new_tokens. Otherwise fall
                # back to (max_new_tokens - already-decoded), which is
                # sglang's worst-case projection.
                def _remaining_decode(req, already_out: int) -> int:
                    try:
                        max_new = int(req.sampling_params.max_new_tokens)
                    except Exception:
                        max_new = 0
                    worst = max(0, max_new - already_out)
                    if not _PREDICT_DECODE:
                        return worst
                    tokens = list(req.origin_input_ids)
                    pred = peek_tree.predict_decode(tokens, _PREDICT_MIN_SAMPLES)
                    if pred is None:
                        return worst
                    ewma, _samples = pred
                    target_total = int(ewma * _PREDICT_SAFETY)
                    target_total = min(target_total, max_new) if max_new > 0 else target_total
                    return max(0, target_total - already_out)

                running_reqs = getattr(running_batch, "reqs", None) or []
                running_commit = 0
                for rr in running_reqs:
                    plen = len(rr.origin_input_ids) + len(rr.output_ids or [])
                    out_len = len(rr.output_ids or [])
                    running_commit += plen + _remaining_decode(rr, out_len)

                max_total = getattr(self, "max_total_num_tokens", None)
                if not max_total:
                    tc = getattr(self, "tree_cache", None)
                    max_total = getattr(tc, "max_total_num_tokens", 0) if tc else 0
                if max_total > 0:
                    effective_budget = int(max_total * (1.0 - _KV_BUDGET_MARGIN))
                    remaining_budget = effective_budget - running_commit
                    if remaining_budget < 0:
                        remaining_budget = 0

                    admitted: list = []
                    deferred: list = []
                    cum_commit = 0
                    for r in waiting_queue:
                        pi = r.prefix_indices
                        main_hit = 0 if pi is None else len(pi)
                        total_tokens = len(r.origin_input_ids) + len(r.output_ids or [])
                        prefill_cost = max(0, total_tokens - main_hit)
                        decode_cost = _remaining_decode(r, len(r.output_ids or []))
                        commit = prefill_cost + decode_cost
                        if cum_commit + commit <= remaining_budget or not admitted:
                            # Always allow at least one admission even if budget
                            # is tight — prevents deadlock when every req exceeds
                            # the budget alone. Subsequent reqs gated normally.
                            admitted.append(r)
                            cum_commit += commit
                        else:
                            deferred.append(r)
                    if deferred:
                        waiting_queue[:] = admitted + deferred

            _peek_call_count[0] += 1
            if _PROFILE:
                _prof["calc_priority_calls"] += 1
                _prof["calc_priority_ns"] += _time.perf_counter_ns() - tcp
            # Matches sglang's LPM contract: True signals prefix_indices
            # have been populated by the policy.
            return True

        SchedulePolicy.calc_priority = _patched_calc_priority
        _log.warning(
            "peek: installed scheduler (check=%d, deprioritize=%d, "
            "rank_by_size=%s, no >128 FCFS fallback)",
            IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD,
            IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
            _RANK_BY_SIZE,
        )

    # Periodic phase-timings dump — always-on when _PHASE_TRACKING is active.
    # Written to a fixed path (no pid suffix) so the client can pick it up
    # without needing to know the server pid. The dump is idempotent: the
    # client reads it once at end-of-run; concurrent-write races during
    # bench are harmless because the client only reads AFTER all requests
    # finish.
    if _PHASE_TRACKING:
        import threading as _pthreading
        import json as _pjson

        _phase_path = _PHASE_DUMP_PATH_TEMPLATE.format(pid=os.getpid())

        def _phase_dump_loop():
            while True:
                # Only write if we actually have data. sglang's tokenizer
                # manager / detokenizer / ipc subprocesses also import us
                # but never populate _phase_timings — skipping them avoids
                # stomping on the scheduler process's file.
                if _phase_timings:
                    try:
                        with open(_phase_path, "w") as f:
                            _pjson.dump(_phase_timings, f)
                    except Exception:
                        pass
                _time.sleep(1.0)

        _pt = _pthreading.Thread(target=_phase_dump_loop, daemon=True)
        _pt.start()
        _log.warning(
            "peek: phase-tracking active; dumping arrive/pick timings to %s",
            _phase_path,
        )

    # Periodic validation dump — same pattern as profile: the scheduler
    # subprocess may be killed hard, so write validation counters to a known
    # file every 2 seconds from a background thread.
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
        _log.warning("peek: validation dump thread started; path=%s", _vpath)

    # Periodic profile dump — sglang's scheduler subprocess may be killed
    # hard so atexit is unreliable. Instead, write stats to a known file
    # every 2 seconds from a background thread.
    if _PROFILE:
        import threading
        import json as _json

        _profile_path_template = os.environ.get(
            "PEEK_ONLINE_PROFILE_PATH", "/tmp/peek_profile_{pid}.json"
        )
        _profile_path = _profile_path_template.format(pid=os.getpid())

        def _dump_loop():
            from peek.online.eviction import _eviction_profile
            while True:
                try:
                    snapshot = dict(_prof)
                    snapshot["eviction"] = _eviction_profile()
                    with open(_profile_path, "w") as f:
                        _json.dump(snapshot, f)
                except Exception:
                    pass
                _time.sleep(2.0)

        t = threading.Thread(target=_dump_loop, daemon=True)
        t.start()
        _log.warning("peek: profile dump thread started; path=%s", _profile_path)


if _ENABLED:
    try:
        _install()
    except Exception as e:
        _log.exception("peek: integration failed; continuing as vanilla: %s", e)
else:
    _log.debug("peek: PEEK_ONLINE_ENABLED not set; running as vanilla sglang")
