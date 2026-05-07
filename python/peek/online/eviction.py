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

"""Demand-aware eviction strategy for sglang's RadixCache.

Plugs into `sglang.srt.mem_cache.evict_policy.EvictionStrategy`. For each
candidate eviction node, computes the number of pending requests whose
tokens start with that node's path and uses that as the primary sort key --
nodes with zero pending demand evict first, LRU breaks ties within a bucket.

Integration:
    from peek.online.eviction import PeekDemandStrategy
    from peek import PendingTree

    tree = PendingTree()
    tree_cache.eviction_strategy = PeekDemandStrategy(tree)

Peek's pending tree must be kept in sync with sglang's waiting queue
(insert on arrival, remove on pick). That's the same lifecycle hook already
scoped for the scheduling integration.
"""

from __future__ import annotations

import atexit
import json as _json
import math
import os
import time
from typing import Tuple

from peek import PendingTree

_PROFILE = os.environ.get("PEEK_ONLINE_PROFILE", "").lower() in ("1", "true", "yes", "on")
_DEBUG = os.environ.get("PEEK_ONLINE_EVICTION_DEBUG", "").lower() in ("1", "true", "yes", "on")

# Eviction policy variants (PEEK_ONLINE_EVICTION_MODE):
#   plain    -- (demandxdepth, last_access). Default.
#   cluster  -- (demandxdepth x (1 + levels), last_access). Extra multiplier
#              for deeper ancestor chains -- favors longer sharing structures.
#              Paper §3.2 primary mode; the cLPM+GM+DL+PE config uses this.
#   recency  -- (demandxdepth − W.age_sec, last_access). Additive recency
#              penalty; older nodes lose demand-protection quickly.
#              PEEK_ONLINE_EVICTION_RECENCY_W (default 100) tokens-at-risk/sec.
#   decay    -- (demandxdepth x exp(-age/tau), last_access). Multiplicative
#              exponential decay. PEEK_ONLINE_EVICTION_DECAY_TAU (default 30s).
#
# Legacy names `demand_cluster`, `demand_recency`, `demand_decay` are
# accepted as aliases for the un-prefixed forms.
_EVICTION_MODE_ALIASES = {
    "demand_cluster": "cluster",
    "demand_recency": "recency",
    "demand_decay": "decay",
}
_RAW_EVICTION_MODE = os.environ.get("PEEK_ONLINE_EVICTION_MODE", "plain").lower()
_EVICTION_MODE = _EVICTION_MODE_ALIASES.get(_RAW_EVICTION_MODE, _RAW_EVICTION_MODE)
if _EVICTION_MODE not in ("plain", "cluster", "recency", "decay"):
    import warnings as _warnings
    _warnings.warn(
        f"peek.eviction: PEEK_ONLINE_EVICTION_MODE={_RAW_EVICTION_MODE!r} not "
        f"recognized; falling back to 'plain'. Valid modes: "
        f"plain, cluster, recency, decay.",
        stacklevel=2,
    )
    _EVICTION_MODE = "plain"
_EVICTION_RECENCY_W = float(os.environ.get("PEEK_ONLINE_EVICTION_RECENCY_W", "100"))
_EVICTION_DECAY_TAU = float(os.environ.get("PEEK_ONLINE_EVICTION_DECAY_TAU", "30.0"))
_DEBUG_PATH = os.environ.get(
    "PEEK_ONLINE_EVICTION_DEBUG_PATH", "/tmp/peek_eviction_debug_{pid}.json"
)
_prof = {"gp_calls": 0, "gp_ns": 0, "path_calls": 0, "path_ns": 0, "demand_calls": 0, "demand_ns": 0}

# Eviction diagnostics -- count of get_priority calls bucketed by pending_demand
# value, plus counts of empty-path (root) and how often peek's signal differs
# from pure LRU (i.e., demand > 0 protecting a node LRU would pick).
_diag = {
    "pid": os.getpid(),
    "first_call_ts": None,
    "last_call_ts": None,
    "gp_calls": 0,
    "gp_root_or_empty_path": 0,     # path was empty -> demand forced to 0 regardless
    "gp_demand_zero": 0,             # path non-empty but demand=0 (= LRU fallback)
    "gp_demand_gt_zero": 0,          # path non-empty, demand>=1 (signal active)
    "gp_demand_by_bucket": {},       # bucket_label -> count
    "max_demand_seen": 0,
    "path_len_when_max_demand": 0,
    "mean_demand_when_nonzero": 0.0,
    "sum_demand_when_nonzero": 0.0,
}


def _bump_bucket(value: int) -> None:
    # Bucket the token-weighted value (demand x depth). Buckets align with
    # meaningful prefill-cost scales: <100 tokens saved is negligible,
    # >10000 means a big cluster x long prefix.
    if value == 0:
        bkt = "0"
    elif value < 100:
        bkt = "1-99"
    elif value < 1000:
        bkt = "100-999"
    elif value < 10000:
        bkt = "1k-10k"
    elif value < 100000:
        bkt = "10k-100k"
    else:
        bkt = "100k+"
    _diag["gp_demand_by_bucket"][bkt] = _diag["gp_demand_by_bucket"].get(bkt, 0) + 1
    if value > _diag["max_demand_seen"]:
        _diag["max_demand_seen"] = value


def _dump_diag() -> None:
    try:
        path = _DEBUG_PATH.format(pid=os.getpid())
        with open(path, "w") as f:
            _json.dump(_diag, f, indent=2)
    except Exception:
        pass


if _DEBUG:
    atexit.register(_dump_diag)

    # Periodic dump thread -- atexit misses subprocesses killed via SIGKILL
    # (which is what our driver does to sglang). Dump every 2 s to a
    # PID-suffixed file so we capture the scheduler subprocess too.
    import threading as _ethreading

    def _periodic_dump_loop() -> None:
        while True:
            if _diag["gp_calls"] > 0:
                _dump_diag()
            time.sleep(2.0)

    _et = _ethreading.Thread(target=_periodic_dump_loop, daemon=True)
    _et.start()


class PeekDemandStrategy:
    """sglang-compatible EvictionStrategy that prefers to evict low-demand
    cache nodes first. Does not inherit from EvictionStrategy so it can be
    used without importing sglang at module load -- duck-typing is enough."""

    def __init__(self, tree: PendingTree) -> None:
        self.tree = tree

    def get_priority(self, node) -> Tuple[int, float]:
        """Returns (subtree_demand, last_access_time) -- sglang's heap evicts
        the lowest tuple first.

        `subtree_demand` = # pending rids whose tokens PASS THROUGH this cache
        node's path (i.e., rids whose full token sequence has this path as a
        prefix). For shared-prefix workloads this is the right signal: the
        intermediate shared-prefix node has subtree_demand = N (all N cluster
        members depend on it), and LRU is blocked from evicting it while any
        sibling is still pending.

        The earlier `terminators_at`-based version under-protected intermediate
        shared-prefix nodes (they had 0 exact terminators even with 100 pending
        siblings) and produced cache thrashing on the G=100 workload."""
        # sglang's eviction only considers LEAVES. A leaf's full path is
        # unique to one completed request (shared_prefix + that request's
        # unique question + answer tail), so pending_demand(leaf_full_path)
        # is almost always 0 -> peek's signal is dead at leaves.
        #
        # Fix: take the max pending_demand over the leaf's ancestor paths.
        # An ancestor with demand N means N pending rids want that subtree
        # alive. Evicting THIS leaf brings the ancestor closer to being
        # evicted too (via cascade when it becomes childless). So leaves
        # whose ancestors have high demand should be protected.
        demand, path_len, n_levels = _max_ancestor_demand(node, self.tree)
        last_access = getattr(node, "last_access_time", 0.0)

        # Compute mode-specific priority. Higher = more protected; sglang
        # heap evicts the lowest first. `demand` above is already
        # demand_count x path_len (tokens-at-risk). Variants layer on top.
        if _EVICTION_MODE == "plain":
            priority = demand
        elif _EVICTION_MODE == "cluster":
            # Multiplier (1 + n_levels) rewards deeper ancestor chains
            # (more levels of sharing) above equal-demand siblings.
            priority = demand * (1 + n_levels)
        elif _EVICTION_MODE == "recency":
            age = max(0.0, time.time() - last_access) if last_access else 0.0
            priority = int(demand - _EVICTION_RECENCY_W * age)
        elif _EVICTION_MODE == "decay":
            age = max(0.0, time.time() - last_access) if last_access else 0.0
            decay = math.exp(-age / _EVICTION_DECAY_TAU) if _EVICTION_DECAY_TAU > 0 else 1.0
            priority = int(demand * decay)
        else:
            priority = demand  # unreachable: validated above

        if _DEBUG:
            now = time.time()
            if _diag["first_call_ts"] is None:
                _diag["first_call_ts"] = now
            _diag["last_call_ts"] = now
            _diag["gp_calls"] += 1
            if path_len == 0:
                _diag["gp_root_or_empty_path"] += 1
            elif demand == 0:
                _diag["gp_demand_zero"] += 1
                _bump_bucket(0)
            else:
                _diag["gp_demand_gt_zero"] += 1
                _bump_bucket(demand)
                _diag["sum_demand_when_nonzero"] += demand
                _diag["mean_demand_when_nonzero"] = (
                    _diag["sum_demand_when_nonzero"] / _diag["gp_demand_gt_zero"]
                )
                if demand > _diag["max_demand_seen"]:
                    _diag["max_demand_seen"] = demand
                    _diag["path_len_when_max_demand"] = path_len
        return (priority, last_access)


def _eviction_profile() -> dict:
    return dict(_prof)


def _max_ancestor_demand(node, tree) -> Tuple[int, int, int]:
    """Walk from `node` up to root and find the ancestor with the maximum
    "tokens-at-risk" = pending_demand(ancestor_path) x len(ancestor_path).

    Returns (max_value, path_len_at_max, levels_walked).

    Why token-weighted (demand x depth) instead of raw demand:
      Raw `pending_demand` counts rids that would benefit from keeping a
      cached prefix. But "benefit" should weight by how much GPU work each
      rid avoids -- which is proportional to the prefix LENGTH that would
      otherwise be re-prefilled. A 10-rid cluster with a 2048-token prefix
      saves 20480 tokens of prefill if protected; a 50-rid cluster with
      only a 32-token shared prefix saves 1600 tokens. Protecting the
      former saves more compute even though raw demand is lower.
      In our current shared-system-prompts workload all clusters share
      ~2048-token prefixes so this is functionally equivalent to demand,
      but the policy generalizes correctly to mixed-prefix workloads.

    Returns (0, 0, 0) if node has no path.
    """
    segments: list[list[int]] = []
    cur = node
    while cur is not None and getattr(cur, "parent", None) is not None and cur.parent is not cur:
        key = getattr(cur, "key", None)
        if key is None:
            break
        tokens = getattr(key, "token_ids", None)
        if tokens is None:
            break
        segments.append(list(tokens))
        cur = cur.parent
    if not segments:
        return 0, 0, 0
    max_value = 0
    max_len = 0
    levels = 0
    ancestor: list[int] = []
    last_d = -1
    for seg in reversed(segments):
        ancestor = ancestor + seg
        levels += 1
        d = tree.pending_demand(ancestor)
        value = d * len(ancestor)
        if value > max_value:
            max_value = value
            max_len = len(ancestor)
        # Early exit on demand collapse (monotonic: deeper paths can only
        # have ≤ rids than shallower).
        if d == 0 and max_value > 0:
            break
        last_d = d
    return max_value, max_len, levels


def _node_path_tokens(node) -> list[int]:
    """Reconstruct the token sequence from sglang's root to `node` by
    walking parent links and concatenating each node's key's token_ids."""
    segments: list[list[int]] = []
    cur = node
    # Stop when we hit the root (whose parent is None or self-loop depending on version).
    while cur is not None and getattr(cur, "parent", None) is not None and cur.parent is not cur:
        key = getattr(cur, "key", None)
        if key is None:
            break
        # sglang's RadixKey stores tokens in `.token_ids`.
        tokens = getattr(key, "token_ids", None)
        if tokens is None:
            break
        segments.append(list(tokens))
        cur = cur.parent
    # segments are leaf -> root; reverse and flatten.
    out: list[int] = []
    for seg in reversed(segments):
        out.extend(seg)
    return out
