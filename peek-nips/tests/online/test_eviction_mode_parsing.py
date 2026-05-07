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

"""Regression tests for ``PEEK_ONLINE_EVICTION_MODE`` parsing.

The paper (§3.2) names the four eviction-priority variants
``plain``, ``cluster``, ``recency``, ``decay``. The README and the vLLM
patch_hook accept those names. SGLang's ``peek.online.eviction`` module
historically used the prefixed names ``demand_cluster`` / ``demand_recency``
/ ``demand_decay`` and would silently fall through its dispatch
``elif`` chain to ``priority = demand`` (i.e., effectively *plain* mode)
when supplied with the documented un-prefixed name -- meaning users
following the README ran a different policy than the paper claims.

These tests pin down the canonical behavior:
  * Paper-canonical names canonicalize to themselves.
  * Legacy ``demand_*`` names alias to the un-prefixed forms.
  * Unknown names fall back to ``plain`` (with a warning).
  * The actual ``get_priority`` dispatch reaches the right branch.

CPU-only, no engine required.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_with_mode(mode: str | None):
    """Reload peek.online.eviction with a specific PEEK_ONLINE_EVICTION_MODE
    value. Returns the freshly-imported module."""
    if mode is None:
        os.environ.pop("PEEK_ONLINE_EVICTION_MODE", None)
    else:
        os.environ["PEEK_ONLINE_EVICTION_MODE"] = mode
    import peek.online.eviction as ev
    importlib.reload(ev)
    return ev


@pytest.mark.parametrize("mode", ["plain", "cluster", "recency", "decay"])
def test_paper_canonical_names_accepted(mode):
    """The four paper-canonical mode names must round-trip unchanged."""
    ev = _reload_with_mode(mode)
    assert ev._EVICTION_MODE == mode


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("demand_cluster", "cluster"),
        ("demand_recency", "recency"),
        ("demand_decay", "decay"),
    ],
)
def test_legacy_demand_prefix_aliases(legacy, canonical):
    """Legacy prefixed names must alias to the un-prefixed forms."""
    ev = _reload_with_mode(legacy)
    assert ev._EVICTION_MODE == canonical


def test_uppercase_normalized():
    """Env var values are case-insensitive (matches the .lower() call)."""
    ev = _reload_with_mode("CLUSTER")
    assert ev._EVICTION_MODE == "cluster"


def test_unknown_falls_back_to_plain_with_warning():
    """An unrecognized mode must warn and fall back to ``plain`` rather
    than silently behaving like one of the real modes."""
    with pytest.warns(UserWarning, match="not recognized"):
        ev = _reload_with_mode("not-a-real-mode")
    assert ev._EVICTION_MODE == "plain"


def test_default_is_plain():
    """No env var -> default ``plain``."""
    ev = _reload_with_mode(None)
    assert ev._EVICTION_MODE == "plain"


def test_get_priority_dispatch_reaches_cluster_branch():
    """End-to-end: with mode=cluster, get_priority must produce the
    cluster-formula priority (demand x (1 + n_levels)), not the plain
    ``demand`` value. The clearest distinguishing case is a node with a
    nonzero ancestor depth, where the multiplier kicks in.

    Since the strategy reads peek's pending tree to derive demand and
    n_levels, we drive both via real PendingTree state."""
    from peek import PendingTree

    # _max_ancestor_demand reads node.key.token_ids -- sglang's RadixKey
    # shape -- to recover the ancestor's prefix path and look up demand.
    class FakeKey:
        def __init__(self, tokens):
            self.token_ids = list(tokens)

    class FakeNode:
        def __init__(self, parent, tokens, value_len, last_access=0.0):
            self.parent = parent
            self.key = FakeKey(tokens)
            self.value = list(range(value_len))
            self.last_access_time = last_access

    class FakeRoot:
        # Root's parent walk terminates here; key is irrelevant since
        # _max_ancestor_demand stops once parent is None or self.
        def __init__(self):
            self.parent = None
            self.key = FakeKey([])
            self.value = []
            self.last_access_time = 0.0

    def build_strategy(mode: str):
        ev = _reload_with_mode(mode)
        # rids 1 and 2 share the [1,2,3] prefix -> pending_demand for any
        # cache path that's a prefix of either rid's tokens is ≥ 1.
        t = PendingTree()
        t.insert(1, [1, 2, 3, 99])
        t.insert(2, [1, 2, 3, 88])
        return ev.PeekDemandStrategy(t)

    root = FakeRoot()
    # Cache leaf carrying tokens [1,2,3] -- exact prefix of both pending rids.
    leaf = FakeNode(root, [1, 2, 3], 3)

    plain_prio, _ = build_strategy("plain").get_priority(leaf)
    cluster_prio, _ = build_strategy("cluster").get_priority(leaf)

    # When demand > 0 (i.e., the leaf's ancestor chain matches a queued
    # prefix), the cluster formula multiplies plain by (1 + n_levels).
    # n_levels >= 1 for any nonempty ancestor chain, so cluster_prio
    # must be strictly greater than plain_prio.
    assert plain_prio > 0, "test setup: pending tree should produce demand > 0"
    assert cluster_prio > plain_prio, (
        f"cluster mode should multiply plain demand: got plain={plain_prio} "
        f"cluster={cluster_prio}"
    )
