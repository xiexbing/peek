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

"""Main-cache match helper: one-pass dualwalk of the pending tree against an
external cache.

This used to host `ClusterAwarePolicy` -- a scoring-based scheduler. Scoring
has been retired in favor of LPM-on-peek (see `peek.lpm_integration`). The
dualwalk remains because it's a useful primitive independent of scheduling:
it lets any caller (validation tests, stress tests, future mechanisms) compute
per-rid main-cache match lengths in one pass over the pending tree rather
than N independent walks.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence

from peek import PendingTree


def compute_main_hits(
    tree: PendingTree,
    rids: Iterable[int],  # accepted for API compatibility; unused by the walk
    cache_match_fn: Callable[[Sequence[int]], int],
    *,
    min_pending_count: int = 2,
) -> Dict[int, int]:
    """Amortized main-cache match across the pending queue via dualwalk.

    Walks peek's tree alongside the external cache in a single pass: at each
    edge, `cache_match_fn` is called ONCE for the accumulated-path tokens.

    `min_pending_count` (default 2) skips dualwalking into subtrees where
    fewer than this many pending reqs share the edge -- typically the
    singleton-tail subtrees unique to one req. Those rids inherit the
    shared-prefix main_hit. Pass `min_pending_count=1` to query every edge
    exactly (useful for tests / tail-cache scenarios).
    """
    del rids  # the walk enumerates all tree terminators
    return tree.compute_main_hits(cache_match_fn, min_pending_count)
