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

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

from peek.offline.trie import PrefixTrie

T = TypeVar("T")


@dataclass
class PeekConfig:
    """Configuration for Peek's reorder and eviction behavior.

    Peek always uses FCFS server scheduling + client-side DFS trie reorder
    + queue-aware eviction.  The server sees FCFS order; the client
    reorders requests by DFS traversal of a prefix trie to maximize
    cache sharing, and the server's queue-aware eviction policy protects
    blocks referenced by pending requests.

    Args:
        max_pending_queue: Maximum requests to reorder at once.
            *None* (default) = reorder the entire pending queue.
        enable_queue_aware_eviction: Whether to use queue-aware eviction
            on the server.  Enabled by default.
    """
    max_pending_queue: int | None = None
    enable_queue_aware_eviction: bool = True

    def should_reorder(self, pending_queue_len: int) -> bool:
        """Whether client-side reorder should be applied."""
        return True

    def schedule_policy_for(self, pending_queue_len: int) -> str:
        """Return the server scheduling policy. FCFS -- PeekEngine controls order."""
        return "fcfs"


def _full_duplication_ratio(sequences: list[list[int]]) -> float:
    """Fraction of sequences that are exact duplicates of another.

    Uses full token sequences (not trie-truncated) so that prompts sharing
    a long prefix but with different suffixes are correctly recognised as
    distinct.
    """
    n = len(sequences)
    if n == 0:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for seq in sequences:
        key = tuple(seq)
        counts[key] = counts.get(key, 0) + 1
    duplicated = sum(c for c in counts.values() if c >= 2)
    return duplicated / n


def reorder_for_prefix_sharing(
    token_id_sequences: list[list[int]],
    *,
    min_sharing_depth: int = 32,
    min_coverage: float = 0.1,
    min_avg_sharing_depth: int = 64,
    max_duplication: float = 0.5,
) -> list[int]:
    """Reorder prompt indices so that prompts sharing the longest common
    token prefixes are adjacent.

    Returns a list where ``result[new_position] = original_index``.

    Skips reorder (returns identity permutation) when:

    1. Prefix sharing is too shallow -- coverage of prompts sharing at least
       *min_sharing_depth* tokens is below *min_coverage*.
    2. Average sharing depth too low -- when the trie-measured average sharing
       depth is below *min_avg_sharing_depth* tokens (default 64), the
       shared prefix is too short for reorder grouping to outweigh the
       disruption to natural ordering (e.g. code completion with ~40-token
       template headers).
    3. Too many exact full-sequence duplicates -- DFS grouping clusters
       identical copies together, but under high concurrency they all arrive
       before the first completes and populates the KV cache.  Natural
       round-robin order spaces duplicates apart, letting each copy reuse
       the previous one's cached KV.  Skipped when the full-sequence
       duplication ratio exceeds *max_duplication*.
    """
    n = len(token_id_sequences)
    if n == 0:
        return []

    if _full_duplication_ratio(token_id_sequences) > max_duplication:
        return list(range(n))

    trie = PrefixTrie()
    for idx, seq in enumerate(token_id_sequences):
        trie.insert(seq, idx)

    coverage, _, avg_sharing_depth = trie.sharing_score(min_depth=min_sharing_depth)
    if coverage < min_coverage:
        return list(range(n))

    if avg_sharing_depth < min_avg_sharing_depth:
        return list(range(n))

    return trie.dfs_order(count_aware=True)


class PeekDispatcher:
    """Incremental trie-based group tagger for online serving.

    Maintains a persistent :class:`PrefixTrie` that tracks all pending
    (waiting-to-be-scheduled) requests.  The trie provides two things:

    **DFS locality** -- groups sharing a prefix path are adjacent in the
    DFS traversal.  This matters for hierarchical prefixes (RAG, multi-
    turn) where group A ``[1,2,3,4,...]`` and group B ``[1,2,3,5,...]``
    share path ``[1,2,3]`` and should be scheduled back-to-back so the
    server's radix cache walk is efficient.

    **Count-aware priority** -- within the DFS ordering, groups are
    sorted by pending request count (largest first).  This ensures the
    server processes high-demand groups before low-demand ones.

    On each :meth:`submit`:

      1. Insert into the trie -- O(max_depth).
      2. If trie structure changed (new group or group removed):
         recompute DFS group order -- O(trie_nodes).
      3. Re-sort the DFS order by pending count -- O(G log G).
      4. Tag request with ``peek:<rank>:<group_hash>:<rid>``.
      5. Send immediately via *send_fn*.
      6. Push pending counts to CacheStateStore.

    On :meth:`remove` (called by PeekEngine at admission, zero delay):

      1. Remove from trie -- O(max_depth).
      2. Decrement pending count.  If group empties, mark structure dirty.
      3. Push pending counts to CacheStateStore.

    Cost per request: O(max_depth + G log G) typical.  The DFS recompute
    (O(trie_nodes)) runs only when the group set changes -- G times over
    the entire workload.

    Usage::

        dispatcher = PeekDispatcher(send_fn=my_send)
        for req in requests:
            dispatcher.submit(req)
    """

    def __init__(
        self,
        send_fn: Callable[[T], None],
        get_token_ids: Callable[[T], list[int]] = lambda r: r["token_ids"],
        max_depth: int = 128,
    ) -> None:
        self._send_fn = send_fn
        self._get_token_ids = get_token_ids
        self._max_depth = max_depth
        self._trie = PrefixTrie(max_depth=max_depth)
        self._next_idx: int = 0
        # Pending count per group key
        self._group_count: dict[tuple[int, ...], int] = {}
        # group_key -> pre-computed hash
        self._group_hashes: dict[tuple[int, ...], int] = {}
        # DFS-ordered group keys (recomputed on structure change)
        self._dfs_keys: list[tuple[int, ...]] = []
        self._structure_dirty: bool = False
        # Shared store for client↔server state exchange
        self._state_store = self._get_state_store()
        # request_id -> (token_ids, prompt_index) for server-initiated remove
        self._rid_to_remove_info: dict[str, tuple[list[int], int]] = {}

    @staticmethod
    def _get_state_store():
        try:
            from peek.offline.engine import CacheStateStore
            return CacheStateStore.get()
        except Exception:
            return None

    def _push_pending_to_server(self) -> None:
        if self._state_store is None:
            return
        pending = {
            self._group_hashes[key]: count
            for key, count in self._group_count.items()
        }
        self._state_store.update_client_pending(pending)

    def _recompute_dfs_keys(self) -> None:
        """Recompute DFS group key order from the trie.

        Only called when the trie structure changes (new group added
        or existing group fully removed).  O(trie_nodes).
        """
        self._dfs_keys = self._trie.dfs_group_keys(count_aware=True)
        self._structure_dirty = False

    def submit(self, request: T) -> None:
        """Insert into trie, rank by DFS + count, tag, and send."""
        idx = self._next_idx
        self._next_idx += 1
        token_ids = self._get_token_ids(request)
        key = tuple(token_ids[: self._max_depth])

        # 1. Trie insert -- O(max_depth)
        self._trie.insert(token_ids, idx)

        # 2. Update pending count; detect structure change
        is_new_group = key not in self._group_count
        self._group_count[key] = self._group_count.get(key, 0) + 1
        if key not in self._group_hashes:
            self._group_hashes[key] = hash(key) & 0xFFFFFFFF

        # 3. Recompute DFS order on structure change -- O(trie_nodes)
        if is_new_group or self._structure_dirty:
            self._recompute_dfs_keys()

        # 4. Sort DFS keys by pending count -- O(G log G)
        #    DFS locality is the primary order; count breaks ties among
        #    groups at the same trie depth.  We sort descending by count
        #    but the DFS sequence determines which groups are "adjacent"
        #    in the server's cache walk.
        ranked_keys = sorted(
            self._dfs_keys,
            key=lambda gk: self._group_count.get(gk, 0),
            reverse=True,
        )
        rank_map = {gk: i for i, gk in enumerate(ranked_keys)}
        rank = rank_map.get(key, 0)
        group_key_hash = self._group_hashes[key]

        # 5. Tag and send
        if isinstance(request, dict):
            orig_rid = request.get("id", request.get("rid", f"req-{idx}"))
            request["rid"] = f"peek:{rank}:{group_key_hash}:{orig_rid}"
            request["_peek_group_order"] = rank
            request["_peek_group_key"] = group_key_hash
            self._rid_to_remove_info[orig_rid] = (token_ids, idx)

        self._send_fn(request)

        # 6. Push fresh pending counts to server
        self._push_pending_to_server()

    def remove(self, token_ids: list[int], prompt_index: int) -> None:
        """Remove a scheduled request from the trie and update counts."""
        key = tuple(token_ids[: self._max_depth])
        self._trie.remove(token_ids, prompt_index)
        if key in self._group_count:
            self._group_count[key] -= 1
            if self._group_count[key] <= 0:
                del self._group_count[key]
                self._group_hashes.pop(key, None)
                # Group gone -- trie structure changed
                self._structure_dirty = True

        self._push_pending_to_server()
