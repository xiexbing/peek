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

from peek.offline.prompt import PromptRequest


class TrieNode:
    __slots__ = ("children", "prompt_indices", "count")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.prompt_indices: list[int] = []
        self.count: int = 0  # number of prompts whose prefix passes through this node


class PrefixTrie:
    def __init__(self, max_depth: int = 128) -> None:
        self.root = TrieNode()
        self.max_depth = max_depth
        self._num_prompts: int = 0

    def insert(self, token_ids: list[int], prompt_index: int) -> None:
        """Insert a prompt's token sequence into the trie.

        Only the first ``max_depth`` tokens are inserted.  Most prefix
        sharing happens in the first 128 tokens (system prompts), so
        deeper tokens add O(N*D) overhead without benefit.

        Tracks pass-through counts at each node for count-aware DFS ordering.
        """
        node = self.root
        depth = min(len(token_ids), self.max_depth)
        for i in range(depth):
            token = token_ids[i]
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.count += 1
        node.prompt_indices.append(prompt_index)
        self._num_prompts += 1

    def remove(self, token_ids: list[int], prompt_index: int) -> bool:
        """Remove a prompt from the trie.  Returns True if found and removed.

        Walks the token path, removes *prompt_index* from the leaf's
        ``prompt_indices``, decrements ``count`` on every ancestor, and
        prunes empty leaf nodes bottom-up.  Cost: O(D).
        """
        depth = min(len(token_ids), self.max_depth)
        # Collect (parent, token, child) along the path.
        path: list[tuple[TrieNode, int, TrieNode]] = []
        node = self.root
        for i in range(depth):
            token = token_ids[i]
            child = node.children.get(token)
            if child is None:
                return False
            path.append((node, token, child))
            node = child

        try:
            node.prompt_indices.remove(prompt_index)
        except ValueError:
            return False

        self._num_prompts -= 1

        # Decrement counts along the path.
        for _, _, child in path:
            child.count -= 1

        # Prune empty nodes bottom-up.
        for parent, token, child in reversed(path):
            if child.count == 0 and not child.children and not child.prompt_indices:
                del parent.children[token]
            else:
                break
        return True

    def clear(self) -> None:
        """Reset the trie to empty state."""
        self.root = TrieNode()
        self._num_prompts = 0

    def build(self, prompts: list[PromptRequest]) -> None:
        """Bulk-insert prompts into the trie."""
        for idx, prompt in enumerate(prompts):
            self.insert(prompt.token_ids, idx)

    def dfs_group_keys(self, count_aware: bool = False) -> list[tuple[int, ...]]:
        """Return leaf-node paths in DFS order -- one key per group.

        Each key is the tuple of tokens along the root-to-leaf path where
        at least one prompt index lives.  Cost: O(trie nodes), independent
        of the number of prompts N.

        When *count_aware* is True, children at each node are visited in
        descending order of subtree count, so the largest prefix-sharing
        groups appear first in the output -- matching :meth:`dfs_order`
        with ``count_aware=True``.
        """
        keys: list[tuple[int, ...]] = []
        # (node, path_tuple)
        stack: list[tuple[TrieNode, tuple[int, ...]]] = [(self.root, ())]
        while stack:
            node, path = stack.pop()
            if node.prompt_indices:
                keys.append(path)
            children = list(node.children.items())
            if count_aware and len(children) > 1:
                # Sort ascending by count so highest-count child is pushed
                # last -> popped first -> appears first in output.
                children.sort(key=lambda x: x[1].count)
            else:
                children.reverse()
            for token, child in children:
                stack.append((child, path + (token,)))
        return keys

    def dfs_order(self, count_aware: bool = False) -> list[int]:
        """DFS traversal returning prompt indices so that prompts sharing the
        longest common prefixes are adjacent.

        When *count_aware* is True, children at each node are visited in
        descending order of subtree count (pass-through count).  This places
        the largest prefix-sharing groups first in the output, maximising
        cache hit potential when the scheduler selects from the front.
        """
        order: list[int] = []
        self._dfs(self.root, order, count_aware)
        return order

    def _dfs(self, node: TrieNode, order: list[int], count_aware: bool = False) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            order.extend(n.prompt_indices)
            children = list(n.children.values())
            if count_aware and len(children) > 1:
                # Sort ascending by count so highest-count child is pushed
                # last -> popped first -> its subtree appears first in output.
                children.sort(key=lambda c: c.count)
            else:
                children.reverse()
            stack.extend(children)

    def sharing_score(self, min_depth: int = 32) -> tuple[float, int, float]:
        """Measure what fraction of prompts share a prefix of at least
        *min_depth* tokens.

        Returns ``(coverage, max_group_size, avg_sharing_depth)`` where:
        - *coverage* is the ratio of prompts that belong to a group of
          size >= 2 at or beyond *min_depth*,
        - *max_group_size* is the largest such group,
        - *avg_sharing_depth* is the count-weighted average depth at which
          groups are found (the typical shared prefix length).

        Uses the same maximal-group logic as :meth:`prefix_groups`: once a node
        at depth >= *min_depth* has count >= 2, its subtree is counted as one
        group and not recursed further.

        Cost: O(trie nodes above *min_depth*), negligible compared to the
        O(N * max_depth) build cost.
        """
        if self._num_prompts == 0:
            return (0.0, 0, 0.0)

        grouped = 0
        max_group = 0
        depth_sum = 0  # sum of depth * count for weighted average
        # (node, depth)
        stack: list[tuple[TrieNode, int]] = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth >= min_depth and node.count >= 2:
                grouped += node.count
                # Walk down single-child chains to find the real branching
                # depth (where the shared prefix actually ends), not just
                # the min_depth threshold where we first detected sharing.
                real_depth = depth
                walk = node
                while len(walk.children) == 1 and not walk.prompt_indices:
                    walk = next(iter(walk.children.values()))
                    real_depth += 1
                depth_sum += real_depth * node.count
                if node.count > max_group:
                    max_group = node.count
                continue  # maximal group -- don't recurse
            for child in node.children.values():
                stack.append((child, depth + 1))

        coverage = grouped / self._num_prompts
        avg_depth = depth_sum / grouped if grouped > 0 else 0.0
        return (coverage, max_group, avg_depth)

    def prefix_groups(self, min_prefix_len: int) -> list[list[int]]:
        """Return groups of prompt indices that share at least
        *min_prefix_len* common prefix tokens."""
        groups: list[list[int]] = []
        self._collect_groups(self.root, 0, min_prefix_len, groups)
        return groups

    def prefix_groups_with_tokens(
        self, min_prefix_len: int,
    ) -> list[tuple[list[int], list[int]]]:
        """Return groups with their shared prefix token sequences.

        Each entry is ``(prefix_tokens, [prompt_indices...])``.
        Only groups with >= 2 members are returned; singletons are left
        for the caller to handle.
        """
        groups: list[tuple[list[int], list[int]]] = []
        # Stack: (node, depth, prefix_tokens)
        stack: list[tuple[TrieNode, int, list[int]]] = [(self.root, 0, [])]
        while stack:
            n, d, prefix = stack.pop()
            if d >= min_prefix_len:
                indices = self._all_indices(n)
                if len(indices) >= 2:
                    groups.append((prefix, indices))
                    continue  # maximal group
            for token, child in n.children.items():
                stack.append((child, d + 1, prefix + [token]))
        return groups

    def _collect_groups(
        self,
        node: TrieNode,
        depth: int,
        min_prefix_len: int,
        groups: list[list[int]],
    ) -> None:
        stack: list[tuple[TrieNode, int]] = [(node, depth)]
        while stack:
            n, d = stack.pop()
            if d >= min_prefix_len:
                indices = self._all_indices(n)
                if len(indices) >= 2:
                    groups.append(indices)
                    continue  # don't recurse further; this group is already maximal
            for child in n.children.values():
                stack.append((child, d + 1))

    @staticmethod
    def _all_indices(node: TrieNode) -> list[int]:
        """Collect all prompt indices in the subtree rooted at *node*."""
        result: list[int] = list(node.prompt_indices)
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            result.extend(n.prompt_indices)
            stack.extend(n.children.values())
        return result
