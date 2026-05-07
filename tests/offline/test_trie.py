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

import pytest
from peek.offline.trie import PrefixTrie
from peek.offline.prompt import PromptRequest


class TestTrieInsertAndDFS:
    def test_single_prompt(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        assert trie.dfs_order() == [0]

    def test_identical_prefixes_are_adjacent(self):
        trie = PrefixTrie()
        # Prompts 0 and 1 share prefix [10, 20]; prompt 2 is different.
        trie.insert([10, 20, 30], 0)
        trie.insert([10, 20, 40], 1)
        trie.insert([50, 60], 2)

        order = trie.dfs_order()
        # 0 and 1 must be adjacent because they share [10, 20]
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[1]) == 1

    def test_three_branches(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        trie.insert([1, 5, 6], 2)
        trie.insert([7, 8], 3)

        order = trie.dfs_order()
        assert set(order) == {0, 1, 2, 3}

        # 0 and 1 share [1,2] -- must be adjacent
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[1]) == 1

    def test_build_from_prompts(self):
        prompts = [
            PromptRequest(id="a", token_ids=[1, 2, 3]),
            PromptRequest(id="b", token_ids=[1, 2, 4]),
            PromptRequest(id="c", token_ids=[5, 6]),
        ]
        trie = PrefixTrie()
        trie.build(prompts)
        order = trie.dfs_order()
        assert set(order) == {0, 1, 2}

        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[1]) == 1

    def test_empty_trie(self):
        trie = PrefixTrie()
        assert trie.dfs_order() == []

    def test_all_identical_tokens(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 3], 1)
        order = trie.dfs_order()
        assert set(order) == {0, 1}
        # Both land on the same leaf node so they are adjacent
        pos = {idx: rank for rank, idx in enumerate(order)}
        assert abs(pos[0] - pos[1]) == 1


class TestSharingScore:
    def test_empty_trie(self):
        trie = PrefixTrie()
        coverage, max_group, _ = trie.sharing_score(min_depth=2)
        assert coverage == 0.0
        assert max_group == 0

    def test_single_prompt(self):
        trie = PrefixTrie()
        trie.insert(list(range(50)), 0)
        coverage, max_group, _ = trie.sharing_score(min_depth=2)
        assert coverage == 0.0
        assert max_group == 0

    def test_no_sharing(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([4, 5, 6], 1)
        trie.insert([7, 8, 9], 2)
        coverage, max_group, _ = trie.sharing_score(min_depth=2)
        assert coverage == 0.0
        assert max_group == 0

    def test_all_share_deep_prefix(self):
        shared = list(range(100))
        trie = PrefixTrie()
        for i in range(5):
            trie.insert(shared + [1000 + i], i)
        coverage, max_group, _ = trie.sharing_score(min_depth=32)
        assert coverage == 1.0
        assert max_group == 5

    def test_partial_sharing(self):
        shared = list(range(50))
        trie = PrefixTrie()
        # 3 prompts share a long prefix
        for i in range(3):
            trie.insert(shared + [1000 + i], i)
        # 2 prompts are unique
        trie.insert([500, 501], 3)
        trie.insert([600, 601], 4)
        coverage, max_group, _ = trie.sharing_score(min_depth=32)
        assert coverage == pytest.approx(3 / 5)
        assert max_group == 3

    def test_shallow_sharing_excluded_by_min_depth(self):
        """Prompts share a prefix of only 4 tokens -- should not count at min_depth=32."""
        trie = PrefixTrie()
        trie.insert([1, 2, 3, 4, 100], 0)
        trie.insert([1, 2, 3, 4, 200], 1)
        trie.insert([1, 2, 3, 4, 300], 2)
        coverage, max_group, _ = trie.sharing_score(min_depth=32)
        assert coverage == 0.0
        assert max_group == 0

        # But they should count with a lower min_depth
        coverage2, max_group2, _ = trie.sharing_score(min_depth=4)
        assert coverage2 == 1.0
        assert max_group2 == 3


class TestPrefixGroups:
    def test_min_prefix_len(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)
        trie.insert([5, 6], 2)

        groups = trie.prefix_groups(min_prefix_len=2)
        # Only prompts 0 and 1 share 2+ tokens
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_no_groups_when_threshold_too_high(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], 0)
        trie.insert([1, 2, 4], 1)

        groups = trie.prefix_groups(min_prefix_len=3)
        assert groups == []

    def test_groups_with_min_prefix_1(self):
        trie = PrefixTrie()
        trie.insert([1, 2], 0)
        trie.insert([1, 3], 1)
        trie.insert([4, 5], 2)

        groups = trie.prefix_groups(min_prefix_len=1)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}
