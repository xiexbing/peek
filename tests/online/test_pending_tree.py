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

from peek import PendingTree


def test_insert_and_contains():
    t = PendingTree()
    assert len(t) == 0
    t.insert(1, [10, 20, 30])
    assert 1 in t
    assert len(t) == 1


def test_insert_duplicate_rid_raises():
    t = PendingTree()
    t.insert(1, [1, 2, 3])
    with pytest.raises(ValueError):
        t.insert(1, [9, 9])


def test_remove():
    t = PendingTree()
    t.insert(1, [1, 2, 3])
    t.remove(1)
    assert 1 not in t
    assert len(t) == 0


def test_remove_missing_raises():
    t = PendingTree()
    with pytest.raises(KeyError):
        t.remove(42)


def test_discard():
    t = PendingTree()
    t.insert(1, [1, 2, 3])
    assert t.discard(1) is True
    assert t.discard(1) is False


def test_longest_shared_prefix():
    t = PendingTree()
    t.insert(1, [1, 2, 3, 4])
    t.insert(2, [1, 2, 5, 6])
    t.insert(3, [7, 8, 9])
    # rid 1 and rid 2 share [1,2]; rid 3 shares nothing.
    assert t.longest_shared_prefix(1) == 2
    assert t.longest_shared_prefix(2) == 2
    assert t.longest_shared_prefix(3) == 0


def test_longest_shared_prefix_absent_rid():
    t = PendingTree()
    t.insert(1, [1, 2, 3])
    assert t.longest_shared_prefix(999) == 0


def test_match_prefix():
    t = PendingTree()
    t.insert(1, [1, 2, 3, 4])
    # Partial-edge match: reported as true prefix length (matches sglang).
    assert t.match_prefix([1, 2, 3]) == 3
    assert t.match_prefix([1, 2, 3, 4]) == 4
    assert t.match_prefix([1, 2, 3, 4, 5]) == 4  # overshoot past pending
    assert t.match_prefix([1, 9]) == 1
    assert t.match_prefix([9, 9]) == 0
    t.insert(2, [1, 2, 9])
    assert t.match_prefix([1, 2]) == 2
    assert t.match_prefix([1, 2, 3]) == 3


def test_churn_keeps_tree_consistent():
    t = PendingTree()
    # Insert a handful of rids with overlapping prefixes.
    paths = {
        1: [1, 2, 3, 4, 5],
        2: [1, 2, 3, 9, 9],
        3: [1, 2, 7],
        4: [7, 8, 9],
        5: [1, 2, 3, 4, 5, 6],
    }
    for rid, tokens in paths.items():
        t.insert(rid, tokens)
    assert len(t) == 5

    # rid 1's longest share with others: rid 5 extends [1,2,3,4,5] → shares all 5.
    assert t.longest_shared_prefix(1) == 5
    # rid 4 shares nothing.
    assert t.longest_shared_prefix(4) == 0

    # Remove all and the tree should collapse to just the root.
    for rid in list(paths):
        t.remove(rid)
    assert len(t) == 0
    assert t.node_count() == 1  # only the root remains


def test_large_random_churn():
    import random

    rng = random.Random(0xC0FFEE)
    t = PendingTree()
    active: dict[int, list[int]] = {}
    for step in range(2000):
        if active and (rng.random() < 0.4 or len(active) > 200):
            rid = rng.choice(list(active))
            t.remove(rid)
            del active[rid]
        else:
            rid = step + 1
            length = rng.randint(1, 30)
            # Biased vocabulary so collisions happen often.
            tokens = [rng.randint(0, 7) for _ in range(length)]
            t.insert(rid, tokens)
            active[rid] = tokens
        assert len(t) == len(active)

    # Drain and confirm clean-up.
    for rid in list(active):
        t.remove(rid)
    assert len(t) == 0
    assert t.node_count() == 1
