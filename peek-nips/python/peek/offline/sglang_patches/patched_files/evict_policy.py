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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class QueueAwareStrategy(EvictionStrategy):
    """Queue-aware eviction: protects blocks referenced by the waiting queue.

    Delegates scoring to ``peek.scheduler.queue_aware_eviction_priority``
    so that the algorithm lives in peek core, not in the sglang patch.
    """

    def get_priority(self, node: "TreeNode") -> Tuple[float, float]:
        from peek.offline.scheduler import queue_aware_eviction_priority
        return queue_aware_eviction_priority(node)


class LRFUStrategy(EvictionStrategy):
    """LRFU: combines frequency (hit_count) and recency (last_access_time).

    Score = hit_count × decay^age.  Popular prefixes keep high scores
    even during short idle gaps; stale prefixes decay naturally.
    O(1) per node — reads two fields already on TreeNode.
    """

    def __init__(self, decay: float = 0.995) -> None:
        import time
        self._decay = decay
        self._time = time

    def get_priority(self, node: "TreeNode") -> float:
        age = self._time.monotonic() - node.last_access_time
        return max(node.hit_count, 1) * (self._decay ** age)
