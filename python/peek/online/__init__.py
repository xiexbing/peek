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

"""PEEK online mode -- streaming-arrival serving path.

Rust-backed pending radix tree + Cluster-LPM (cLPM) scheduler +
queue-aware eviction, installed into SGLang or vLLM by importing the
matching engine patch hook (peek.online.engines.<engine>.patch_hook)
with the appropriate PEEK_* environment flags set.
"""

from peek._core import PendingTree
from peek.online.eviction import PeekDemandStrategy
from peek.online.policy import compute_main_hits

__all__ = [
    "PendingTree",
    "PeekDemandStrategy",
    "compute_main_hits",
]
