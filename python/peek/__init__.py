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

"""PEEK -- Queue-informed KV cache management for LLM serving.

Two operating modes:
  * peek.online   -- streaming-arrival path (Rust-backed pending tree,
                    Cluster-LPM scheduling, queue-aware eviction);
                    imported by env-var-gated patch hooks.
  * peek.offline  -- batch path (Python prefix trie, DFS reorder,
                    queue-aware eviction); also publishes runtime
                    patch installers for SGLang and vLLM.

The Rust pending-tree primitive is exported at the top level for
convenience and is shared by both modes when a Rust toolchain is
available; the offline path remains usable without it.
"""

# Rust extension lives at the top level. Both modes can use it; the
# offline mode degrades to its pure-Python trie if the extension is not
# built, so the import is best-effort.
try:
    from peek._core import PendingTree  # type: ignore
    _CORE_AVAILABLE = True
except ImportError:  # pragma: no cover -- pure-Python (offline-only) install.
    PendingTree = None  # type: ignore
    _CORE_AVAILABLE = False

from peek.online.eviction import PeekDemandStrategy

__all__ = [
    "PendingTree",
    "PeekDemandStrategy",
    "online",
    "offline",
]
