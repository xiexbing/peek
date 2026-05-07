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

"""Peek -- Prefix-Sharing Batch Reordering for LLM Inference."""

# Auto-patch supported backends on import
from peek.offline.install import install_all as _install_all
_install_all()

from peek.offline.prompt import PromptRequest
from peek.offline.trie import PrefixTrie, TrieNode
from peek.offline.reorder import (
    reorder_for_prefix_sharing,
    PeekConfig,
    PeekDispatcher,
)
from peek.offline.scheduler import (
    detect_sharing,
    reorder_requests_by_prefix,
    update_queue_ref_counts,
    schedule_hook_vllm,
    vllm_on_schedule,
    detect_sharing_sglang,
    sglang_reorder_waiting_queue,
    sglang_group_schedule,
    vllm_reorder_waiting_queue,
    vllm_group_schedule,
    sglang_pre_schedule,
    sglang_should_run_prefix_matching,
)
from peek.offline.engine_vllm import VllmPeekEngine

__all__ = [
    "PeekConfig",
    "PeekDispatcher",
    "PromptRequest",
    "PrefixTrie",
    "TrieNode",
    "reorder_for_prefix_sharing",
    "detect_sharing",
    "reorder_requests_by_prefix",
    "update_queue_ref_counts",
    "schedule_hook_vllm",
    "vllm_on_schedule",
    "VllmPeekEngine",
    "detect_sharing_sglang",
    "sglang_reorder_waiting_queue",
    "sglang_group_schedule",
    "vllm_reorder_waiting_queue",
    "vllm_group_schedule",
    "sglang_pre_schedule",
    "sglang_should_run_prefix_matching",
]
