#!/usr/bin/env python3
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

"""Install Peek queue-aware eviction patches into an existing sglang installation.

Patches 4 files:
  1. evict_policy.py     — adds QueueAwareStrategy class
  2. radix_cache.py      — adds queue_ref_count attr, strategy case, ref counting methods
  3. schedule_policy.py   — adds queue-ref bookkeeping in _compute_prefix_matches
  4. server_args.py       — adds "queue-aware" to RADIX_EVICTION_POLICY_CHOICES

Usage:
    python sglang_patches/install.py                    # auto-detect sglang location
    python sglang_patches/install.py /path/to/sglang    # explicit sglang package path

To verify:
    python -c "from sglang.srt.mem_cache.evict_policy import QueueAwareStrategy; print('OK')"
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path


def find_sglang_dir(explicit_path: str | None = None) -> Path:
    """Locate the sglang package directory."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified sglang path does not exist: {p}")

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise ImportError(
            "Cannot find sglang installation. Install it first or pass the path explicitly."
        )
    return Path(spec.origin).parent


def backup(path: Path) -> None:
    """Create a .bak backup if one doesn't already exist."""
    bak = path.with_suffix(path.suffix + ".peek_bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} -> {bak.name}")


# -----------------------------------------------------------------------
# Patch 1: evict_policy.py — add QueueAwareStrategy
# -----------------------------------------------------------------------

QUEUE_AWARE_STRATEGY_CODE = '''

class QueueAwareStrategy(EvictionStrategy):
    """Queue-aware eviction: protects blocks referenced by the waiting queue.

    Delegates scoring to ``peek.scheduler.queue_aware_eviction_priority``
    so that the algorithm lives in peek core, not in the sglang patch.
    """

    def get_priority(self, node: "TreeNode") -> Tuple[float, float]:
        from peek.offline.scheduler import queue_aware_eviction_priority
        return queue_aware_eviction_priority(node)
'''


def patch_evict_policy(sglang_dir: Path) -> bool:
    path = sglang_dir / "srt" / "mem_cache" / "evict_policy.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()
    changed = False

    if "QueueAwareStrategy" not in text:
        backup(path)
        text = text.rstrip() + "\n" + QUEUE_AWARE_STRATEGY_CODE
        changed = True
        print(f"  PATCHED: evict_policy.py — added QueueAwareStrategy")

    # Fix stale superkv references (from old project name)
    if "superkv" in text:
        if not changed:
            backup(path)
        text = text.replace("superkv", "peek")
        changed = True
        print(f"  PATCHED: evict_policy.py — replaced superkv → peek")

    if changed:
        path.write_text(text)
    else:
        print(f"  OK: evict_policy.py already has QueueAwareStrategy")
    return True


# -----------------------------------------------------------------------
# Patch 2: radix_cache.py — queue_ref_count + strategy + methods
# -----------------------------------------------------------------------

MATCH_PREFIX_READONLY_METHOD = '''
    # ------------------------------------------------------------------
    # Read-only prefix matching (no LRU timestamp update)
    # ------------------------------------------------------------------

    def match_prefix_readonly(self, params: MatchPrefixParams) -> MatchResult:
        """Like match_prefix but without updating LRU timestamps.

        Used by Peek's scheduler to score cache state for all prefix
        groups without poisoning eviction priority.  Only the groups
        that are actually admitted should later get a real match_prefix
        (via init_next_round_input) which refreshes their timestamps.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key, update_lru=False)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )
'''

QUEUE_REF_METHODS = '''
    # ------------------------------------------------------------------
    # Queue-aware eviction: reference counting
    # ------------------------------------------------------------------

    def inc_queue_ref(self, node: TreeNode) -> None:
        """Increment queue_ref_count on *node* and all ancestors up to root.

        Called when a waiting-queue request's prefix match lands on *node*,
        indicating that every block on the path from root to *node* is needed
        by at least one pending request.
        """
        while node != self.root_node:
            node.queue_ref_count += 1
            node = node.parent

    def dec_queue_ref(self, node: TreeNode) -> None:
        """Decrement queue_ref_count on *node* and all ancestors up to root."""
        while node != self.root_node:
            node.queue_ref_count = max(0, node.queue_ref_count - 1)
            node = node.parent

    def reset_all_queue_refs(self) -> None:
        """Reset queue_ref_count to 0 on every node in the tree.

        Called at the start of each scheduling cycle before recomputing
        queue references from the current waiting queue.
        """
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            node.queue_ref_count = 0
            for child in node.children.values():
                stack.append(child)
'''


def patch_radix_cache(sglang_dir: Path) -> bool:
    path = sglang_dir / "srt" / "mem_cache" / "radix_cache.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()
    changed = False

    # 2a. Add QueueAwareStrategy import
    if "QueueAwareStrategy" not in text:
        backup(path)
        # Find the evict_policy import block and add QueueAwareStrategy
        text = text.replace(
            "from sglang.srt.mem_cache.evict_policy import (",
            "from sglang.srt.mem_cache.evict_policy import (\n    QueueAwareStrategy,",
            1,
        )
        # If the import was a single-line style, try that too
        if "QueueAwareStrategy" not in text:
            text = re.sub(
                r"(from sglang\.srt\.mem_cache\.evict_policy import .+)",
                r"\1\nfrom sglang.srt.mem_cache.evict_policy import QueueAwareStrategy",
                text,
                count=1,
            )
        changed = True
        print(f"  PATCHED: radix_cache.py — added QueueAwareStrategy import")

    # 2b. Add queue_ref_count attribute to TreeNode.__init__
    if "queue_ref_count" not in text:
        if not changed:
            backup(path)
        # Insert after the priority attribute line
        text = re.sub(
            r"(self\.priority = priority\n)",
            r"\1        # queue-aware eviction: number of waiting-queue requests referencing\n"
            r"        # this node (or an ancestor/descendant sharing its prefix blocks)\n"
            r"        self.queue_ref_count = 0\n",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — added queue_ref_count to TreeNode")

    # 2c. Add queue-aware eviction strategy case
    if '"queue-aware"' not in text:
        if not changed:
            backup(path)
        # Insert the queue-aware case before the else/raise ValueError
        text = re.sub(
            r'(elif self\.eviction_policy == "priority":\s*\n\s*self\.eviction_strategy.*?PriorityStrategy\(\)\n)',
            r'\1        elif self.eviction_policy == "queue-aware":\n'
            r'            self.eviction_strategy: EvictionStrategy = QueueAwareStrategy()\n',
            text,
            count=1,
        )
        # Update the error message
        text = re.sub(
            r"Supported policies: '.*?'\.",
            "Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'queue-aware'.",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — added queue-aware eviction strategy case")

    # 2d. Fix _split_node to inherit queue_ref_count
    if "queue_ref_count = child.queue_ref_count" not in text:
        if not changed:
            backup(path)
        # In _split_node, after `new_node.lock_ref = child.lock_ref`,
        # add `new_node.queue_ref_count = child.queue_ref_count`
        text = re.sub(
            r"(new_node\.lock_ref = child\.lock_ref\n)"
            r"(\s*new_node\.key = child\.key\[:split_len\])",
            r"\1        new_node.queue_ref_count = child.queue_ref_count\n\2",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — _split_node inherits queue_ref_count")

    # 2e. Add queue ref methods to RadixCache class
    if "inc_queue_ref" not in text:
        if not changed:
            backup(path)
        # Insert before evictable_size method
        text = re.sub(
            r"(\n    def evictable_size\(self\):)",
            QUEUE_REF_METHODS + r"\1",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — added inc/dec/reset_all_queue_refs methods")

    # 2f. Add update_lru param to _match_prefix_helper
    if "update_lru" not in text:
        if not changed:
            backup(path)
        # Replace _match_prefix_helper signature and LRU lines
        text = re.sub(
            r"def _match_prefix_helper\(self, node: TreeNode, key: RadixKey\):\n"
            r"(\s+)access_time = time\.monotonic\(\)\n"
            r"\s+node\.last_access_time = access_time\n",
            r"def _match_prefix_helper(self, node: TreeNode, key: RadixKey, update_lru: bool = True):\n"
            r"\1if update_lru:\n"
            r"\1    access_time = time.monotonic()\n"
            r"\1    node.last_access_time = access_time\n"
            r"\1else:\n"
            r"\1    access_time = None\n",
            text,
            count=1,
        )
        # Update child.last_access_time inside the while loop
        text = re.sub(
            r"(\s+)(child = node\.children\[child_key\]\n)"
            r"\s+child\.last_access_time = access_time\n",
            r"\1\2"
            r"\1if update_lru:\n"
            r"\1    child.last_access_time = access_time\n",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — _match_prefix_helper supports update_lru=False")

    # 2g. Add match_prefix_readonly method
    if "match_prefix_readonly" not in text:
        if not changed:
            backup(path)
        text = re.sub(
            r"(\n    def insert\(self, params: InsertParams\))",
            MATCH_PREFIX_READONLY_METHOD + r"\1",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: radix_cache.py — added match_prefix_readonly method")

    # 2h. Fix stale superkv references (from old project name)
    if "superkv" in text:
        if not changed:
            backup(path)
        text = text.replace("superkv", "peek")
        changed = True
        print(f"  PATCHED: radix_cache.py — replaced superkv → peek")

    if not changed:
        print(f"  OK: radix_cache.py already fully patched")

    if changed:
        path.write_text(text)

    return True


# -----------------------------------------------------------------------
# Patch 3: schedule_policy.py — PeekEngine hook (2 minimal patches)
#
# Replaces the old 5-patch approach (3a–3e) which contaminated the LPM
# baseline by running sglang_pre_schedule on ALL servers unconditionally.
#
# Now: a single hook in calc_priority delegates to PeekEngine only when
# eviction_policy == "queue-aware".  Non-Peek servers run stock SGLang.
# -----------------------------------------------------------------------

# Patch 3a: Hook at the top of calc_priority that delegates to PeekEngine.
# Uses lazy-init so the import + is_enabled check happen once.
PEEK_ENGINE_HOOK = """\
        # --- PeekEngine: delegate when queue-aware eviction is active ---
        if not hasattr(self, '_peek_engine'):
            from peek.offline.engine import PeekEngine
            self._peek_engine = (
                PeekEngine(self.tree_cache, self)
                if PeekEngine.is_enabled(self.tree_cache) else None
            )
        if self._peek_engine is not None:
            return self._peek_engine.run(waiting_queue, running_batch)
        # --- stock SGLang below ---
"""

# Patch 3b: Skip pre-matched requests inside _compute_prefix_matches.
# PeekEngine sets _peek_matched=True after group-level matching; these
# requests already have prefix_indices/last_node populated.
PEEK_MATCHED_SKIP = """\
            # PeekEngine: skip requests already matched at group level
            if getattr(r, '_peek_matched', False):
                continue
"""


def patch_schedule_policy(sglang_dir: Path) -> bool:
    path = sglang_dir / "srt" / "managers" / "schedule_policy.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()
    changed = False

    # 3a. PeekEngine hook at the top of calc_priority.
    # Insert after "def calc_priority(...) -> bool:" line.
    if "PeekEngine" not in text:
        backup(path)
        new_text = re.sub(
            r"(def calc_priority\(\s*"
            r"self, waiting_queue.*?running_batch.*?\)"
            r"\s*->\s*bool:\n)",
            r"\1" + PEEK_ENGINE_HOOK,
            text,
            count=1,
            flags=re.DOTALL,
        )
        if new_text != text:
            text = new_text
            changed = True
            print(f"  PATCHED: schedule_policy.py — PeekEngine hook in calc_priority")
        else:
            print(f"  WARNING: Could not patch PeekEngine hook — calc_priority pattern not found")

    # 3b. Skip pre-matched requests in _compute_prefix_matches.
    # Insert at the start of the per-request loop body:
    #   for r in waiting_queue:
    #       <insert here>
    #       prefix_ids = ...
    if "_peek_matched" not in text:
        if not changed:
            backup(path)
        new_text = re.sub(
            r"(for r in waiting_queue:\n)"
            r"(\s+)(prefix_ids = r\.origin_input_ids)",
            r"\1" + PEEK_MATCHED_SKIP + r"\2\3",
            text,
            count=1,
        )
        if new_text != text:
            text = new_text
            changed = True
            print(f"  PATCHED: schedule_policy.py — _peek_matched skip in _compute_prefix_matches")
        else:
            print(f"  WARNING: Could not patch _peek_matched skip — loop pattern not found")

    # 3c. Fix stale superkv references (from old project name)
    if "superkv" in text:
        if not changed:
            backup(path)
        text = text.replace("superkv", "peek")
        changed = True
        print(f"  PATCHED: schedule_policy.py — replaced superkv → peek")

    if not changed:
        print(f"  OK: schedule_policy.py already fully patched")

    if changed:
        path.write_text(text)

    return True


# -----------------------------------------------------------------------
# Patch 4: server_args.py — add "queue-aware" to eviction policy choices
# -----------------------------------------------------------------------

def patch_server_args(sglang_dir: Path) -> bool:
    path = sglang_dir / "srt" / "server_args.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()

    if '"queue-aware"' in text:
        print(f"  OK: server_args.py already has queue-aware in choices")
        return True

    backup(path)

    # Add "queue-aware" to RADIX_EVICTION_POLICY_CHOICES
    text = re.sub(
        r'RADIX_EVICTION_POLICY_CHOICES\s*=\s*\[([^\]]*)\]',
        lambda m: f'RADIX_EVICTION_POLICY_CHOICES = [{m.group(1).rstrip()}, "queue-aware"]',
        text,
        count=1,
    )

    path.write_text(text)
    print(f"  PATCHED: server_args.py — added 'queue-aware' to eviction choices")
    return True


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None

    print("Peek: Installing queue-aware eviction patches into sglang")
    print()

    try:
        sglang_dir = find_sglang_dir(explicit)
    except (ImportError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  sglang location: {sglang_dir}")
    print()

    results = [
        ("evict_policy.py", patch_evict_policy(sglang_dir)),
        ("radix_cache.py", patch_radix_cache(sglang_dir)),
        ("schedule_policy.py", patch_schedule_policy(sglang_dir)),
        ("server_args.py", patch_server_args(sglang_dir)),
    ]

    print()
    all_ok = all(r for _, r in results)
    if all_ok:
        print("All patches applied successfully.")
        print()
        print("Verify with:")
        print('  python -c "from sglang.srt.mem_cache.evict_policy import QueueAwareStrategy; print(\'OK\')"')
        print()
        print("Usage:")
        print("  python -m sglang.launch_server --model-path <model> --radix-eviction-policy queue-aware")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"WARNING: Some patches failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
