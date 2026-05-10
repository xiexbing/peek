#!/usr/bin/env python3
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

"""Install Peek queue-aware eviction patches into an existing vLLM v1 installation.

Patches 4 files:
  1. kv_cache_utils.py  -- adds queue_ref_count to KVCacheBlock, queue-aware popleft
  2. block_pool.py      -- adds reset/inc queue ref methods, queue-aware get_new_blocks
  3. scheduler.py       -- adds _update_queue_refs() hook in schedule()
  4. cache.py           -- adds enable_queue_aware_eviction CacheConfig field

Usage:
    python vllm_patches/install.py                  # auto-detect vllm location
    python vllm_patches/install.py /path/to/vllm    # explicit vllm package path
    python vllm_patches/install.py --revert         # restore each patched file
                                                    # from its .peek_bak backup

To verify:
    python -c "from vllm.v1.core.kv_cache_utils import KVCacheBlock; b = KVCacheBlock(0); print('queue_ref_count' in dir(b))"
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path


def find_vllm_dir(explicit_path: str | None = None) -> Path:
    """Locate the vllm package directory."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified vllm path does not exist: {p}")

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise ImportError(
            "Cannot find vllm installation. Install it first or pass the path explicitly."
        )
    return Path(spec.origin).parent


def backup(path: Path) -> None:
    """Create a .bak backup if one doesn't already exist."""
    bak = path.with_suffix(path.suffix + ".peek_bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} -> {bak.name}")


# Files this installer patches; relative to the vllm package root.
PATCHED_FILES = (
    Path("v1") / "core" / "kv_cache_utils.py",
    Path("v1") / "core" / "block_pool.py",
    Path("v1") / "core" / "sched" / "scheduler.py",
    Path("config") / "cache.py",
)


def revert(vllm_dir: Path) -> int:
    """Restore each patched file from its .peek_bak backup.

    Returns the number of files restored. Files without a backup are
    reported and skipped.
    """
    restored = 0
    for rel in PATCHED_FILES:
        path = vllm_dir / rel
        bak = path.with_suffix(path.suffix + ".peek_bak")
        if not bak.exists():
            print(f"  SKIP: {rel} -- no backup at {bak.name}")
            continue
        shutil.copy2(bak, path)
        bak.unlink()
        print(f"  RESTORED: {rel} from {bak.name}")
        restored += 1
    return restored


# -----------------------------------------------------------------------
# Patch 1: kv_cache_utils.py -- add queue_ref_count to KVCacheBlock
# -----------------------------------------------------------------------

def patch_kv_cache_utils(vllm_dir: Path) -> bool:
    path = vllm_dir / "v1" / "core" / "kv_cache_utils.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()
    changed = False

    # 1a. Add queue_ref_count attribute to KVCacheBlock
    if "queue_ref_count" not in text:
        backup(path)
        # Insert after is_null: bool = False
        text = re.sub(
            r"(    is_null: bool = False\n)",
            r"\1\n"
            r"    # Queue-aware eviction: number of waiting-queue requests\n"
            r"    # whose prefix includes this block.\n"
            r"    queue_ref_count: int = 0\n",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: kv_cache_utils.py -- added queue_ref_count to KVCacheBlock")

    # 1b. Add queue-aware popleft to FreeKVCacheBlockQueue
    if "popleft_queue_aware" not in text:
        if not changed:
            backup(path)
        # Insert the method after the existing popleft_n method.
        # Find the end of popleft_n by matching the return statement.
        queue_aware_method = '''
    def popleft_queue_aware(self, n: int) -> list["KVCacheBlock"]:
        """Pop *n* blocks, preferring those with lowest queue_ref_count.

        Two-tier eviction:
          Tier 1: blocks with queue_ref_count == 0 (unprotected, LRU order)
          Tier 2: blocks with queue_ref_count > 0 (protected, lowest cost first)

        Falls back to plain LRU (popleft_n) when no blocks are protected.
        """
        if n > self.num_free_blocks:
            raise ValueError(
                f"Cannot pop {n} blocks from the free list "
                f"(only {self.num_free_blocks} free)"
            )

        # Collect all free blocks with their queue_ref_count
        unprotected: list["KVCacheBlock"] = []
        protected: list["KVCacheBlock"] = []
        curr = self.fake_free_list_head.next_free_block
        while curr is not self.fake_free_list_tail and curr is not None:
            if curr.queue_ref_count == 0:
                unprotected.append(curr)
            else:
                protected.append(curr)
            curr = curr.next_free_block

        # If nothing is protected, just use fast LRU path
        if not protected:
            return self.popleft_n(n)

        # Pick from unprotected first (LRU order), then protected (lowest cost)
        # Cost = queue_ref_count (lower = cheaper to evict)
        protected.sort(key=lambda b: b.queue_ref_count)
        victims = unprotected[:n]
        if len(victims) < n:
            victims.extend(protected[: n - len(victims)])

        for block in victims:
            self.remove(block)

        return victims

'''
        # Insert before the last method or at end of class
        text = re.sub(
            r"(\n    def get_all_free_blocks\b)",
            queue_aware_method + r"\1",
            text,
            count=1,
        )
        # Fallback: insert before end of file if get_all_free_blocks not found
        if "popleft_queue_aware" not in text:
            # Try inserting after popleft_n
            text = re.sub(
                r"(        return ret\n\n    def remove\b)",
                r"\1",
                text,
                count=1,
            )
            # Just append to the class by finding the remove method and adding after
            text = text.rstrip() + "\n" + queue_aware_method
        changed = True
        print(f"  PATCHED: kv_cache_utils.py -- added popleft_queue_aware method")

    if not changed:
        print(f"  OK: kv_cache_utils.py already fully patched")

    if changed:
        path.write_text(text)

    return True


# -----------------------------------------------------------------------
# Patch 2: block_pool.py -- queue ref methods + queue-aware get_new_blocks
# -----------------------------------------------------------------------

QUEUE_REF_METHODS = '''
    # ------------------------------------------------------------------
    # Queue-aware eviction: reference counting
    # ------------------------------------------------------------------

    def reset_queue_ref_counts(self) -> None:
        """Reset queue_ref_count to 0 on all blocks."""
        for block in self.blocks:
            block.queue_ref_count = 0

    def inc_queue_ref_count(self, blocks: list["KVCacheBlock"]) -> None:
        """Increment queue_ref_count on a list of cached blocks."""
        for block in blocks:
            if not block.is_null:
                block.queue_ref_count += 1

'''


def patch_block_pool(vllm_dir: Path) -> bool:
    path = vllm_dir / "v1" / "core" / "block_pool.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()
    changed = False

    # 2a. Add queue ref methods
    if "reset_queue_ref_counts" not in text:
        backup(path)
        # Insert before get_num_free_blocks
        text = re.sub(
            r"(\n    def get_num_free_blocks\(self\))",
            QUEUE_REF_METHODS + r"\1",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: block_pool.py -- added reset/inc_queue_ref_counts methods")

    # 2b. Add enable_queue_aware_eviction flag to __init__
    if "enable_queue_aware_eviction" not in text:
        if not changed:
            backup(path)
        # Add parameter to __init__
        text = re.sub(
            r"(        metrics_collector: KVCacheMetricsCollector \| None = None,\n    \):)",
            r"        metrics_collector: KVCacheMetricsCollector | None = None,\n"
            r"        enable_queue_aware_eviction: bool = False,\n    ):",
            text,
            count=1,
        )
        # Add attribute after self.metrics_collector = metrics_collector
        # Auto-enable via PEEK_QUEUE_AWARE env var
        text = re.sub(
            r"(        self\.metrics_collector = metrics_collector\n)",
            r"\1        import os as _os\n"
            r"        self.enable_queue_aware_eviction = (\n"
            r"            enable_queue_aware_eviction\n"
            r'            or _os.environ.get("PEEK_QUEUE_AWARE") == "1"\n'
            r"        )\n",
            text,
            count=1,
        )
        changed = True
        print(f"  PATCHED: block_pool.py -- added enable_queue_aware_eviction flag")

    # 2c. Use queue-aware eviction in get_new_blocks when enabled
    if "popleft_queue_aware" not in text:
        if not changed:
            backup(path)
        text = text.replace(
            "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)",
            "        if self.enable_queue_aware_eviction:\n"
            "            ret: list[KVCacheBlock] = self.free_block_queue.popleft_queue_aware(num_blocks)\n"
            "        else:\n"
            "            ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)",
            1,
        )
        changed = True
        print(f"  PATCHED: block_pool.py -- get_new_blocks uses queue-aware eviction")

    if not changed:
        print(f"  OK: block_pool.py already fully patched")

    if changed:
        path.write_text(text)

    return True


# -----------------------------------------------------------------------
# Patch 3: scheduler.py -- prefix-aware scheduling + queue-aware eviction
# -----------------------------------------------------------------------

PEEK_SCHEDULER_CODE = '''\
        # Peek: all feature gating is inside vllm_on_schedule
        from peek.offline.scheduler import vllm_on_schedule as _peek_hook
        _peek_hook(self)

'''


def patch_scheduler(vllm_dir: Path) -> bool:
    path = vllm_dir / "v1" / "core" / "sched" / "scheduler.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()

    marker = "_peek_hook"

    if marker in text:
        print(f"  OK: scheduler.py already fully patched")
        return True

    backup(path)

    # Remove old-style patch if present (from prior versions)
    if "_peek_enabled" in text or "reset_queue_ref_counts" in text:
        text = re.sub(
            r'        # (?:Peek|Queue-aware).*?(?=        # (?:NOTE|There\'s no))',
            '',
            text,
            count=1,
            flags=re.DOTALL,
        )

    # Insert at the beginning of schedule(), after the method signature
    text = re.sub(
        r'(    def schedule\(self\) -> SchedulerOutput:\n'
        r'        # NOTE\(woosuk\) on the scheduling algorithm:\n)',
        r'\1' + PEEK_SCHEDULER_CODE,
        text,
        count=1,
    )

    if marker not in text:
        # Fallback: try a broader match
        text = re.sub(
            r'(    def schedule\(self\) -> SchedulerOutput:\n)',
            r'\1' + PEEK_SCHEDULER_CODE,
            text,
            count=1,
        )

    if marker in text:
        path.write_text(text)
        print(f"  PATCHED: scheduler.py -- added Peek scheduling hook")
        return True
    else:
        print(f"  FAILED: scheduler.py -- could not find insertion point")
        return False


# -----------------------------------------------------------------------
# Patch 4: server CLI -- add --enable-queue-aware-eviction flag
# -----------------------------------------------------------------------

def patch_cache_config(vllm_dir: Path) -> bool:
    """Add enable_queue_aware_eviction to CacheConfig and wire it through."""
    path = vllm_dir / "config" / "cache.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return False

    text = path.read_text()

    if "enable_queue_aware_eviction" in text:
        print(f"  OK: cache.py already has enable_queue_aware_eviction")
        return True

    backup(path)

    # Add field to CacheConfig dataclass, after enable_prefix_caching
    text = re.sub(
        r"(    enable_prefix_caching: bool = True\n)",
        r"\1    enable_queue_aware_eviction: bool = False\n",
        text,
        count=1,
    )

    if "enable_queue_aware_eviction" in text:
        path.write_text(text)
        print(f"  PATCHED: cache.py -- added enable_queue_aware_eviction field")
        return True
    else:
        print(f"  FAILED: cache.py -- could not find insertion point")
        return False


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    do_revert = False
    if "--revert" in args:
        do_revert = True
        args = [a for a in args if a != "--revert"]
    explicit = args[0] if args else None

    if do_revert:
        print("Peek: Reverting prefix-aware scheduling + queue-aware eviction patches from vLLM")
    else:
        print("Peek: Installing prefix-aware scheduling + queue-aware eviction patches into vLLM")
    print()

    try:
        vllm_dir = find_vllm_dir(explicit)
    except (ImportError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  vllm location: {vllm_dir}")
    print()

    if do_revert:
        n = revert(vllm_dir)
        print()
        print(f"Reverted {n} file(s).")
        return

    results = [
        ("kv_cache_utils.py", patch_kv_cache_utils(vllm_dir)),
        ("block_pool.py", patch_block_pool(vllm_dir)),
        ("scheduler.py", patch_scheduler(vllm_dir)),
        ("cache.py", patch_cache_config(vllm_dir)),
    ]

    print()
    all_ok = all(r for _, r in results)
    if all_ok:
        print("All patches applied successfully.")
        print()
        print("Verify with:")
        print('  python -c "from vllm.v1.core.kv_cache_utils import KVCacheBlock; '
              "b = KVCacheBlock(0); print('queue_ref_count:', b.queue_ref_count)\"")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"WARNING: Some patches failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
