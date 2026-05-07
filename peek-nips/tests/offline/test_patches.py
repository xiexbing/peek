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

"""Pre-flight checks: verify Peek patches are installed on sglang.

Run before any benchmark:
    python tests/test_patches.py

Exit code 0 = all good, non-zero = patches missing.
"""

import sys


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"  [OK] {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main() -> int:
    print("Peek patch verification\n")
    results = []

    # 1. sglang importable
    results.append(check(
        "sglang is installed",
        lambda: __import__("sglang"),
    ))

    # 2. QueueAwareStrategy class exists
    results.append(check(
        "QueueAwareStrategy in evict_policy",
        lambda: getattr(
            __import__("sglang.srt.mem_cache.evict_policy", fromlist=["QueueAwareStrategy"]),
            "QueueAwareStrategy",
        ),
    ))

    # 3. radix_cache has queue_ref methods
    def check_radix_cache():
        mod = __import__("sglang.srt.mem_cache.radix_cache", fromlist=["RadixCache"])
        cls = mod.RadixCache
        for method in ["inc_queue_ref", "dec_queue_ref", "reset_all_queue_refs"]:
            assert hasattr(cls, method), f"RadixCache missing {method}"
    results.append(check("RadixCache has queue-ref methods", check_radix_cache))

    # 4. TreeNode has queue_ref_count
    def check_tree_node():
        mod = __import__("sglang.srt.mem_cache.radix_cache", fromlist=["TreeNode"])
        node_cls = mod.TreeNode
        # Check the attribute exists in __init__ or __slots__
        node = node_cls()
        assert hasattr(node, "queue_ref_count"), "TreeNode missing queue_ref_count"
    results.append(check("TreeNode has queue_ref_count attribute", check_tree_node))

    # 5. server_args accepts queue-aware
    def check_server_args():
        mod = __import__("sglang.srt.server_args", fromlist=["ServerArgs"])
        # Find the eviction policy choices
        import inspect
        src = inspect.getsource(mod)
        assert "queue-aware" in src, "server_args.py missing 'queue-aware' in eviction choices"
    results.append(check("server_args accepts 'queue-aware' eviction policy", check_server_args))

    # 6. peek reorder module works
    results.append(check(
        "peek.reorder importable",
        lambda: __import__("peek.reorder", fromlist=["reorder_for_prefix_sharing"]),
    ))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed.")

    if passed < total:
        print("\nFix: run 'python sglang_patches/install.py' then re-run this test.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
