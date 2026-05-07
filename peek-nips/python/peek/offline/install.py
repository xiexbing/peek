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

"""Install Peek patches into supported inference backends.

Usage:
    python -m peek.install          # patch all detected backends
    python -m peek.install sglang   # patch sglang only
    python -m peek.install vllm     # patch vllm only

Also called automatically by peek.__init__ on first import.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

_PATCHES_DIR = Path(__file__).parent / "sglang_patches"
_VLLM_PATCHES_DIR = Path(__file__).parent / "vllm_patches"


def _is_sglang_patched() -> bool:
    """Check if sglang already has queue-aware eviction."""
    try:
        mod = importlib.import_module("sglang.srt.mem_cache.evict_policy")
        return hasattr(mod, "QueueAwareStrategy")
    except (ImportError, ModuleNotFoundError):
        return False


def _is_vllm_patched() -> bool:
    """Check if vllm v1 already has queue-aware eviction patches."""
    try:
        mod = importlib.import_module("vllm.v1.core.kv_cache_utils")
        block_cls = getattr(mod, "KVCacheBlock", None)
        if block_cls is None:
            return False
        # Check if the dataclass has queue_ref_count field
        return "queue_ref_count" in (getattr(block_cls, "__dataclass_fields__", {}) or {}
                                     ) or hasattr(block_cls(0), "queue_ref_count")
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def patch_sglang(force: bool = False) -> bool:
    """Apply queue-aware eviction patches to sglang.

    Returns True if patches were applied (or already present).
    Returns False if sglang is not installed.
    """
    try:
        importlib.import_module("sglang")
    except ImportError:
        return False

    if not force and _is_sglang_patched():
        return True

    install_script = _PATCHES_DIR / "install.py"
    if not install_script.exists():
        print("peek: sglang_patches/install.py not found", file=sys.stderr)
        return False

    result = subprocess.run(
        [sys.executable, str(install_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"peek: sglang patch failed:\n{result.stderr}", file=sys.stderr)
        return False

    # Reload the patched modules
    for mod_name in list(sys.modules):
        if mod_name.startswith("sglang.srt.mem_cache") or mod_name.startswith("sglang.srt.managers"):
            del sys.modules[mod_name]

    return True


def patch_vllm(force: bool = False) -> bool:
    """Apply queue-aware eviction patches to vllm.

    Returns True if patches were applied (or already present).
    Returns False if vllm is not installed.
    """
    try:
        importlib.import_module("vllm")
    except ImportError:
        return False

    if not force and _is_vllm_patched():
        return True

    install_script = _VLLM_PATCHES_DIR / "install.py"
    if not install_script.exists():
        print("peek: vllm_patches/install.py not found", file=sys.stderr)
        return False

    result = subprocess.run(
        [sys.executable, str(install_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"peek: vllm patch failed:\n{result.stderr}", file=sys.stderr)
        return False

    return True


def install_all(force: bool = False) -> None:
    """Patch all detected backends."""
    patched = []
    if patch_sglang(force=force):
        patched.append("sglang")
    if patch_vllm(force=force):
        patched.append("vllm")
    if patched:
        print(f"peek: patches verified for {', '.join(patched)}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    for target in targets:
        if target == "all":
            install_all(force=True)
        elif target == "sglang":
            patch_sglang(force=True)
        elif target == "vllm":
            patch_vllm(force=True)
        else:
            print(f"Unknown target: {target}. Use 'sglang', 'vllm', or 'all'.")
            sys.exit(1)
