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

"""Internal patcher functions used by `peek.offline.__init__` (auto-patch on
import) and the `peek.offline.install` CLI entry point.

Kept separate from `install.py` so that running `python -m peek.offline.install`
does not import the same module twice (once as `peek.offline.install` via the
package's auto-patch, then again as `__main__`), which produces a noisy
RuntimeWarning under runpy.
"""

import importlib
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


def _run_installer(install_script: Path, *extra_args: str) -> bool:
    """Run an engine-specific install.py with optional flags."""
    if not install_script.exists():
        print(f"peek: {install_script} not found", file=sys.stderr)
        return False
    result = subprocess.run(
        [sys.executable, str(install_script), *extra_args],
        capture_output=True, text=True,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return False
    return True


def revert_sglang() -> bool:
    """Restore patched sglang files from their .peek_bak backups.

    Returns True if the revert helper ran successfully (even if some files
    had no backup); False if sglang isn't importable or the helper failed.
    """
    try:
        importlib.import_module("sglang")
    except ImportError:
        print("peek: sglang not installed; nothing to revert.", file=sys.stderr)
        return False
    ok = _run_installer(_PATCHES_DIR / "install.py", "--revert")
    # Force a re-import so the restored modules are picked up next access.
    for mod_name in list(sys.modules):
        if mod_name.startswith("sglang.srt.mem_cache") or mod_name.startswith("sglang.srt.managers"):
            del sys.modules[mod_name]
    return ok


def revert_vllm() -> bool:
    """Restore patched vllm files from their .peek_bak backups.

    Returns True if the revert helper ran successfully (even if some files
    had no backup); False if vllm isn't importable or the helper failed.
    """
    try:
        importlib.import_module("vllm")
    except ImportError:
        print("peek: vllm not installed; nothing to revert.", file=sys.stderr)
        return False
    return _run_installer(_VLLM_PATCHES_DIR / "install.py", "--revert")


def revert_all() -> None:
    """Revert all detected backends."""
    reverted = []
    if revert_sglang():
        reverted.append("sglang")
    if revert_vllm():
        reverted.append("vllm")
    if reverted:
        print(f"peek: reverted {', '.join(reverted)}")
