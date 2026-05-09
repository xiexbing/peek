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

"""Auto-import peek's vllm hook in every Python interpreter on this PYTHONPATH.

vllm v1 spawns its EngineCore in a child Python process via
multiprocessing.get_context("spawn"). Monkey-patches applied in the parent
do NOT inherit into spawn children -- they're fresh Python interpreters.
A `sitecustomize.py` on the child's PYTHONPATH is imported automatically by
the site machinery before any user code, so it's the cleanest injection
point for both parent and child.

Usage from a launch script:

    export PYTHONPATH="$REPO_ROOT/scripts/peek_sitecustomize:${PYTHONPATH:-}"
    PEEK_ONLINE_CLPM=1 python -m vllm.entrypoints.openai.api_server ...

This shim is a no-op when no PEEK_* flag is set, so the same PYTHONPATH
addition is safe for vanilla baseline runs too.
"""

import os as _os

# Resolve PEEK_PRESET *before* the activation check below, so that a bare
# `PEEK_PRESET=peek-online` (with no per-flag PEEK_ONLINE_* vars set) still
# triggers patch_hook installation. peek.preset.apply() is idempotent and
# uses os.environ.setdefault, so explicit user-set flags still win.
try:
    from peek.preset import apply as _apply_preset
    _apply_preset()
except Exception:
    # peek not importable in this interpreter (e.g., pure-Python install
    # without the Rust core) -- fall through to per-flag detection.
    pass


def _peek_active() -> bool:
    for _name in (
        "PEEK_ONLINE_ENABLED", "PEEK_ONLINE_SCHEDULER", "PEEK_ONLINE_LPM", "PEEK_ONLINE_CLPM",
        "PEEK_ONLINE_EVICTION", "PEEK_ONLINE_PHASE_TRACKING",
    ):
        if _os.environ.get(_name, "").lower() in ("1", "true", "yes", "on", "full"):
            return True
    return False


if _peek_active():
    import logging as _logging
    _log = _logging.getLogger("peek.sitecustomize")
    for _mod in (
        "peek.online.engines.vllm.patch_hook",
        "peek.online.engines.sglang.patch_hook",
    ):
        try:
            __import__(_mod)
        except ModuleNotFoundError:
            # Engine not installed in this interpreter; skip silently.
            pass
        except Exception as _e:
            _log.warning(
                "peek sitecustomize: %s import failed (%s); engine will run vanilla",
                _mod, _e,
            )
