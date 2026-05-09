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

"""CLI entry point for installing Peek patches into supported inference backends.

Usage:
    python -m peek.offline.install          # patch all detected backends
    python -m peek.offline.install sglang   # patch sglang only
    python -m peek.offline.install vllm     # patch vllm only

Importing `peek.offline` also runs `install_all()` automatically so that
downstream `from peek.offline import ...` users get patched backends without
an extra step.

The implementation lives in `peek.offline._patcher`; this file is a thin
CLI shim only.
"""

import sys

if __name__ == "__main__":
    from peek.offline._patcher import install_all, patch_sglang, patch_vllm

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
