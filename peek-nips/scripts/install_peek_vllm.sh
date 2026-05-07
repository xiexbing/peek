#!/bin/bash
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

# Install peek + vllm 0.19.1 (the canonical W-series engine for vllm).
# Idempotent: re-running is safe. Each step skips if already satisfied.
#
# Usage:
#   bash scripts/install_peek_vllm.sh                      # peek + vllm + bench deps
#   SKIP_BENCH=1 bash scripts/install_peek_vllm.sh         # peek + vllm only (no aiohttp/transformers/datasets)
#
# Env knobs:
#   PY                Python interpreter (default: python3 from PATH)
#   CARGO_HOME        Where rustup installs (default: $HOME/.cargo)
#   SKIP_RUST=1       Don't try to install Rust -- assume cargo is on PATH
#   SKIP_BUILD=1      Skip the maturin build (already-built peek)
#   SKIP_VLLM=1       Don't install vllm (already installed)
#   SKIP_BENCH=1      Don't install bench deps
#   VLLM_VERSION      Override pinned vllm version (default 0.19.1)
#
# Engine-coexistence note: vllm and sglang pin incompatible torch versions
# and CANNOT share one Python env. Use install_peek_sglang.sh in a separate
# env if you also need sglang.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
SKIP_RUST="${SKIP_RUST:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"

echo "[peek+vllm] repo=$REPO_ROOT"
echo "[peek+vllm] python=$($PY --version 2>&1)  ($(command -v "$PY"))"

# ---------- 1. Rust toolchain (cargo) ------------------------------------
if [[ "$SKIP_RUST" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    if [[ -x "$CARGO_HOME/bin/cargo" ]]; then
      echo "[peek+vllm] cargo found at $CARGO_HOME/bin/cargo (sourcing env)"
      # shellcheck disable=SC1090
      . "$CARGO_HOME/env"
    else
      echo "[peek+vllm] installing Rust via rustup (minimal profile)"
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
      # shellcheck disable=SC1090
      . "$CARGO_HOME/env"
    fi
  fi
  echo "[peek+vllm] cargo: $(cargo --version)"
fi

# ---------- 2. maturin (build backend) -----------------------------------
if ! "$PY" -m pip show maturin >/dev/null 2>&1; then
  echo "[peek+vllm] installing maturin"
  "$PY" -m pip install --quiet maturin
fi
echo "[peek+vllm] maturin: $("$PY" -m maturin --version 2>&1 || maturin --version)"

# ---------- 3. Build + install peek (Rust + Python wheel) ----------------
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[peek+vllm] building peek native module via 'maturin develop --release'"
  ( cd "$REPO_ROOT" && "$PY" -m maturin develop --release )
fi

if "$PY" -c "import peek; from peek import PendingTree; PendingTree()" 2>/dev/null; then
  echo "[peek+vllm] peek import OK"
else
  echo "[peek+vllm] ERROR: peek failed to import after build" >&2
  exit 1
fi

# ---------- 4. vllm engine ------------------------------------------------
if [[ "$SKIP_VLLM" != "1" ]]; then
  if ! "$PY" -c "import vllm" 2>/dev/null; then
    echo "[peek+vllm] installing vllm==$VLLM_VERSION (large download -- be patient)"
    "$PY" -m pip install --quiet "vllm==$VLLM_VERSION"
  fi
  echo "[peek+vllm] vllm: $("$PY" -c 'import vllm; print(vllm.__version__)')"
  if "$PY" -c "import peek.online.engines.vllm.patch_hook" 2>/dev/null; then
    echo "[peek+vllm] peek.online.engines.vllm.patch_hook import OK"
  else
    echo "[peek+vllm] ERROR: peek.online.engines.vllm.patch_hook import failed" >&2
    exit 1
  fi
  # vllm's flash-attn / fused-kernel paths use ninja at first inference;
  # missing it triggers FileNotFoundError mid-bench. Pre-install for safety.
  if ! "$PY" -c "import ninja" 2>/dev/null; then
    echo "[peek+vllm] installing ninja (required for JIT kernel build)"
    "$PY" -m pip install --quiet ninja
  fi
  # The W3 vLLM DP=2 driver uses sglang_router as the front-end across
  # two vLLM TP=2 workers (see benchmarks/w3/run_w3_vllm_dp2.sh). Without
  # it, the router exits with ModuleNotFoundError and the bench gets
  # connection failures.
  if ! "$PY" -c "import sglang_router" 2>/dev/null; then
    echo "[peek+vllm] installing sglang_router (required for W3 vLLM DP=2 driver)"
    "$PY" -m pip install --quiet sglang-router
  fi
fi

# ---------- 5. bench/test deps -------------------------------------------
if [[ "$SKIP_BENCH" != "1" ]]; then
  echo "[peek+vllm] installing bench deps (aiohttp, transformers, datasets)"
  "$PY" -m pip install --quiet aiohttp transformers datasets
  echo "[peek+vllm] bench deps installed"
fi

# ---------- 6. (optional) apply offline source-level patches -------------
if [[ "${APPLY_OFFLINE_PATCHES:-0}" == "1" ]]; then
  echo "[peek+vllm] applying offline source-level patches to vllm"
  "$PY" -m peek.offline.install vllm
fi

REPO_ROOT_ABS="$(cd "$(dirname "$0")/.." && pwd)"

echo
echo "[peek+vllm] done."
echo
echo "[peek+vllm] To run with PEEK ONLINE (cluster-LPM scheduler + queue-aware eviction):"
echo "  # vllm v1 spawns engine workers in child processes; the sitecustomize shim"
echo "  # ensures peek's monkey-patches reach every spawned interpreter."
echo "  export PYTHONPATH=\"$REPO_ROOT_ABS/scripts/peek_sitecustomize:\${PYTHONPATH:-}\""
echo "  PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \\"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <hf-id> \\"
echo "    --enable-prefix-caching \\"
echo "    --gpu-memory-utilization 0.9"
echo
echo "[peek+vllm] To run with PEEK OFFLINE (queue-aware eviction):"
echo "  APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_vllm.sh   # one-time"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <hf-id> \\"
echo "    --enable-prefix-caching"
echo "  # (offline patches activate automatically once installed; vllm has no"
echo "  #  --radix-eviction-policy CLI knob -- the patched BlockPool prefers"
echo "  #  zero-demand victims whenever queue tracking is active.)"
echo
echo "[peek+vllm] To run baseline (no PEEK):"
echo "  python -m vllm.entrypoints.openai.api_server --model <hf-id> --enable-prefix-caching"
