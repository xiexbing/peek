#!/bin/bash
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

# Install peek + sglang 0.5.9 (the canonical W-series engine for sglang).
# Idempotent: re-running is safe. Each step skips if already satisfied.
#
# Usage:
#   bash scripts/install_peek_sglang.sh                    # peek + sglang + bench deps
#   SKIP_BENCH=1 bash scripts/install_peek_sglang.sh       # peek + sglang only
#
# Env knobs:
#   PY                 Python interpreter (default: python3 from PATH)
#   CARGO_HOME         Where rustup installs (default: $HOME/.cargo)
#   SKIP_RUST=1        Don't try to install Rust — assume cargo is on PATH
#   SKIP_BUILD=1       Skip the maturin build (already-built peek)
#   SKIP_SGLANG=1      Don't install sglang (already installed)
#   SKIP_BENCH=1       Don't install bench deps
#   SKIP_LIBNUMA=1     Don't apt-install libnuma-dev (already present)
#   SGLANG_VERSION     Override pinned sglang version (default 0.5.9)
#
# Engine-coexistence note: sglang and vllm pin incompatible torch versions
# and CANNOT share one Python env. Use install_peek_vllm.sh in a separate
# env if you also need vllm.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
SKIP_RUST="${SKIP_RUST:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_LIBNUMA="${SKIP_LIBNUMA:-0}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"

echo "[peek+sglang] repo=$REPO_ROOT"
echo "[peek+sglang] python=$($PY --version 2>&1)  ($(command -v "$PY"))"

# ---------- 0. libnuma (sglang/torch runtime requirement) ----------------
# sglang's torch build links libnuma; without it the engine fails to import.
if [[ "$SKIP_LIBNUMA" != "1" ]]; then
  if ! ldconfig -p 2>/dev/null | grep -q libnuma; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "[peek+sglang] installing libnuma-dev (apt)"
      apt-get update -qq && apt-get install -y libnuma-dev
    else
      echo "[peek+sglang] WARNING: libnuma not found and apt-get unavailable. " \
           "Install libnuma-dev (or your distro's equivalent) before importing sglang." >&2
    fi
  fi
fi

# ---------- 1. Rust toolchain (cargo) ------------------------------------
if [[ "$SKIP_RUST" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    if [[ -x "$CARGO_HOME/bin/cargo" ]]; then
      echo "[peek+sglang] cargo found at $CARGO_HOME/bin/cargo (sourcing env)"
      # shellcheck disable=SC1090
      . "$CARGO_HOME/env"
    else
      echo "[peek+sglang] installing Rust via rustup (minimal profile)"
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
      # shellcheck disable=SC1090
      . "$CARGO_HOME/env"
    fi
  fi
  echo "[peek+sglang] cargo: $(cargo --version)"
fi

# ---------- 2. maturin (build backend) -----------------------------------
if ! "$PY" -m pip show maturin >/dev/null 2>&1; then
  echo "[peek+sglang] installing maturin"
  "$PY" -m pip install --quiet maturin
fi
echo "[peek+sglang] maturin: $("$PY" -m maturin --version 2>&1 || maturin --version)"

# ---------- 3. Build + install peek (Rust + Python wheel) ----------------
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[peek+sglang] building peek native module via 'maturin develop --release'"
  ( cd "$REPO_ROOT" && "$PY" -m maturin develop --release )
fi

if "$PY" -c "import peek; from peek import PendingTree; PendingTree()" 2>/dev/null; then
  echo "[peek+sglang] peek import OK"
else
  echo "[peek+sglang] ERROR: peek failed to import after build" >&2
  exit 1
fi

# ---------- 4. sglang engine ---------------------------------------------
if [[ "$SKIP_SGLANG" != "1" ]]; then
  if ! "$PY" -c "import sglang" 2>/dev/null; then
    echo "[peek+sglang] installing sglang[all]==$SGLANG_VERSION (large download — be patient)"
    # uv is faster than pip for sglang's heavy dep tree; fall back to pip
    # if uv isn't available.
    if command -v uv >/dev/null 2>&1; then
      uv pip install "sglang[all]==$SGLANG_VERSION"
    else
      "$PY" -m pip install --quiet "sglang[all]==$SGLANG_VERSION"
    fi
  fi
  echo "[peek+sglang] sglang: $("$PY" -c 'import sglang; print(sglang.__version__)')"
  if "$PY" -c "import peek.online.engines.sglang.patch_hook" 2>/dev/null; then
    echo "[peek+sglang] peek.online.engines.sglang.patch_hook import OK"
  else
    echo "[peek+sglang] ERROR: peek.online.engines.sglang.patch_hook import failed" >&2
    exit 1
  fi
fi

# ---------- 5. bench/test deps -------------------------------------------
if [[ "$SKIP_BENCH" != "1" ]]; then
  echo "[peek+sglang] installing bench deps (aiohttp, transformers, datasets)"
  "$PY" -m pip install --quiet aiohttp transformers datasets
  echo "[peek+sglang] bench deps installed"
fi

# ---------- 6. (optional) apply offline source-level patches -------------
if [[ "${APPLY_OFFLINE_PATCHES:-0}" == "1" ]]; then
  echo "[peek+sglang] applying offline source-level patches to sglang"
  "$PY" -m peek.offline.install sglang
fi

echo
echo "[peek+sglang] done."
echo
echo "[peek+sglang] To run with PEEK ONLINE (cluster-LPM scheduler + queue-aware eviction):"
echo "  PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \\"
echo "  python -m sglang.launch_server \\"
echo "    --model <hf-id> \\"
echo "    --schedule-policy lpm \\"
echo "    --enable-cache-report"
echo "  # (peek.online.engines.sglang.patch_hook is auto-imported by sglang's"
echo "  #  scheduler process when PEEK_ONLINE_* flags are set.)"
echo
echo "[peek+sglang] To run with PEEK OFFLINE (DFS reorder + queue-aware eviction):"
echo "  APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_sglang.sh   # one-time"
echo "  python -m sglang.launch_server \\"
echo "    --model <hf-id> \\"
echo "    --radix-eviction-policy queue-aware"
echo
echo "[peek+sglang] To run baseline (no PEEK):"
echo "  python -m sglang.launch_server --model <hf-id>"
