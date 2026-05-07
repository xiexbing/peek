"""Engine-marked CPU-only tests for PEEK's vLLM integration.

These tests verify that ``peek.online.engines.vllm.patch_hook`` actually
installs its monkey-patches on vLLM's Scheduler and BlockPool when the
relevant ``PEEK_ONLINE_*`` flags are set, and that the patched code path
runs end-to-end on a CPU-only BlockPool (no model, no GPU).

The test module sets the relevant environment variables at import time --
the patch_hook reads them once when first imported, so tests cannot toggle
them retroactively. Run with:

    pytest tests/online/test_vllm_patches.py

Marker: ``engine`` (requires vllm but no GPU).
"""

from __future__ import annotations

import os

# Must be set BEFORE the patch_hook is imported. If a parent process or
# earlier test already imported patch_hook with different flags, the
# patch is locked in and these setdefault calls are no-ops; the install-
# state assertions below will then catch the mismatch.
os.environ.setdefault("PEEK_ONLINE_SCHEDULER", "1")
os.environ.setdefault("PEEK_ONLINE_EVICTION", "1")
os.environ.setdefault("PEEK_ONLINE_CLPM", "1")
os.environ.setdefault("PEEK_ONLINE_EVICTION_MODE", "cluster")

import pytest

pytest.importorskip("vllm")

pytestmark = pytest.mark.engine


@pytest.fixture(scope="module")
def installed():
    """Import patch_hook once for the whole module and return its install
    state. Subsequent imports inside this process are idempotent."""
    import peek.online.engines.vllm.patch_hook  # noqa: F401  (import side-effect)
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.block_pool import BlockPool

    return {
        "Scheduler": Scheduler,
        "BlockPool": BlockPool,
        "peek_installed": getattr(Scheduler, "_peek_installed", False),
    }


def test_scheduler_class_marked_installed(installed):
    """patch_hook sets ``Scheduler._peek_installed=True`` exactly once
    per process. This is the canonical 'PEEK is live' marker."""
    assert installed["peek_installed"] is True


def test_scheduler_methods_replaced(installed):
    """The three Scheduler entry points PEEK hooks
    (``schedule``, ``add_request``, ``finish_requests``) must no longer
    point at vLLM's stock implementations once patch_hook installs.

    We compare against a fresh import of the unwrapped classes from
    ``vllm.v1.core.sched.scheduler`` -- but since patch_hook overwrites
    them in place, we instead check function identity against the
    qualname/module heuristic that the patched closures define.
    """
    Scheduler = installed["Scheduler"]
    # Patched closures live in peek.online.engines.vllm.patch_hook;
    # stock methods live in vllm.v1.core.sched.scheduler.
    for name in ("schedule", "add_request", "finish_requests"):
        method = getattr(Scheduler, name)
        # Unbound function; .__module__ is set to the *defining* module.
        mod = getattr(method, "__module__", "")
        assert mod.startswith("peek."), (
            f"Scheduler.{name} should be patched by peek but lives in {mod!r}; "
            f"PEEK_ONLINE_SCHEDULER did not take effect"
        )


def test_blockpool_get_new_blocks_replaced(installed):
    """Eviction hook patches ``BlockPool.get_new_blocks`` so victim
    selection consults the pending-tree demand index. Verify the
    method now lives in peek's patch module."""
    BlockPool = installed["BlockPool"]
    method = BlockPool.get_new_blocks
    mod = getattr(method, "__module__", "")
    assert mod.startswith("peek."), (
        f"BlockPool.get_new_blocks should be patched by peek but lives in {mod!r}; "
        f"PEEK_ONLINE_EVICTION did not take effect"
    )


def test_blockpool_get_new_blocks_returns_blocks(installed):
    """End-to-end: a CPU-only BlockPool with PEEK's patched
    get_new_blocks still allocates blocks correctly when there is no
    pending demand to consult (the no-sharing path)."""
    BlockPool = installed["BlockPool"]
    pool = BlockPool(
        num_gpu_blocks=64,
        enable_caching=True,
        hash_block_size=16,
    )
    free_before = pool.get_num_free_blocks()
    assert free_before > 0

    blocks = pool.get_new_blocks(4)
    assert len(blocks) == 4
    assert pool.get_num_free_blocks() == free_before - 4

    # ref_cnt was bumped to 1 by get_new_blocks (caching path)
    for blk in blocks:
        assert blk.ref_cnt == 1


def test_install_is_idempotent(installed):
    """Re-importing patch_hook in the same process must be a no-op --
    the gate is ``Scheduler._peek_installed``. Verify the patched
    method identities don't change across a second import."""
    Scheduler = installed["Scheduler"]
    BlockPool = installed["BlockPool"]
    methods_before = {
        "Scheduler.schedule": id(Scheduler.schedule),
        "Scheduler.add_request": id(Scheduler.add_request),
        "Scheduler.finish_requests": id(Scheduler.finish_requests),
        "BlockPool.get_new_blocks": id(BlockPool.get_new_blocks),
    }

    # Force a re-import by clearing the cache. patch_hook should detect
    # it's already installed via Scheduler._peek_installed and bail
    # before re-wrapping.
    import importlib
    import peek.online.engines.vllm.patch_hook as ph
    importlib.reload(ph)

    methods_after = {
        "Scheduler.schedule": id(Scheduler.schedule),
        "Scheduler.add_request": id(Scheduler.add_request),
        "Scheduler.finish_requests": id(Scheduler.finish_requests),
        "BlockPool.get_new_blocks": id(BlockPool.get_new_blocks),
    }
    assert methods_before == methods_after, (
        "patch_hook re-import should be a no-op but method identities changed"
    )


def test_pending_tree_top_level_export():
    """patch_hook imports ``PendingTree`` and ``PeekDemandStrategy`` from
    the top-level ``peek`` package. Both must be present -- a regression
    in __init__.py would silently disable the integration."""
    from peek import PendingTree, PeekDemandStrategy
    assert PendingTree is not None
    # Smoke: instantiate and exercise the most basic ops.
    t = PendingTree()
    t.insert(1, [10, 20, 30])
    t.insert(2, [10, 20, 40])
    # Sharing exposed via the API: both rids share prefix [10, 20].
    assert t.has_sharing() is True
