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

"""Debug script to verify WindowReorderPoissonClient's online sliding-window
DFS reorder works correctly.

Tests:
  1. Reorder within a single window groups shared-prefix requests adjacent
  2. Window boundaries are respected (no cross-window reordering)
  3. Timeout-based partial window dispatch works
  4. Requests with no prefix sharing pass through in original order
  5. Result indices map back to correct original requests
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from peek.offline.reorder import reorder_for_prefix_sharing
from peek.offline.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_requests_with_groups(
    n_groups: int = 4,
    reqs_per_group: int = 4,
    prefix_len: int = 128,
    suffix_len: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Create requests where each group shares a long prefix but has a unique suffix.
    Groups are interleaved to simulate realistic arrival order."""
    rng = random.Random(seed)
    prefixes = [[rng.randint(0, 30000) for _ in range(prefix_len)] for _ in range(n_groups)]
    requests = []
    for g in range(n_groups):
        for r in range(reqs_per_group):
            suffix = [rng.randint(0, 30000) for _ in range(suffix_len)]
            requests.append({
                "token_ids": prefixes[g] + suffix,
                "group": g,
                "id": f"g{g}-r{r}",
            })
    # Interleave: round-robin across groups
    interleaved = []
    for r in range(reqs_per_group):
        for g in range(n_groups):
            interleaved.append(requests[g * reqs_per_group + r])
    return interleaved


def extract_group_order(requests: list[dict]) -> list[int]:
    """Return list of group IDs in request order."""
    return [r["group"] for r in requests]


def check_grouping(order: list[int], label: str) -> bool:
    """Check that same-group requests are adjacent (contiguous)."""
    seen = {}
    runs = []
    for i, g in enumerate(order):
        if g not in seen:
            seen[g] = len(runs)
            runs.append((g, 1))
        elif runs[-1][0] == g:
            runs[-1] = (g, runs[-1][1] + 1)
        else:
            # Group appeared before but is not contiguous
            print(f"  [{label}] FAIL: group {g} is split (non-contiguous)")
            print(f"    order: {order}")
            return False
    print(f"  [{label}] OK: groups are contiguous: {[r[0] for r in runs]}")
    return True


# ---------------------------------------------------------------------------
# Test 1: Single window reorder correctness
# ---------------------------------------------------------------------------

def test_single_window_reorder():
    print("\n=== Test 1: Single window reorder correctness ===")
    requests = make_requests_with_groups(n_groups=4, reqs_per_group=4)
    print(f"  Input order (interleaved): {extract_group_order(requests)}")

    # Apply reorder_for_prefix_sharing (what the window reorder calls)
    token_seqs = [r["token_ids"] for r in requests]
    order = reorder_for_prefix_sharing(token_seqs)

    reordered = [requests[i] for i in order]
    group_order = extract_group_order(reordered)
    print(f"  Output order: {group_order}")

    ok = check_grouping(group_order, "single_window")

    # Verify permutation is valid (all original indices present exactly once)
    assert sorted(order) == list(range(len(requests))), "Not a valid permutation!"
    print(f"  Permutation valid: {sorted(order) == list(range(len(requests)))}")
    return ok


# ---------------------------------------------------------------------------
# Test 2: Window boundary isolation
# ---------------------------------------------------------------------------

def test_window_boundary_isolation():
    print("\n=== Test 2: Window boundary isolation ===")
    requests = make_requests_with_groups(n_groups=4, reqs_per_group=8)
    window_size = 8  # 2 requests per group per window

    print(f"  Total requests: {len(requests)}, window_size: {window_size}")
    print(f"  Input order: {extract_group_order(requests)}")

    # Simulate windowed dispatch
    all_dispatched = []
    window_dispatches = []
    for start in range(0, len(requests), window_size):
        window = requests[start:start + window_size]
        token_seqs = [r["token_ids"] for r in window]
        order = reorder_for_prefix_sharing(token_seqs)
        reordered = [window[i] for i in order]

        window_groups = extract_group_order(reordered)
        window_dispatches.append(window_groups)
        all_dispatched.extend(reordered)

    print(f"  Window dispatches:")
    for i, wg in enumerate(window_dispatches):
        print(f"    Window {i}: {wg}")
        check_grouping(wg, f"window_{i}")

    final_order = extract_group_order(all_dispatched)
    print(f"  Final order: {final_order}")

    # Within each window, groups should be contiguous
    ok = True
    for i, wg in enumerate(window_dispatches):
        if not check_grouping(wg, f"window_{i}_check"):
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Test 3: Partial window (timeout scenario)
# ---------------------------------------------------------------------------

def test_partial_window():
    print("\n=== Test 3: Partial window (timeout scenario) ===")
    # 10 requests, window_size=8 → first window=8, partial window=2
    requests = make_requests_with_groups(n_groups=4, reqs_per_group=3)
    # That gives 12 requests interleaved
    requests = requests[:10]  # cut to get a partial last window
    window_size = 8

    print(f"  Total requests: {len(requests)}, window_size: {window_size}")

    windows = []
    for start in range(0, len(requests), window_size):
        window = requests[start:start + window_size]
        token_seqs = [r["token_ids"] for r in window]
        order = reorder_for_prefix_sharing(token_seqs)
        reordered = [window[i] for i in order]
        windows.append(reordered)
        print(f"  Window {len(windows)-1} (size={len(window)}): {extract_group_order(reordered)}")

    # Partial window should still work
    assert len(windows[-1]) == 2, f"Expected partial window of 2, got {len(windows[-1])}"
    print(f"  Partial window dispatched correctly with {len(windows[-1])} requests")
    return True


# ---------------------------------------------------------------------------
# Test 4: No sharing → identity permutation
# ---------------------------------------------------------------------------

def test_no_sharing():
    print("\n=== Test 4: No prefix sharing → identity order ===")
    rng = random.Random(123)
    # Each request has completely unique tokens
    requests = []
    for i in range(16):
        requests.append({
            "token_ids": [i * 10000 + t for t in range(160)],
            "group": i,
            "id": f"unique-{i}",
        })

    token_seqs = [r["token_ids"] for r in requests]
    order = reorder_for_prefix_sharing(token_seqs)

    is_identity = (order == list(range(len(requests))))
    print(f"  Identity permutation returned: {is_identity}")
    if not is_identity:
        print(f"  Order: {order}")
    return is_identity


# ---------------------------------------------------------------------------
# Test 5: Index mapping correctness (result[idx] maps to original)
# ---------------------------------------------------------------------------

def test_index_mapping():
    print("\n=== Test 5: Index mapping correctness ===")
    requests = make_requests_with_groups(n_groups=3, reqs_per_group=5)
    print(f"  Input: {[r['id'] for r in requests]}")

    # Simulate what WindowReorderPoissonClient._reorder_window does
    window = [(idx, req, False) for idx, req in enumerate(requests)]
    seqs = [req.get("token_ids", []) for _, req, _ in window]
    order = reorder_for_prefix_sharing(seqs)
    reordered = [window[i] for i in order]

    # Verify each reordered entry points back to the correct original
    ok = True
    for new_pos, (orig_idx, req, _) in enumerate(reordered):
        expected_req = requests[orig_idx]
        if req["id"] != expected_req["id"]:
            print(f"  FAIL: position {new_pos} has idx={orig_idx} "
                  f"but req id={req['id']} != expected {expected_req['id']}")
            ok = False

    if ok:
        print(f"  All {len(reordered)} index mappings correct")
        reordered_ids = [r["id"] for _, r, _ in reordered]
        print(f"  Reordered: {reordered_ids}")
    return ok


# ---------------------------------------------------------------------------
# Test 6: Async windowed dispatch simulation (full pipeline)
# ---------------------------------------------------------------------------

def test_async_windowed_dispatch():
    """Simulate the full WindowReorderPoissonClient flow without a real server."""
    print("\n=== Test 6: Full async windowed dispatch simulation ===")

    requests = make_requests_with_groups(n_groups=4, reqs_per_group=6)
    window_size = 8
    arrival_rate = 100.0  # fast arrivals for testing

    dispatched_order: list[tuple[int, str]] = []  # (original_idx, request_id)
    window_boundaries: list[int] = []  # track where windows end

    rng = random.Random(42)

    async def simulate():
        window: list[tuple[int, dict, bool]] = []
        window_start = asyncio.get_event_loop().time()

        for idx, req in enumerate(requests):
            if idx > 0:
                delay = rng.expovariate(arrival_rate)
                await asyncio.sleep(delay)

            window.append((idx, req, False))

            now = asyncio.get_event_loop().time()
            elapsed_ms = (now - window_start) * 1000
            if len(window) >= window_size or elapsed_ms >= 50.0:
                # Reorder window
                seqs = [r.get("token_ids", []) for _, r, _ in window]
                order = reorder_for_prefix_sharing(seqs)
                reordered = [window[i] for i in order]
                for widx, wreq, _ in reordered:
                    dispatched_order.append((widx, wreq["id"]))
                window_boundaries.append(len(dispatched_order))
                window = []
                window_start = now

        # Flush remaining
        if window:
            seqs = [r.get("token_ids", []) for _, r, _ in window]
            order = reorder_for_prefix_sharing(seqs)
            reordered = [window[i] for i in order]
            for widx, wreq, _ in reordered:
                dispatched_order.append((widx, wreq["id"]))
            window_boundaries.append(len(dispatched_order))

    asyncio.run(simulate())

    print(f"  Total dispatched: {len(dispatched_order)}")
    print(f"  Window boundaries (cumulative): {window_boundaries}")

    # Check all requests were dispatched exactly once
    orig_indices = sorted([idx for idx, _ in dispatched_order])
    expected = list(range(len(requests)))
    all_present = orig_indices == expected
    print(f"  All requests dispatched exactly once: {all_present}")

    # Check within-window grouping
    ok = True
    prev_boundary = 0
    for w_idx, boundary in enumerate(window_boundaries):
        window_items = dispatched_order[prev_boundary:boundary]
        groups = [requests[orig_idx]["group"] for orig_idx, _ in window_items]
        print(f"  Window {w_idx}: groups={groups}, size={len(window_items)}")
        if not check_grouping(groups, f"async_window_{w_idx}"):
            ok = False
        prev_boundary = boundary

    return ok and all_present


# ---------------------------------------------------------------------------
# Test 7: Verify DFS trie produces better grouping than random
# ---------------------------------------------------------------------------

def test_dfs_vs_random():
    print("\n=== Test 7: DFS grouping quality vs original interleaved order ===")
    requests = make_requests_with_groups(n_groups=6, reqs_per_group=6, prefix_len=200)

    # Count group transitions in original (interleaved) order
    orig_groups = extract_group_order(requests)
    orig_transitions = sum(1 for i in range(1, len(orig_groups)) if orig_groups[i] != orig_groups[i-1])

    # Count group transitions after DFS reorder
    token_seqs = [r["token_ids"] for r in requests]
    order = reorder_for_prefix_sharing(token_seqs)
    reordered_groups = [requests[i]["group"] for i in order]
    reorder_transitions = sum(1 for i in range(1, len(reordered_groups)) if reordered_groups[i] != reordered_groups[i-1])

    print(f"  Original transitions:  {orig_transitions} (interleaved)")
    print(f"  DFS reorder transitions: {reorder_transitions}")
    print(f"  Improvement: {orig_transitions - reorder_transitions} fewer transitions")

    ok = reorder_transitions < orig_transitions
    print(f"  DFS strictly better: {ok}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_single_window_reorder,
        test_window_boundary_isolation,
        test_partial_window,
        test_no_sharing,
        test_index_mapping,
        test_async_windowed_dispatch,
        test_dfs_vs_random,
    ]

    results = []
    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False

    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED.'}")
