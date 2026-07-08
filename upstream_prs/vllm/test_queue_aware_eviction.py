# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for queue-aware KV cache eviction.

Upstream location: tests/v1/core/test_queue_aware_eviction.py

Runs on CPU with no model: exercises the free-list victim selection and the
block-pool reference counting directly.
"""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def _queue(n):
    return FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(n)])


def test_queue_ref_count_is_a_real_field():
    # KVCacheBlock is a slotted dataclass; queue_ref_count must be declared.
    block = KVCacheBlock(0)
    assert block.queue_ref_count == 0
    block.queue_ref_count += 1
    assert block.queue_ref_count == 1


def test_protected_blocks_skipped_while_unprotected_exist():
    q = _queue(6)
    blocks = q.get_all_free_blocks()
    blocks[1].queue_ref_count = 2  # protected
    blocks[3].queue_ref_count = 1  # protected
    victims = q.popleft_queue_aware(3)
    # Unprotected blocks 0, 2, 4 are taken in LRU order; 1 and 3 are spared.
    assert [b.block_id for b in victims] == [0, 2, 4]


def test_spill_takes_cheapest_protected_first():
    q = _queue(4)
    blocks = q.get_all_free_blocks()
    blocks[1].queue_ref_count = 5  # expensive
    blocks[2].queue_ref_count = 1  # cheap
    # Only two unprotected (0, 3); the third victim spills to the cheapest
    # protected block (ref 1), not the expensive one (ref 5).
    victims = q.popleft_queue_aware(3)
    assert [b.block_id for b in victims] == [0, 3, 2]


def test_fast_path_matches_popleft_n_when_nothing_protected():
    q = _queue(4)
    victims = q.popleft_queue_aware(2)
    assert [b.block_id for b in victims] == [0, 1]
    assert q.num_free_blocks == 2


def test_block_pool_ref_counting_no_leakage():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    a, b, c = pool.blocks[1], pool.blocks[2], pool.blocks[3]

    pool.inc_queue_ref_count([a, b])
    pool.inc_queue_ref_count([b])  # b needed by two requests
    assert (a.queue_ref_count, b.queue_ref_count, c.queue_ref_count) == (1, 2, 0)

    pool.reset_queue_ref_counts()
    assert all(blk.queue_ref_count == 0 for blk in pool.blocks)

    # Next step references only c; a and b must stay at zero.
    pool.inc_queue_ref_count([c])
    assert (a.queue_ref_count, b.queue_ref_count, c.queue_ref_count) == (0, 0, 1)


def test_inc_skips_null_block():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    pool.inc_queue_ref_count([pool.null_block])
    assert pool.null_block.queue_ref_count == 0


def test_get_new_blocks_uses_queue_aware_when_enabled():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    pool.enable_queue_aware_eviction = True
    # Protect the block at the LRU head so it is not the first victim.
    head = pool.free_block_queue.get_all_free_blocks()[0]
    head.queue_ref_count = 1
    pool._queue_ref_touched.append(head)

    new_blocks = pool.get_new_blocks(1)
    assert head not in new_blocks  # protected block was skipped
