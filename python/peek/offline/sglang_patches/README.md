# Peek SGLang Patches: Queue-Aware Eviction

Patches for sglang to add queue-aware KV cache eviction.

**Tested on:** sglang v0.5.9

## What it does

Adds a `--radix-eviction-policy queue-aware` option to sglang that protects
KV cache blocks referenced by requests in the waiting queue from eviction.

### Two-tier eviction model

- **Tier 1 (probationary):** Unreferenced blocks -- evicted first, LRU order
- **Tier 2 (protected):** Blocks referenced by waiting-queue requests -- evicted
  last, ordered by `queue_ref_count * num_tokens` (cheapest-to-recompute first)

## Files modified (4 files)

| File | Location in sglang | What changed |
|------|-------------------|--------------|
| `evict_policy.py` | `srt/mem_cache/evict_policy.py` | Added `QueueAwareStrategy` class |
| `radix_cache.py` | `srt/mem_cache/radix_cache.py` | Added `queue_ref_count` attr, strategy case, `inc/dec/reset_all_queue_refs()` methods |
| `schedule_policy.py` | `srt/managers/schedule_policy.py` | Added queue-ref reset + inc calls in `_compute_prefix_matches()` |
| `server_args.py` | `srt/server_args.py` | Added `"queue-aware"` to `RADIX_EVICTION_POLICY_CHOICES` |

## Installation

### Option 1: Auto-patch (recommended)

```bash
python sglang_patches/install.py
```

The script auto-detects sglang's location, creates `.peek_bak` backups, and
applies all patches idempotently (safe to run multiple times).

### Option 2: Manual copy

Copy the full patched files from `patched_files/` to their sglang locations:

```bash
SGLANG=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent)")
cp patched_files/evict_policy.py    "$SGLANG/srt/mem_cache/evict_policy.py"
cp patched_files/radix_cache.py     "$SGLANG/srt/mem_cache/radix_cache.py"
cp patched_files/schedule_policy.py "$SGLANG/srt/managers/schedule_policy.py"
cp patched_files/server_args.py     "$SGLANG/srt/server_args.py"
```

## Verify

```bash
python -c "from sglang.srt.mem_cache.evict_policy import QueueAwareStrategy; print('OK')"
```

## Usage

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --radix-eviction-policy queue-aware \
    --schedule-policy fcfs
```
