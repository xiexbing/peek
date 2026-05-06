# PEEK — example commands

Copy-paste recipes for each combination of mode (online / offline) and
engine (sglang / vllm). Pick the row, run the install once, then the
launch command.

| #   | mode    | engine | scheduler win                                    | eviction win                                       |
| --- | ------- | ------ | ------------------------------------------------ | -------------------------------------------------- |
| 1   | online  | sglang | cLPM dual-walk + cluster-aware admission         | queue-aware eviction (`demand_cluster` default)    |
| 2   | online  | vllm   | cLPM reorder of `self.waiting`                   | queue-aware eviction (`BlockPool.get_new_blocks`)  |
| 3   | offline | sglang | DFS prefix-trie reorder of the waiting batch     | `--radix-eviction-policy queue-aware`              |
| 4   | offline | vllm   | DFS reorder via `_update_queue_refs()` hook      | queue-aware victim pick in `BlockPool`             |

In every example below, replace `<hf-id>` with a HuggingFace model id
(e.g. `Qwen/Qwen2.5-32B-Instruct`).

---

## 1. Online + SGLang

**Install once** (Rust toolchain + conda env + sglang 0.5.9 + peek):

```bash
bash scripts/install_peek_sglang.sh
```

**Launch the server** with cluster-aware scheduling and queue-aware eviction:

```bash
PEEK_ONLINE_SCHEDULER=1 \
PEEK_ONLINE_EVICTION=1 \
PEEK_ONLINE_CLPM=1 \
PEEK_ONLINE_CLPM_GROUP_MAJOR=1 \
PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 \
PEEK_ONLINE_EVICTION_MODE=cluster \
python -m sglang.launch_server \
  --model <hf-id> \
  --schedule-policy lpm \
  --enable-cache-report \
  --mem-fraction-static 0.88 \
  --host 127.0.0.1 --port 30000
```

**Send a request** (vanilla OpenAI-style):

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<hf-id>","messages":[{"role":"user","content":"Hello"}]}'
```

**Drop the dynamic-lane fairness controller** (use a fixed 70/30 split):

```bash
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \
  PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_BIGLANE_SHARE=0.7 \
  python -m sglang.launch_server --model <hf-id> ...
```

**Eviction-only ablation** (stock LPM scheduler + queue-aware eviction):

```bash
PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=cluster \
  python -m sglang.launch_server --model <hf-id> --schedule-policy lpm ...
```

---

## 2. Online + vLLM

**Install once** (Rust toolchain + conda env + vllm 0.19.1 + peek):

```bash
bash scripts/install_peek_vllm.sh
```

**Launch the server**. vLLM v1 spawns its `EngineCore` in a child
process via `multiprocessing.spawn`, so the patch must reach the child;
the `sitecustomize.py` shim handles that:

```bash
export PYTHONPATH="$(pwd)/scripts/peek_sitecustomize:${PYTHONPATH:-}"

PEEK_ONLINE_SCHEDULER=1 \
PEEK_ONLINE_EVICTION=1 \
PEEK_ONLINE_CLPM=1 \
PEEK_ONLINE_CLPM_GROUP_MAJOR=1 \
PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 \
PEEK_ONLINE_EVICTION_MODE=cluster \
python -m vllm.entrypoints.openai.api_server \
  --model <hf-id> \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port 8000
```

> **Note.** `--enable-prefix-caching` is required: peek's hooks
> (`find_longest_cache_hit` for cLPM, demand index for eviction) all
> read from vLLM's prefix cache. With APC off, `PEEK_ONLINE_*` flags
> install but produce no useful change.

**Send a request:**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<hf-id>","messages":[{"role":"user","content":"Hello"}]}'
```

**Multi-GPU TP=2** (just add `--tensor-parallel-size 2`):

```bash
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_CLPM=1 PEEK_ONLINE_EVICTION=1 \
  python -m vllm.entrypoints.openai.api_server \
    --model <hf-id> --tensor-parallel-size 2 --enable-prefix-caching ...
```

---

## 3. Offline + SGLang

**Install once** (apply the source-level patches to sglang's installed files):

```bash
bash scripts/install_peek_sglang.sh
APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_sglang.sh
# OR: python -m peek.offline.install sglang
```

The patches add a new `--radix-eviction-policy queue-aware` choice and
hook `_compute_prefix_matches()` so that when the queue-aware policy is
on, the offline scheduler reorders the waiting queue via DFS over the
pending prefix trie.

**Launch the server:**

```bash
python -m sglang.launch_server \
  --model <hf-id> \
  --radix-eviction-policy queue-aware \
  --enable-cache-report \
  --mem-fraction-static 0.88 \
  --host 127.0.0.1 --port 30000
```

**Send a request** — same `curl` as in example 1.

**Optional client-side reorder** (instead of server-side, e.g. when you
can't or don't want to patch the server). Set the offline env-var and
use the Python API:

```python
from peek.offline import PeekDispatcher, PeekConfig

dispatcher = PeekDispatcher(PeekConfig(max_concurrent=64))
# ... call dispatcher.submit(prompt, ...) for each request
```

```bash
PEEK_OFFLINE_ENABLE=1 python my_client.py
```

---

## 4. Offline + vLLM

**Install once** (apply the source-level patches to vllm's installed files):

```bash
bash scripts/install_peek_vllm.sh
APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_vllm.sh
# OR: python -m peek.offline.install vllm
```

The patches add `queue_ref_count` to vLLM's `KVCacheBlock`, a queue-aware
victim pick in `BlockPool.get_new_blocks`, and a `_update_queue_refs()`
hook in `Scheduler.schedule()`.

**Launch the server:**

```bash
PEEK_OFFLINE_SERVER_REORDER=1 \
python -m vllm.entrypoints.openai.api_server \
  --model <hf-id> \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port 8000
```

vLLM has no `--radix-eviction-policy` CLI knob, so the queue-aware
eviction is gated by `PEEK_OFFLINE_SERVER_REORDER=1` instead.

**Send a request** — same `curl` as in example 2.

---

## Reverting

**Online:** unset all `PEEK_ONLINE_*` env vars and relaunch — the patch
hook becomes a no-op when no flag is set.

**Offline:** revert the source-level patches:

```bash
python -m peek.offline.install sglang --revert    # sglang
python -m peek.offline.install vllm   --revert    # vllm
```

The `.peek_bak` backups created at install time are used to restore the
original engine source files.

---

## Mixing modes

Online and offline modes are **mutually exclusive on the same engine
process**. Online uses runtime monkey-patching of the live engine
process; offline modifies the engine's source files on disk. They patch
overlapping methods (`Scheduler.schedule`, `BlockPool.get_new_blocks`)
and produce undefined behavior if both are active simultaneously.

If you want to A/B online vs offline, install offline patches once and
**either** set `PEEK_ONLINE_*` flags (online wins, offline patches
become near-no-ops because the online hook overrides) **or** set
`PEEK_OFFLINE_SERVER_REORDER=1` / `--radix-eviction-policy queue-aware`
(offline wins). Run separate processes if you want a clean A/B.
