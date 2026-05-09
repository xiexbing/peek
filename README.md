# PEEK

**Predictive, queue-informed KV cache management for LLM serving.**

PEEK reasons about the prefix-sharing structure of *waiting* requests
and uses it to drive cluster-aware scheduling and queue-aware eviction
in existing LLM serving engines (SGLang, vLLM). PEEK installs by
monkey-patching at import time -- no fork of either upstream engine is
required, and an unmodified engine binary can be flipped between
vanilla and PEEK-enabled with environment flags.

PEEK ships **two operating modes**:

- **`peek.online`** *(this paper)* -- streaming-arrival serving. Rust-backed
  pending radix tree + Cluster-LPM (cLPM) scheduler + queue-aware eviction.
  Targets continuous arrivals under tight latency budgets.
- **`peek.offline`** -- batch-style serving. Python prefix trie + DFS reorder
  + queue-aware eviction (`queue-aware` cache policy). Targets known-batch
  and offline-throughput regimes.

Both modes share the same companion paper (online mode is the focus of
the submitted paper; offline mode is documented in Appendix~A).

## What PEEK does

LLM serving engines optimize against the *current* state of their KV
cache, but ignore the prefix-sharing structure of requests *waiting* in
the CPU queue. PEEK closes that gap with three composable mechanisms:

1. **Pending-tree dual-walk for prefix matching** *(online)* -- one paired
   traversal of PEEK's queue tree against the engine's prefix cache yields
   every waiting request's longest-prefix-match in $O(C \cdot D)$ instead
   of stock LPM's $O(N \cdot D)$.
2. **Cluster-aware admission and eviction** *(both modes)* -- admit the
   pioneer of a large queued cluster ahead of unrelated singletons so its
   siblings inherit the freshly cached prefix; a co-designed eviction hook
   applies the same signal in reverse, protecting blocks ancestral to
   queued demand.
3. **Multi-lane stride scheduler** *(online)* -- interleave a cache-locality
   lane (cLPM) with an arrival-time fairness lane to bound starvation
   under streaming arrivals; an EMA-smoothed dynamic-lane controller
   adapts the lane share to queue composition.

A `has_sharing` guard short-circuits cycles where the queue contains no
prefix-sharing structure, so PEEK adds no measurable overhead on
non-sharing workloads.

## Quick start

PEEK is a Rust extension exposed to Python via [PyO3](https://pyo3.rs)
and built with [maturin](https://github.com/PyO3/maturin). Online mode
requires the Rust core; offline mode is pure Python.

```bash
# 1. Build PEEK into the active Python env
pip install maturin
maturin develop --release

# 2. Verify
pytest tests/online/                  # online suite (Rust core required)
pytest tests/offline/                 # offline suite (pure Python)
cargo test --release                  # Rust unit tests
```

For a full single-command bootstrap on a fresh box (Rust toolchain +
conda env + engine + PEEK):

```bash
bash scripts/install_peek_sglang.sh                              # peek + sglang 0.5.9
APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_sglang.sh      # also apply offline source-level patches
bash scripts/install_peek_vllm.sh                                # peek + vllm 0.19.1
APPLY_OFFLINE_PATCHES=1 bash scripts/install_peek_vllm.sh        # also apply offline source-level patches
```

> **Note.** SGLang and vLLM pin incompatible torch versions and cannot
> share a single Python environment. Use **separate** envs if you need
> both engines on the same machine.

See [`EXAMPLES.md`](EXAMPLES.md) for copy-paste launch commands per
(mode x engine) combination plus ablation recipes (eviction-only,
fixed lane share, multi-GPU TP=2, client-side dispatch).

## Smoke test (~5 min, no GPU required)

A CPU-only sanity check that exercises the Rust core, cLPM sort, and
no-sharing guard:

```bash
maturin develop --release
pytest -m "not gpu" tests/online/ tests/offline/
cargo test --release
```

All tests should pass on a laptop with no LLM engine installed.

## Online mode -- streaming arrivals (paper focus)

`peek.online` installs only when the engine imports
`peek.online.engines.<name>.patch_hook` *and* at least one `PEEK_ONLINE_*`
environment flag is set. With no flags the shim is a no-op and the
engine runs vanilla.

### One-shot preset

Setting `PEEK_PRESET=peek-online` enables the paper's primary configuration
(scheduler + cLPM + group-major + dynamic lane + cluster-mode eviction)
without having to set each `PEEK_ONLINE_*` flag individually:

```bash
PEEK_PRESET=peek-online python -m sglang.launch_server --schedule-policy lpm ...
```

Explicit per-flag env vars still win, so ablations work: e.g.,
`PEEK_PRESET=peek-online PEEK_ONLINE_CLPM_DYNAMIC_LANE=0` keeps the rest of
the preset and turns dynamic lane off. See `peek.preset` for the full
mapping.

### SGLang

SGLang's scheduler runs in the launching Python process, so peek's
`patch_hook` must be imported *before* `sglang.launch_server`. The
provided `sitecustomize.py` shim does that automatically when placed on
`PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/scripts/peek_sitecustomize:${PYTHONPATH:-}"
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \
  python -m sglang.launch_server --schedule-policy lpm ...
```

> **Note.** Without `peek_sitecustomize` on `PYTHONPATH` the
> `PEEK_ONLINE_*` flags will be visible to the process but the
> `patch_hook` is never imported, so the engine runs vanilla sglang.

### vLLM

vLLM v1 spawns its `EngineCore` in a child process via
`multiprocessing.spawn`, so the patch must reach the child. Add the
provided `sitecustomize.py` to `PYTHONPATH` so the import fires
automatically in every interpreter:

```bash
export PYTHONPATH="$(pwd)/scripts/peek_sitecustomize:${PYTHONPATH:-}"
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \
  python -m vllm.entrypoints.openai.api_server \
    --enable-prefix-caching --gpu-memory-utilization 0.9 ...
```

### Online activation flags

| flag                                  | scope                                                    |
| ------------------------------------- | -------------------------------------------------------- |
| `PEEK_PRESET=peek-online`             | one-shot: enable the paper's primary configuration       |
| `PEEK_ONLINE_SCHEDULER=1`             | install scheduler hooks (cLPM-eligible)                  |
| `PEEK_ONLINE_EVICTION=1`              | install queue-aware eviction                             |
| `PEEK_ONLINE_CLPM=1`                  | enable multi-lane (cache-locality + fairness) scheduler  |
| `PEEK_ONLINE_CLPM_GROUP_MAJOR=1`      | group-major admission (sibling-batched, "GM")            |
| `PEEK_ONLINE_CLPM_DYNAMIC_LANE=1`     | dynamic lane-share controller ("DL", anti-starvation)    |
| `PEEK_ONLINE_CLPM_BIGLANE_SHARE=0.7`  | static cache-locality lane share (DYNAMIC\_LANE off)     |
| `PEEK_ONLINE_EVICTION_MODE=cluster`   | one of `plain` / `cluster` / `recency` / `decay`         |
| `PEEK_ONLINE_PHASE_TRACKING=1`        | per-rid arrival/admission/completion dump (debug)        |

The paper's primary configuration is **cLPM+GM+DL+PE**, which corresponds
to:

```
PEEK_ONLINE_SCHEDULER=1
PEEK_ONLINE_EVICTION=1
PEEK_ONLINE_CLPM=1
PEEK_ONLINE_CLPM_GROUP_MAJOR=1
PEEK_ONLINE_CLPM_DYNAMIC_LANE=1
PEEK_ONLINE_EVICTION_MODE=cluster
```

## Offline mode -- batched / scheduler-side

`peek.offline` patches the engine's source files directly (under
`sglang_patches/` and `vllm_patches/`), adding a `queue-aware` cache
eviction policy that activates the offline scheduler. The patches are
applied automatically the first time `peek.offline` is imported.

```python
import peek.offline   # auto-patches sglang and vllm if installed
```

Or apply patches explicitly:

```bash
python -m peek.offline.install            # patch all detected backends
python -m peek.offline.install sglang     # patch sglang only
python -m peek.offline.install vllm       # patch vllm only
```

Launch with the new eviction policy:

```bash
python -m sglang.launch_server --radix-eviction-policy queue-aware ...
```

| flag                                  | scope                                               |
| ------------------------------------- | --------------------------------------------------- |
| `--radix-eviction-policy queue-aware` | (CLI) enable offline reorder + queue-aware eviction |
| `PEEK_OFFLINE_ENABLE=1`               | enable client-side `PeekEngine` reorder             |
| `PEEK_OFFLINE_SERVER_REORDER=1`       | enable server-side scheduling hook                  |
| `PEEK_QUEUE_AWARE=1`                  | enable queue-aware eviction (auto-set by patch)     |

`PEEK_PRESET=peek-offline` is the one-shot equivalent: sets the three
`PEEK_OFFLINE_*` / `PEEK_QUEUE_AWARE` flags above to their default-on
values. Explicit env vars still override the preset.

## Reproducing the paper

The paper evaluates PEEK on five workloads (W1-W5) on SGLang 0.5.9 and
vLLM 0.19.1. All datasets and models are publicly available; below is
the per-workload mapping and the commands to launch each.

### Workloads

| W   | Hardware                       | Model                              | Dataset                                  |
| --- | ------------------------------ | ---------------------------------- | ---------------------------------------- |
| W1  | 1xH100 80GB                    | Qwen2.5-32B-Instruct               | LooGLE long-document, 100 group prompts (~1024 tok), Zipf-α=1.0 |
| W2  | 1xH100 80GB                    | Qwen2.5-32B-Instruct               | LooGLE long-document, 40 group prompts (~8192 tok), Zipf-α=1.0  |
| W3  | 2-4xH100 80GB (TP=2; DP=1/2)   | Llama-3.1-70B-Instruct             | Multi-GPU scaling of W1 (cell C) and W2 (cell B)                |
| W4  | 1xH100 80GB                    | Mistral-Small-24B-Instruct-2501    | Mooncake `conversation_trace` (FAST'25), 4 rounds, 50 ms inter-turn; `agentic_only` and `agentic_shared` (1402-token shared prompt) |
| W5  | 1xH100 80GB                    | Gemma-2-27B-it                     | LMSYS arena chat: `cell_C_long` (512-4096 tok), `cell_C_short` (32-1024 tok) |

### Dataset fetch

| dataset                          | source                                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| LooGLE                           | `huggingface.co/datasets/bigai-nlco/LooGLE`                                             |
| Mooncake `conversation_trace`    | FAST'25 release (`github.com/kvcache-ai/Mooncake`)                                      |
| LMSYS arena chat                 | `huggingface.co/datasets/lmsys/lmsys-chat-1m` (CC BY-NC; research-use only)             |

### Models

All HuggingFace IDs:

```
Qwen/Qwen2.5-32B-Instruct
meta-llama/Llama-3.1-70B-Instruct
mistralai/Mistral-Small-24B-Instruct-2501
google/gemma-2-27b-it
```

### Shared experiment configuration

| component                | value                                                          |
| ------------------------ | -------------------------------------------------------------- |
| Engines                  | SGLang 0.5.9, vLLM 0.19.1                                      |
| Framework                | PyTorch 2.9.1, Python 3.12.13, CUDA graphs **on**              |
| Memory budget            | SGLang `mem_fraction_static=0.88`; vLLM `gpu_memory_utilization=0.9` |
| Seeds                    | 42, 142, 242 (3 reps; reported numbers are seed means; ± = seed std-dev) |
| Warmup                   | First ~10-20 % of requests excluded from metric aggregation    |
| Load levels (per cell)   | *moderate* (sustained queue ≈60-100); *heavy* (≈150-200)       |

### Commands per workload (paper primary configuration)

`<sched>` is the SGLang scheduler choice; PEEK uses `lpm` (PEEK overrides
the LPM tiebreak). `<port>` is `30000` for SGLang, `8000` for vLLM.

**SGLang (W1, W2, W4, W5):**

```bash
export PYTHONPATH="$(pwd)/scripts/peek_sitecustomize:${PYTHONPATH:-}"
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \
PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 \
PEEK_ONLINE_EVICTION_MODE=cluster \
python -m sglang.launch_server \
  --model <hf-id> --schedule-policy lpm \
  --enable-cache-report --mem-fraction-static 0.88 \
  --host 127.0.0.1 --port 30000
```

**vLLM (W1, W2, W4, W5):**

```bash
export PYTHONPATH="$(pwd)/scripts/peek_sitecustomize:${PYTHONPATH:-}"
PEEK_ONLINE_SCHEDULER=1 PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_CLPM=1 \
PEEK_ONLINE_CLPM_GROUP_MAJOR=1 PEEK_ONLINE_CLPM_DYNAMIC_LANE=1 \
PEEK_ONLINE_EVICTION_MODE=cluster \
python -m vllm.entrypoints.openai.api_server \
  --model <hf-id> --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port 8000
```

**W3 multi-GPU:** add `--tensor-parallel-size 2` for TP=2; for DP=2 launch
two replicas (different ports) and a front-end that routes by `request_id`
hash.

### Stock baselines

For each engine, the strongest stock baseline used in the paper:

```bash
# SGLang baseline (LPM+LRU)
python -m sglang.launch_server --model <hf-id> --schedule-policy lpm ...

# vLLM baseline (FCFS+APC+LRU)
python -m vllm.entrypoints.openai.api_server --model <hf-id> --enable-prefix-caching ...
```

Run each (baseline, PEEK) at the same load level and report cache hit /
TTFT / E2E / throughput across the 3 seeds. Per-workload ablation
recipes (cLPM-only, cLPM+GM, cLPM+GM+PE, etc.) follow the same flag
combinations, dropping individual `PEEK_ONLINE_CLPM_*` flags.

## Layout

```
peek/
├── src/                                Shared Rust core (used by online)
│   ├── lib.rs                          PyO3 module entry
│   ├── tree.rs                         Arena-backed radix tree
│   └── pending.rs                      Pending-tree manager
├── python/peek/
│   ├── __init__.py                     Top-level package; re-exports PendingTree
│   ├── _core.pyi                       Stubs for the Rust extension
│   ├── online/
│   │   ├── lpm_integration.py          cLPM Python sort + lane logic
│   │   ├── eviction.py                 Queue-aware eviction strategy
│   │   ├── policy.py
│   │   └── engines/
│   │       ├── sglang/patch_hook.py    Monkey-patches for sglang ≥ 0.5.9
│   │       └── vllm/patch_hook.py      Monkey-patches for vllm v1 (0.19.1)
│   └── offline/
│       ├── install.py                  Patch installer
│       ├── trie.py                     Prefix trie + DFS reorder
│       ├── reorder.py                  PeekDispatcher + PeekConfig
│       ├── scheduler.py                Sched/eviction hooks
│       ├── engine.py                   PeekEngine + CacheStateStore
│       ├── engine_vllm.py              vLLM-specific shadow-cache engine
│       ├── prompt.py                   PromptRequest dataclass
│       ├── sglang_patches/             Source-level sglang patches + applier
│       ├── vllm_patches/               Source-level vllm patches + applier
│       └── benchmarks/                 Workload helpers used by tests
├── scripts/
│   ├── install_peek_sglang.sh          Bootstrap install (PEEK + sglang)
│   ├── install_peek_vllm.sh            Bootstrap install (PEEK + vllm)
│   └── peek_sitecustomize/             Site-customize shim for vllm spawn workers
├── patches/                            Optional bench_serving Mooncake-trace patch
└── tests/
    ├── online/                         Rust-core + cLPM correctness suite
    └── offline/                        Trie + reorder + scheduler hooks
```

## Tests

| marker          | what it needs                                                | how to run                              |
| --------------- | ------------------------------------------------------------ | --------------------------------------- |
| *(unmarked)*    | CPU only -- no engine import                                  | `pytest -m "not gpu and not engine"`    |
| `engine`        | sglang or vllm Python modules importable (CPU OK, no model)  | `pytest -m engine`                      |
| `gpu`           | a CUDA-capable GPU + an actual engine running a real model   | `pytest -m gpu`                         |

```bash
# CPU-only smoke check (default for laptops / CI without a GPU):
pytest -m "not gpu"
pytest tests/online/
pytest tests/offline/
cargo test --release

# Full suite (requires a GPU + sglang/vllm installed):
pytest tests/
```

## Tested versions

| component | version |
| --------- | ------- |
| sglang    | 0.5.9   |
| vllm      | 0.19.1  |
| torch     | 2.9.1   |
| Python    | 3.12    |
| Rust      | stable  |

## Citing PEEK

> *PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving.*
> Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen. 2026.

The paper is included in this repository at [`paper/peek.pdf`](paper/peek.pdf).
A BibTeX entry will be added once the paper is posted to a public archive.
See [`AUTHORS`](AUTHORS) for the author list and [`CITATION.cff`](CITATION.cff)
for a machine-readable citation.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE)
for the full text and [`NOTICE`](NOTICE) for the copyright notice. By
contributing to this repository you agree to license your contribution
under the same terms.
