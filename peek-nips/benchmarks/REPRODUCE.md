# Reproducing PEEK W1-W5

End-to-end recipe for the five workloads reported in the paper. Read
top-to-bottom on a fresh H100 box and you should reach numbers that
match the paper's appendix tables within the documented tolerance.

> Per-workload context (cell parameters, ablation policies, predictions)
> lives in each `wN/README.md`. This file is the cross-cutting recipe.
> The W3 deep-dive (`w3/REPRODUCE.md`) has the most detailed model-load
> and timeout guidance; refer to it for any 70B-specific question.

---

## 0. Hardware checklist

| Workload   | Hardware required             | Notes                                       |
| ---------- | ----------------------------- | ------------------------------------------- |
| W1, W2     | 1 x H100 80GB                 | Qwen2.5-32B-Instruct, bf16                  |
| W3 (DP=1)  | 2 x H100 80GB (TP=2)          | Llama-3.1-70B-Instruct, gated -- see step 2 |
| W3 (DP=2)  | 4 x H100 80GB (TP=2, 2 reps)  | Skip if you only have 2 GPUs                |
| W4         | 1 x H100 80GB                 | Mistral-Small-24B-Instruct-2501             |
| W5         | 1 x H100 80GB                 | Gemma-2-27b-it (gated; accept terms)        |

OS: any recent Linux with CUDA 12.x and a modern NVIDIA driver. The
install scripts `apt-get install libnuma-dev` on Debian/Ubuntu (skip on
non-apt distros and install your distro equivalent manually -- the
script's libnuma probe will warn).

## 1. Install (two separate Python envs, one per engine)

SGLang and vLLM pin **incompatible** torch versions; they cannot share a
single Python env. Reviewers who only need to verify one engine can
build only that env.

```bash
# SGLang env (used by W1, W2, W3, W4, W5 SGLang side)
python3 -m venv /path/to/envs/sglang
source /path/to/envs/sglang/bin/activate
PY=$(which python) bash scripts/install_peek_sglang.sh
deactivate

# vLLM env (used by W1, W2, W3, W4, W5 vLLM side)
python3 -m venv /path/to/envs/vllm
source /path/to/envs/vllm/bin/activate
PY=$(which python) bash scripts/install_peek_vllm.sh
deactivate
```

Each install script:

- Installs Rust (rustup) if missing
- Installs maturin and builds the PEEK Rust core via `maturin develop --release`
- Installs the engine (`sglang[all]==0.5.9` or `vllm==0.19.1`) and verifies imports
- Installs `ninja` (both engines JIT-compile fused kernels at first inference)
- Installs bench dependencies (`aiohttp`, `transformers`, `datasets`)
- (SGLang only) Applies `patches/sglang/bench_serving.patch` so the W4
  driver's `PEEK_AGENT_INTER_TURN_*` and `PEEK_SHARED_SYSTEM_PROMPT_PATH`
  env vars take effect. Skip with `SKIP_BENCH_SERVING_PATCH=1` only if
  you don't intend to run W4.

For the **W3 DP=2 routing layer**, also install the prefix-aware router
in **both** envs that will host DP=2 (the vLLM driver also fronts with
the SGLang router):

```bash
source /path/to/envs/sglang/bin/activate ; pip install sglang-router
source /path/to/envs/vllm/bin/activate   ; pip install sglang-router
```

> If `uv pip install sglang[all]==0.5.9` times out on a CUDA wheel,
> retry with `UV_HTTP_TIMEOUT=600 bash scripts/install_peek_sglang.sh`.

## 2. Models (HuggingFace)

```bash
export HF_HOME=/path/to/hf_cache              # ~250 GB free required for all 4 models
export HF_HUB_ENABLE_HF_TRANSFER=1            # 5-10x faster downloads
huggingface-cli login                          # required for gated models
```

Two of the four models are **gated** -- accept the license terms in your
HuggingFace account first, otherwise the download will 401.

| Model                                       | Workloads | Gated? | Approx. weight size |
| ------------------------------------------- | --------- | ------ | ------------------- |
| `Qwen/Qwen2.5-32B-Instruct`                 | W1, W2    | No     | ~65 GB              |
| `meta-llama/Llama-3.1-70B-Instruct`         | W3        | **Yes** | ~140 GB             |
| `mistralai/Mistral-Small-24B-Instruct-2501` | W4        | No     | ~48 GB              |
| `google/gemma-2-27b-it`                     | W5        | **Yes** | ~54 GB              |

```bash
hf download Qwen/Qwen2.5-32B-Instruct
hf download meta-llama/Llama-3.1-70B-Instruct --exclude "original/*"
hf download mistralai/Mistral-Small-24B-Instruct-2501
hf download google/gemma-2-27b-it
```

## 3. Datasets

| Dataset                          | Source                                                                | Used by | Auto-fetched? |
| -------------------------------- | --------------------------------------------------------------------- | ------- | ------------- |
| LooGLE                           | `bigainlco/LooGLE` on HuggingFace                                     | W1, W2, W3 | yes (via `datasets`) |
| Mooncake `conversation_trace`    | `kvcache-ai/Mooncake` (FAST'25 release) -- bundled in `w4/data/`      | W4         | bundled       |
| LMSYS arena chat                 | `lmsys/lmsys-chat-1m` on HuggingFace (CC BY-NC; requires HF_TOKEN)    | W5         | yes (via `datasets`) |

The W4 trace and the optional shared-system prompt are bundled in-tree
at `benchmarks/w4/data/conversation_trace_le6k.jsonl` and
`benchmarks/w4/data/shared_system_prompt.txt` (see
`benchmarks/w4/data/README.md` for provenance and re-fetch
instructions).

LMSYS-Chat-1M is gated on HuggingFace; accept the dataset's terms and
ensure your `HF_TOKEN` is exported before the first W5 run.

## 4. Smoke test (under 5 minutes, no full benchmark)

Sanity-check that the install actually works before committing 50+
GPU-hours:

```bash
# CPU-only -- exercises the Rust core and policy logic, no engine needed
maturin develop --release
pytest -m "not gpu" tests/online/ tests/offline/
cargo test --release
```

All tests should pass on a laptop. If the Rust extension fails to build
or any test fails here, fix that before running benchmarks.

## 5. Run the workloads

Each workload's `run_w*_<engine>.sh` driver:

- Defaults to **3 seeds** (42, 142, 242) -- override with `SEEDS="42"` for a single-seed sanity run
- Defaults to **all configured cells** -- override with `CELLS="..."`
- Defaults to **all configured policies** -- override with `POLICIES="..."`
- Skips runs whose output JSON already exists (re-runs are resume-safe)
- Logs server stdout/err alongside the JSON, with a `_run_<policy>.log`
  per bench
- Mean SGLang `mem_fraction_static=0.88`; mean vLLM `gpu_memory_utilization=0.9`

### W1 -- shared-prompt chat (paper §4.1, Tables 10, 11, 12)

1 x H100 80GB. ~35 GPU-hours for the full 156-run matrix; ~5 GPU-hours
for cell C alone (the primary cell).

```bash
source /path/to/envs/sglang/bin/activate
SEEDS="42 142 242" bash benchmarks/w1/run_w1_sglang.sh
# results -> benchmarks/w1/results/

deactivate
source /path/to/envs/vllm/bin/activate
SEEDS="42 142 242" bash benchmarks/w1/run_w1_vllm.sh
# results -> benchmarks/w1/results_vllm/
```

### W2 -- long-document RAG (paper §4.2, Tables 13, 14)

1 x H100 80GB. Cell B (7x KV pressure) is the single paper-reported
cell; per-seed wallclock is ~25 min on SGLang, ~35 min on vLLM (the
8K-token prefix dominates server-side prefill time).

```bash
source /path/to/envs/sglang/bin/activate
CELLS=B SEEDS="42 142 242" bash benchmarks/w2/run_w2_sglang.sh
# results -> benchmarks/w2/results/

deactivate
source /path/to/envs/vllm/bin/activate
CELLS=B SEEDS="42 142 242" bash benchmarks/w2/run_w2_vllm.sh
# results -> benchmarks/w2/results_vllm/
```

### W3 -- multi-GPU 70B (paper §4.3, Tables 15, 16)

2 x H100 (DP=1) or 4 x H100 (DP=2). 9-18 GPU-hours for the full 4-driver
sweep at 3 seeds. **The complete W3 recipe is in
[`w3/REPRODUCE.md`](w3/REPRODUCE.md)** -- it covers
`huggingface-cli login`, page-cache pre-warm for 70B safetensors,
all four drivers (SGLang DP=1/DP=2, vLLM DP=1/DP=2), and a full
per-stage wallclock budget.

Quick form:

```bash
source /path/to/envs/sglang/bin/activate
SEEDS="42 142 242" bash benchmarks/w3/run_w3_sglang.sh        # DP=1, 2xH100
SEEDS="42 142 242" bash benchmarks/w3/run_w3_sglang_dp2.sh    # DP=2, 4xH100

deactivate
source /path/to/envs/vllm/bin/activate
SEEDS="42 142 242" bash benchmarks/w3/run_w3_vllm.sh          # DP=1, 2xH100
SEEDS="42 142 242" bash benchmarks/w3/run_w3_vllm_dp2.sh      # DP=2, 4xH100
```

### W4 -- agentic Mooncake bursts (paper §4.4, Tables 17, 18)

1 x H100 80GB. ~3 GPU-hours per engine for the 3-policy x 2-cell x
3-seed matrix (per scenario).

The paper reports two scenarios: `agentic_only` (no shared prompt) and
`agentic_shared` (1402-token shared prompt prepended). The drivers run
**`agentic_shared`** by default; for `agentic_only`, set
`SHARED_SYSTEM_PROMPT_PATH=""`.

```bash
source /path/to/envs/sglang/bin/activate
# agentic_shared (default)
SEEDS="42 142 242" bash benchmarks/w4/run_w4_sglang.sh
# agentic_only
SHARED_SYSTEM_PROMPT_PATH="" SEEDS="42 142 242" \
  RESULTS_DIR=$PWD/benchmarks/w4/results_only \
  bash benchmarks/w4/run_w4_sglang.sh
# results -> benchmarks/w4/results/  and  benchmarks/w4/results_only/

deactivate
source /path/to/envs/vllm/bin/activate
# agentic_shared (default; lands in results_vllm/agentic_shared/)
SEEDS="42 142 242" bash benchmarks/w4/run_w4_vllm.sh
# agentic_only -- override RESULTS_DIR so it doesn't overwrite the shared tree
SHARED_SYSTEM_PROMPT_PATH="" SEEDS="42 142 242" \
  RESULTS_DIR=$PWD/benchmarks/w4/results_vllm/agentic_only \
  bash benchmarks/w4/run_w4_vllm.sh
# results -> benchmarks/w4/results_vllm/agentic_shared/  and  ...results_vllm/agentic_only/
```

> **Important:** the SGLang side requires the `bench_serving` patch
> (auto-applied by `install_peek_sglang.sh`). If you skipped it, the
> `PEEK_AGENT_INTER_TURN_*` env vars set by the driver are silent
> no-ops and the run does not match the paper's W4 protocol. Verify
> with `grep -c "PEEK PATCH" $(python -c 'import sglang.bench_serving as m; print(m.__file__)')` -- expect 6.

### W5 -- singleton chat no-regress (paper §4.4, Tables 19, 20)

1 x H100 80GB. ~6 GPU-hours per engine for the 3-policy x 2-cell x
3-seed matrix on SGLang; vLLM is faster (single-rate cells, no
`rate_*` axis).

> The paper's SGLang `C_long` row averages **seed 242 only** (other two
> seeds saturated below the calibrated rate). `compare_to_paper.py`
> honours this restriction by default; pass
> `--no-seed-restriction` to use all present seeds.

```bash
source /path/to/envs/sglang/bin/activate
SEEDS="42 142 242" bash benchmarks/w5/run_w5_sglang.sh
# results -> benchmarks/w5/results/

deactivate
source /path/to/envs/vllm/bin/activate
SEEDS="42 142 242" bash benchmarks/w5/run_w5_vllm.sh
# results -> benchmarks/w5/results_vllm/
```

## 6. Aggregate and verify against the paper

Each workload ships a per-W aggregator + a `compare_to_paper.py`
helper. Run them after the corresponding driver(s) finish.

```bash
# W1
python3 benchmarks/w1/aggregate.py
python3 benchmarks/w1/aggregate.py --results-dir benchmarks/w1/results_vllm \
                                   --baseline fcfs_apc_lru
python3 benchmarks/w1/compare_to_paper.py     # Tables 10, 11

# W2
python3 benchmarks/w2/aggregate.py
python3 benchmarks/w2/aggregate.py --results-dir benchmarks/w2/results_vllm \
                                   --baseline fcfs_apc_lru
python3 benchmarks/w2/compare_to_paper.py     # Tables 13, 14

# W3
python3 benchmarks/w3/compare_to_paper.py     # Tables 15, 16

# W4
python3 benchmarks/w4/aggregate.py
# W4 has TWO scenarios (agentic_only, agentic_shared). The vLLM driver
# splits them by RESULTS_DIR; the SGLang driver does not, so reviewers
# who ran SGLang twice (once per scenario, with separate RESULTS_DIRs
# as in step 5 above) must point compare_to_paper.py at both trees:
RESULTS_BASE_SGLANG_AGENTIC_ONLY=$PWD/benchmarks/w4/results_only \
RESULTS_BASE_SGLANG_AGENTIC_SHARED=$PWD/benchmarks/w4/results \
  python3 benchmarks/w4/compare_to_paper.py     # Tables 17, 18

# W5
python3 benchmarks/w5/aggregate.py
python3 benchmarks/w5/aggregate.py --results-dir benchmarks/w5/results_vllm \
                                   --baseline fcfs_apc_lru
python3 benchmarks/w5/compare_to_paper.py     # Tables 19, 20
```

`compare_to_paper.py` exits 0 if every observed metric is within the
configured tolerance (default ±20% on relative delta -- accommodates
single-seed runs and minor hardware/driver variation). Override with
`TOL_PCT=10` for a tighter check, or `TOL_PCT=30` for a wobblier
single-seed sanity run. Each script prints a per-row "OK" / "!!" tag.

## 7. Wallclock budget (rough, full 3-seed paper config)

| Workload | SGLang  | vLLM    | Combined | Hardware            |
| -------- | ------- | ------- | -------- | ------------------- |
| W1       | ~25 hr  | ~10 hr  | ~35 hr   | 1xH100              |
| W2       | ~5 hr   | ~6 hr   | ~11 hr   | 1xH100              |
| W3       | ~5 hr   | ~5 hr   | 9-18 hr  | 2-4xH100            |
| W4       | ~3 hr   | ~3 hr   | ~6 hr    | 1xH100              |
| W5       | ~6 hr   | ~3 hr   | ~9 hr    | 1xH100              |

Single-seed sanity runs (`SEEDS=42` on every workload) cut these by
~3x; cell C alone for W1 cuts ~70%; agentic_shared alone for W4 cuts
~50%. Use those for a fast first pass, then commit to the full 3-seed
matrix once the smoke run lines up with the paper.

## 8. Common failure modes

| Symptom                                          | Fix                                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: 'ninja'` mid-bench           | `pip install ninja` in the active env                                                                                |
| `server failed to start in 1800s`                | Pre-warm the model into page cache (W3 step 3); check `_server_*.log`                                                |
| `RuntimeError: CUDA out of memory` on DP=2       | Lower `MEM_FRAC` (sglang) or `GPU_MEM_UTIL` (vllm) to 0.85                                                           |
| `cannot import name 'GlmAsrConfig'` in sglang log | Benign -- sglang ignores it and loads other models normally                                                          |
| Benchmark hangs at 0%                             | Kill engine procs (`pkill -9 -f sglang.launch_server`), confirm GPUs free with `nvidia-smi`, re-run                  |
| W4 SGLang has 0% turn-1/turn-2 split, fixed      | The `bench_serving.patch` did not apply; re-run install or apply manually (see `patches/sglang/README.md`)           |
| `compare_to_paper.py` reports MISSING            | Driver did not produce that (cell, rate, policy, seed) -- re-run with `CELLS=...`, `POLICIES=...`, `SEEDS=...` knobs |
| W2 vLLM throughput is 0.000                      | The `--max-model-len` is too small for 8K prefix + decode -- bump it to `MAX_MODEL_LEN=12288` in the driver env      |
| Llama-3.1-70B / Gemma-2 download 401             | Accept the license on HuggingFace, then re-run `huggingface-cli login` and retry the download                        |
| Long benchmarks die overnight from terminal close | Run drivers under `nohup`, `tmux`, or `screen`; the `_run_chain_*.sh` scripts in `w3/` are designed for unattended execution |

## 9. What's not directly reproducible from this kit

- **Production-traffic Pareto-frontier figures** (Fig. 8-13, etc.) --
  derived from the same JSON results as the tables; reviewers who want
  to regenerate the figures can read the per-cell numbers from
  `aggregated.csv` (per-W) and plot directly.
- **Calibration probes** -- the per-cell `r_sat` rates in each driver
  were computed during paper development from `errored=0` and
  `slo_attainment_pct >= 90%` sweeps. The calibrated values are baked
  into the run scripts; the probe scripts themselves are noise for
  reviewers and were dropped.
- **CUDA/PyTorch version drift** -- the paper used PyTorch 2.9.1 with
  CUDA graphs on. Newer PyTorch / SGLang / vLLM versions are likely to
  shift baseline numbers by single-digit % in either direction; pin the
  versions in `scripts/install_peek_*.sh` or expect a wobble.
