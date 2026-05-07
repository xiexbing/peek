# Reproducing W3 (paper §4.3, Tables 15 & 16)

End-to-end recipe for the W3 multi-GPU 70B experiment. Covers both engines
(SGLang 0.5.9, vLLM 0.19.1) at both topologies (DP=1 on 2xH100, DP=2 on
4xH100) for both cells (B=long-doc RAG, C=shared-prompt chat).

When done, `compare_to_paper.py` prints a side-by-side table of every
observed metric vs the paper value, with a green "OK" / red "!!" tag.

## 0. Hardware

- **DP=1 only**: 2 x H100 80GB
- **DP=1 + DP=2**: 4 x H100 80GB

If you have only 2 GPUs, skip the DP=2 drivers -- the DP=1 drivers still
populate Tables 15/16 columns 1-2 (SGLang) and 1-2 (vLLM).

## 1. Install (two separate Python envs)

SGLang and vLLM pin incompatible torch versions and **cannot** share one
environment.

```bash
# SGLang env
python3 -m venv /path/to/envs/sglang
source /path/to/envs/sglang/bin/activate
PY=$(which python) bash scripts/install_peek_sglang.sh
deactivate

# vLLM env
python3 -m venv /path/to/envs/vllm
source /path/to/envs/vllm/bin/activate
PY=$(which python) bash scripts/install_peek_vllm.sh
deactivate
```

Each install script:
- Installs Rust (rustup) if missing
- Installs maturin and builds the PEEK Rust core via `maturin develop --release`
- Installs the engine (sglang 0.5.9 or vllm 0.19.1) and verifies imports
- Installs `ninja` (sglang's fa3 backend and vLLM's fused kernels both JIT-compile via ninja at first inference)

> If `uv pip install sglang[all]==0.5.9` times out on a CUDA wheel, retry with
> `UV_HTTP_TIMEOUT=600 bash scripts/install_peek_sglang.sh` -- the default
> 30 s is too short for 600 MiB+ CUDA archives on slow links.

For DP=2 SGLang you also need the prefix-aware router:

```bash
source /path/to/envs/sglang/bin/activate
pip install sglang-router
```

## 2. Download the model

`meta-llama/Llama-3.1-70B-Instruct` is gated on Hugging Face. Accept the
license terms in your HF account, then:

```bash
export HF_HOME=/path/to/hf_cache              # ~150 GB free required
export HF_HUB_ENABLE_HF_TRANSFER=1            # 5-10x faster downloads
huggingface-cli login                          # paste token
hf download meta-llama/Llama-3.1-70B-Instruct --exclude "original/*"
```

## 3. (Optional) Pre-warm the model into page cache on slow filesystems

If your model lives on a network filesystem (NFS, MooseFS, S3-FUSE), the
TP=2 sharded load can take 15-30 min per server start. Reading the safetensors
into the OS page cache once cuts subsequent loads dramatically:

```bash
cat $HF_HOME/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/*/model-*.safetensors > /dev/null
```

Skip this on local SSD.

## 4. Run the four W3 drivers

All four drivers default to seed sweep `42 142 242` (paper config). Override
with `SEEDS=42` for a single-seed sanity check.

### 4a. SGLang DP=1 (2xH100, single TP=2 replica)

```bash
source /path/to/envs/sglang/bin/activate
SEEDS="42 142 242" bash benchmarks/w3/run_w3_sglang.sh
# results -> benchmarks/w3/results_sglang/
```

Runs cells C and B back-to-back for both policies (`lpm_lru`, `clpm_gm_dl_pe`).

### 4b. SGLang DP=2 (4xH100, two TP=2 replicas behind sglang-router)

```bash
# cell C (chat-like)
SEEDS="42 142 242" CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
  CONCURRENCY=180 DECODE_MIX="" \
  bash benchmarks/w3/run_w3_sglang_dp2.sh

# cell B (RAG-like)
SEEDS="42 142 242" CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
  CONCURRENCY=180 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" \
  bash benchmarks/w3/run_w3_sglang_dp2.sh
# results -> benchmarks/w3/results_sglang_dp2/
```

### 4c. vLLM DP=1 (2xH100)

```bash
deactivate                        # leave sglang env
source /path/to/envs/vllm/bin/activate
SEEDS="42 142 242" bash benchmarks/w3/run_w3_vllm.sh
# results -> benchmarks/w3/results_vllm/
```

### 4d. vLLM DP=2 (4xH100, two TP=2 vLLM workers behind sglang-router)

Note: the vLLM DP=2 driver uses the sglang-router as the front-end, so
`sglang-router` must also be installed in the **vLLM** env:

```bash
pip install sglang-router

# cell C
SEEDS="42 142 242" CELL=C NGROUPS=88 PREFIX_TOKENS=1500 N=1000 WARMUP=200 \
  CONCURRENCY=360 DECODE_MIX="" \
  bash benchmarks/w3/run_w3_vllm_dp2.sh

# cell B
SEEDS="42 142 242" CELL=B NGROUPS=14 PREFIX_TOKENS=4096 N=500 WARMUP=100 \
  CONCURRENCY=360 DECODE_MIX="10:128,25:512,30:1024,25:2048,10:4096" \
  bash benchmarks/w3/run_w3_vllm_dp2.sh
# results -> benchmarks/w3/results_vllm_dp2/
```

## 5. Verify against paper Tables 15 & 16

```bash
python3 benchmarks/w3/compare_to_paper.py
# or for a single-seed spot-check:
SEED=42 python3 benchmarks/w3/compare_to_paper.py
```

Output is one row per (engine x dp x cell x policy x metric) with the
paper value, your observed value, and a relative delta. Default tolerance
is ±20 % (single-seed runs wobble vs the paper's 3-seed means); override
with `TOL_PCT=10`.

Exit code 0 if every metric is within tolerance, 1 otherwise.

## 6. Wall-clock budget

Per (engine x topology) combination, single seed:

| Stage | Approx. time |
|---|---|
| Model load (2xH100 first time, slow FS) | 15-25 min x 2 servers |
| Model load (subsequent, page cache warm) | 5-10 min x 2 servers |
| Cell C bench (1000 reqs) | 3-10 min |
| Cell B bench (500 reqs) | 7-20 min |
| **Total per (engine x dp), 1 seed** | **45-90 min** |
| **Total all 4 x 1 seed** | **3-6 hr** |
| **Total all 4 x 3 seeds (paper config)** | **9-18 hr** |

The dominant cost is the 70B model load. Pre-warming the page cache (step 3)
is the single biggest speedup on network-FS hosts.

## 7. Common failure modes

| Symptom | Fix |
|---|---|
| `FileNotFoundError: 'ninja'` mid-bench | `pip install ninja` in the active env |
| `server failed to start in 3600s` | Pre-warm cache (step 3); check `_server_*.log` for the real error |
| `RuntimeError: CUDA out of memory` on DP=2 | Lower `MEM_FRAC` (sglang) or `GPU_MEM_UTIL` (vLLM); 0.85 / 0.88 are safer |
| Server log shows `cannot import name 'GlmAsrConfig'` | Benign -- sglang ignores it and loads other models normally |
| Benchmark hangs at 0% | Kill all engine procs (`pkill -9 -f sglang.launch_server`), confirm GPUs free with `nvidia-smi`, then re-run |
