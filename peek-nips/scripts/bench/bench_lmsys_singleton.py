"""W5 bench client: 100% singleton workload from LMSYS-Chat-1M.

Real-text chat queries with no shared system prompt and no cross-request
prefix sharing. Each request is unique by construction (English-only,
first-turn-only, dedup'd). Used as the no-regression test for peek vs
stock LPM: if peek hurts here, its overhead matters even on no-sharing
workloads.

Reuses dispatch_one / ReqMetric / WorkloadRequest / summarize from
bench_shared_prompts.py for output-format consistency with W3 results.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from typing import List, Optional

import aiohttp

# Reuse the existing bench machinery.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_shared_prompts as bsp


def _load_lmsys_singletons(
    n_target: int,
    seed: int,
    model_name: str,
    prompt_len_min: int = 32,
    prompt_len_max: int = 4096,
    dedupe_prefix_chars: int = 100,
    pool_size: int = 50000,
) -> List[str]:
    """Load N unique English first-turn user messages from LMSYS-Chat-1M.

    Filtering pipeline:
      1. language == 'English' (tokenizer consistency)
      2. first turn (conversation[0]) where role == 'user'
      3. prompt token length in [prompt_len_min, prompt_len_max] (Qwen/Gemma tokenizer)
      4. dedupe by exact-match on first dedupe_prefix_chars characters
      5. random sample n_target from the survivors

    pool_size caps how many rows we stream-scan before stopping (bench
    needs a few thousand survivors; full 1M scan is unnecessary).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[bench-lmsys] HF datasets unavailable ({e}); aborting.", file=sys.stderr)
        sys.exit(1)
    try:
        tokenizer = bsp._get_tokenizer(model_name)
    except Exception as e:
        print(f"[bench-lmsys] tokenizer load failed ({e}); aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"[bench-lmsys] streaming lmsys/lmsys-chat-1m (English-first-turn, target={n_target})...", file=sys.stderr)
    rows = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    seen_prefix = set()
    survivors: List[str] = []
    scanned = 0
    for row in rows:
        scanned += 1
        if scanned > pool_size:
            break
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text:
            continue
        # length filter (in tokens)
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        if len(ids) < prompt_len_min or len(ids) > prompt_len_max:
            continue
        # dedupe
        key = text[:dedupe_prefix_chars].lower().strip()
        if key in seen_prefix:
            continue
        seen_prefix.add(key)
        survivors.append(text)
        if len(survivors) >= n_target * 3:
            # 3x oversample so seed-driven sampling has variance
            break

    print(f"[bench-lmsys] scanned={scanned} survivors={len(survivors)}", file=sys.stderr)
    if len(survivors) < n_target:
        print(f"[bench-lmsys] WARNING: only {len(survivors)} survivors for n_target={n_target}; "
              f"increase pool_size or relax length filter", file=sys.stderr)
    rng = random.Random(seed ^ 0x1357A1B5)
    rng.shuffle(survivors)
    return survivors[:n_target]


def generate_workload_singleton(
    n: int,
    rate_req_per_s: float,
    seed: int,
    model_name: str,
    prompt_len_min: int = 32,
    prompt_len_max: int = 4096,
    max_tokens_min: int = 64,
    max_tokens_max: int = 256,
) -> List[bsp.WorkloadRequest]:
    """One unique LMSYS prompt per request, random per-req max_tokens.

    No shared system prompt. group_id=-1 marks all reqs as singleton.
    """
    prompts = _load_lmsys_singletons(
        n_target=n, seed=seed, model_name=model_name,
        prompt_len_min=prompt_len_min, prompt_len_max=prompt_len_max,
    )
    if len(prompts) < n:
        if not prompts:
            print(f"[bench-lmsys] FATAL: 0 prompts loaded; aborting", file=sys.stderr)
            sys.exit(1)
        print(f"[bench-lmsys] only {len(prompts)} prompts available for N={n}; "
              f"cycling to fill (will introduce some sharing — verify pool_size)", file=sys.stderr)
        base = list(prompts)
        i = 0
        while len(prompts) < n:
            prompts.append(base[i % len(base)])
            i += 1
    rng = random.Random(seed)
    requests: List[bsp.WorkloadRequest] = []
    t = 0.0
    for i in range(n):
        if rate_req_per_s > 0:
            t += rng.expovariate(rate_req_per_s)
        max_toks = rng.randint(max_tokens_min, max_tokens_max)
        requests.append(
            bsp.WorkloadRequest(
                req_id=i,
                group_id=-1,
                system_prompt="",  # no shared system prompt for singletons
                user_message=prompts[i],
                max_tokens=max_toks,
                target_tokens=max_toks,
                arrival_time_s=t if rate_req_per_s > 0 else 0.0,
            )
        )
    return requests


async def run_sweep(args) -> dict:
    workload = generate_workload_singleton(
        n=args.n,
        rate_req_per_s=args.rate,
        seed=args.seed,
        model_name=args.model,
        prompt_len_min=args.prompt_len_min,
        prompt_len_max=args.prompt_len_max,
        max_tokens_min=args.max_tokens_min,
        max_tokens_max=args.max_tokens_max,
    )
    print(
        f"[bench-lmsys] label={args.label} n={args.n} rate={args.rate} "
        f"prompt_len=[{args.prompt_len_min},{args.prompt_len_max}] "
        f"max_tokens=[{args.max_tokens_min},{args.max_tokens_max}] "
        f"concurrency_cap={args.concurrency} warmup={args.warmup_reqs}"
    )
    metrics: List[bsp.ReqMetric] = []
    in_flight = {"cur": 0, "max": 0}
    sem = (
        asyncio.Semaphore(args.concurrency)
        if args.concurrency > 0
        else None
    )
    connector = aiohttp.TCPConnector(limit=max(1024, args.concurrency * 2 + 16))
    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as client:
        tasks = []
        for req in workload:
            async def _fire(req=req):
                if args.rate > 0 and req.arrival_time_s > 0:
                    elapsed = time.perf_counter() - wall_start
                    if req.arrival_time_s > elapsed:
                        await asyncio.sleep(req.arrival_time_s - elapsed)
                if sem is not None:
                    async with sem:
                        m = await bsp.dispatch_one(
                            req, args.endpoint, args.model, client, in_flight
                        )
                else:
                    m = await bsp.dispatch_one(
                        req, args.endpoint, args.model, client, in_flight
                    )
                metrics.append(m)
            tasks.append(asyncio.create_task(_fire()))
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall_start
    return bsp.summarize(metrics, wall, args, in_flight_max=in_flight["max"])


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--rate", type=float, default=3.0)
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-reqs", type=int, default=100)
    p.add_argument("--ttft-slo-ms", type=float, default=2000)
    p.add_argument("--tpot-slo-ms", type=float, default=100)
    p.add_argument("--e2e-slo-ms", type=float, default=60000)
    p.add_argument("--label", default="lmsys_singleton")
    p.add_argument("--output", required=True)
    p.add_argument("--save-per-request", action="store_true")
    p.add_argument("--prompt-len-min", type=int, default=32)
    p.add_argument("--prompt-len-max", type=int, default=4096)
    p.add_argument("--max-tokens-min", type=int, default=64)
    p.add_argument("--max-tokens-max", type=int, default=256)
    # Args that bsp.summarize() expects but are no-ops here:
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--prefix-tokens", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--decode-mix", default="")
    p.add_argument("--hide-max-tokens", action="store_true")
    p.add_argument("--dataset", default="lmsys")
    p.add_argument("--distribution", default="uniform")
    p.add_argument("--zipf-alpha", type=float, default=1.0)
    p.add_argument("--phase-dump-path", default="/tmp/peek_phases_{pid}.json")
    return p.parse_args()


def main() -> None:
    args = _parse()
    result = asyncio.run(run_sweep(args))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[bench-lmsys] wrote {args.output}")


if __name__ == "__main__":
    main()
