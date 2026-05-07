#!/usr/bin/env python3
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

"""Standalone Poisson-arrival client for online serving benchmarks.

Scheduler-agnostic: works with any SGLang/vLLM server regardless of
scheduling policy (fcfs, lpm, peek, dfs-weight, etc.).  Just point it
at a running server and it sends requests with Poisson inter-arrivals.

Usage:
    # Against a running SGLang server on port 30000
    python benchmarks/poisson_client.py \
        --base-url http://localhost:30000 \
        --workload shared_system_prompts \
        --n 200 --arrival-rate 30 --max-concurrent 64

    # Against a running vLLM server on port 8000
    python benchmarks/poisson_client.py \
        --base-url http://localhost:8000 \
        --backend vllm \
        --workload few_shot_mmlu \
        --n 100 --arrival-rate 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure peek is importable
_pkg_dir = Path(__file__).resolve().parent.parent
_repo_root = str(_pkg_dir.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_pkg_str = str(_pkg_dir)
if _pkg_str not in sys.path:
    sys.path.insert(0, _pkg_str)

from peek.offline.benchmarks.workloads import WORKLOAD_GENERATORS, to_request_dicts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    request_id: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    e2e_latency_s: float
    ttft_s: float
    server_e2e_latency_s: float


@dataclass
class RunResult:
    n_requests: int
    metrics: list[RequestMetrics] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def total_cached_tokens(self) -> int:
        return sum(m.cached_tokens for m in self.metrics)

    @property
    def cache_hit_rate(self) -> float:
        total_input = sum(m.input_tokens for m in self.metrics)
        return self.total_cached_tokens / total_input if total_input else 0.0

    @property
    def request_throughput(self) -> float:
        return self.n_requests / self.total_time_s if self.total_time_s else 0.0

    @property
    def output_token_throughput(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return sum(m.output_tokens for m in self.metrics) / self.total_time_s

    @property
    def mean_ttft_ms(self) -> float:
        if not self.metrics:
            return 0.0
        return 1000 * sum(m.ttft_s for m in self.metrics) / len(self.metrics)

    @property
    def p99_ttft_ms(self) -> float:
        if not self.metrics:
            return 0.0
        ttfts = sorted(m.ttft_s for m in self.metrics)
        return 1000 * ttfts[int(len(ttfts) * 0.99)]

    @property
    def mean_e2e_latency_ms(self) -> float:
        if not self.metrics:
            return 0.0
        return 1000 * sum(m.e2e_latency_s for m in self.metrics) / len(self.metrics)

    @property
    def p99_e2e_latency_ms(self) -> float:
        if not self.metrics:
            return 0.0
        lats = sorted(m.e2e_latency_s for m in self.metrics)
        return 1000 * lats[int(len(lats) * 0.99)]

    def summary(self) -> str:
        lines = [
            f"  Requests:          {self.n_requests}",
            f"  Total time:        {self.total_time_s:.2f}s",
            f"  Throughput:        {self.request_throughput:.1f} req/s",
            f"  Output tok/s:      {self.output_token_throughput:.1f}",
            f"  Cache hit rate:    {self.cache_hit_rate*100:.1f}%",
            f"  Cached tokens:     {self.total_cached_tokens}",
            f"  Mean TTFT:         {self.mean_ttft_ms:.1f} ms",
            f"  P99 TTFT:          {self.p99_ttft_ms:.1f} ms",
            f"  Mean E2E:          {self.mean_e2e_latency_ms:.1f} ms",
            f"  P99 E2E:           {self.p99_e2e_latency_ms:.1f} ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Poisson client -- backend-agnostic
# ---------------------------------------------------------------------------

async def _send_one_sglang(
    session,
    sem: asyncio.Semaphore,
    req: dict,
    idx: int,
    results: list[RequestMetrics | None],
    max_new_tokens: int,
    base_url: str,
    stream: bool,
) -> None:
    # Use peek-tagged rid if present (from PeekDispatcher), else original id.
    rid = req.get("rid", req.get("id", f"req-{idx}"))
    payload = {
        "input_ids": req["token_ids"],
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "rid": rid,
        "stream": stream,
    }
    t_start = time.monotonic()
    output_tokens = 0
    cached_tokens = 0
    server_e2e = 0.0
    ttft = 0.0
    first_token_seen = False

    async with sem:
        try:
            url = f"{base_url}/generate"
            if stream:
                async with session.post(url, json=payload) as resp:
                    async for line in resp.content:
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if not first_token_seen:
                            ttft = time.monotonic() - t_start
                            first_token_seen = True
                        meta = data.get("meta_info", {})
                        cached_tokens = meta.get("cached_tokens", 0)
                        output_tokens = meta.get("completion_tokens", output_tokens)
                        server_e2e = meta.get("e2e_request_latency", 0.0)
            else:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    ttft = time.monotonic() - t_start
                    first_token_seen = True
                    meta = data.get("meta_info", {})
                    cached_tokens = meta.get("cached_tokens", 0)
                    output_tokens = meta.get("completion_tokens", 0)
                    server_e2e = meta.get("e2e_request_latency", 0.0)
        except Exception as e:
            print(f"  Warning: request {req.get('id', idx)} failed: {e}")
            return

    e2e = time.monotonic() - t_start
    if not first_token_seen:
        ttft = e2e

    results[idx] = RequestMetrics(
        request_id=req.get("id", f"req-{idx}"),
        input_tokens=len(req["token_ids"]),
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        e2e_latency_s=e2e,
        ttft_s=ttft,
        server_e2e_latency_s=server_e2e,
    )


async def _send_one_vllm(
    session,
    sem: asyncio.Semaphore,
    req: dict,
    idx: int,
    results: list[RequestMetrics | None],
    max_new_tokens: int,
    base_url: str,
    stream: bool,
) -> None:
    payload = {
        "prompt_token_ids": req["token_ids"],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": stream,
    }
    t_start = time.monotonic()
    output_tokens = 0
    cached_tokens = 0
    ttft = 0.0
    first_token_seen = False

    async with sem:
        try:
            url = f"{base_url}/v1/completions"
            if stream:
                async with session.post(url, json=payload) as resp:
                    async for line in resp.content:
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if not first_token_seen:
                            ttft = time.monotonic() - t_start
                            first_token_seen = True
                        usage = data.get("usage", {})
                        output_tokens = usage.get("completion_tokens", output_tokens)
            else:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    ttft = time.monotonic() - t_start
                    first_token_seen = True
                    usage = data.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
        except Exception as e:
            print(f"  Warning: request {req.get('id', idx)} failed: {e}")
            return

    e2e = time.monotonic() - t_start
    if not first_token_seen:
        ttft = e2e

    results[idx] = RequestMetrics(
        request_id=req.get("id", f"req-{idx}"),
        input_tokens=len(req["token_ids"]),
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        e2e_latency_s=e2e,
        ttft_s=ttft,
        server_e2e_latency_s=0.0,
    )


async def run_poisson(
    base_url: str,
    requests: list[dict],
    arrival_rate: float,
    max_concurrent: int = 64,
    max_new_tokens: int = 32,
    stream: bool = True,
    backend: str = "sglang",
    seed: int = 42,
) -> RunResult:
    """Send requests with Poisson inter-arrivals to any running server."""
    import aiohttp

    rng = random.Random(seed)
    send_fn = _send_one_sglang if backend == "sglang" else _send_one_vllm

    sem = asyncio.Semaphore(max_concurrent)
    results: list[RequestMetrics | None] = [None] * len(requests)
    tasks: list[asyncio.Task] = []

    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        for idx, req in enumerate(requests):
            if idx > 0:
                delay = rng.expovariate(arrival_rate)
                await asyncio.sleep(delay)
            task = asyncio.create_task(
                send_fn(session, sem, req, idx, results, max_new_tokens, base_url, stream)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)
    total_time = time.monotonic() - t0

    metrics = [r for r in results if r is not None]
    return RunResult(n_requests=len(requests), metrics=metrics, total_time_s=total_time)


async def run_poisson_peek(
    base_url: str,
    requests: list[dict],
    arrival_rate: float,
    max_concurrent: int = 64,
    max_new_tokens: int = 32,
    stream: bool = True,
    backend: str = "sglang",
    seed: int = 42,
    accumulate_ms: float = 50.0,
) -> RunResult:
    """Send requests with Poisson arrivals + PeekDispatcher trie-based tagging.

    Same Poisson timing as :func:`run_poisson` (same seed, same rate).
    Each request is inserted into PeekDispatcher's incremental trie,
    tagged with ``peek:<rank>:<key>:<rid>`` (where *rank* is the
    count-aware DFS group rank), and sent immediately.  The server
    reads the tags to reconstruct group structure in O(N).
    """
    import aiohttp
    from peek.offline.reorder import PeekDispatcher

    rng = random.Random(seed)
    send_fn = _send_one_sglang if backend == "sglang" else _send_one_vllm

    sem = asyncio.Semaphore(max_concurrent)
    results: list[RequestMetrics | None] = [None] * len(requests)
    tasks: list[asyncio.Task] = []
    tasks_lock = asyncio.Lock()
    idx_map: dict[str, int] = {req.get("id", f"req-{i}"): i for i, req in enumerate(requests)}

    loop = asyncio.get_event_loop()
    session_holder: list = []

    def dispatch_fn(req: dict) -> None:
        """Send a tagged request concurrently.

        remove() is handled by PeekEngine.run() directly (step 9) --
        zero delay, same process.  No client-side callback needed.
        """
        orig_id = req.get("id", "")
        idx = idx_map.get(orig_id, 0)

        async def _send():
            session = session_holder[0]
            task = asyncio.create_task(
                send_fn(session, sem, req, idx, results,
                        max_new_tokens, base_url, stream)
            )
            async with tasks_lock:
                tasks.append(task)

        asyncio.run_coroutine_threadsafe(_send(), loop)

    dispatcher = PeekDispatcher(send_fn=dispatch_fn)

    t0 = time.monotonic()

    async with aiohttp.ClientSession() as session:
        session_holder.append(session)

        # Submit requests with Poisson inter-arrivals.
        # PeekDispatcher.submit() inserts into its trie, computes
        # count-aware DFS rank, tags, and calls dispatch_fn immediately.
        for i, req in enumerate(requests):
            if i > 0:
                delay = rng.expovariate(arrival_rate)
                await asyncio.sleep(delay)
            dispatcher.submit(req)

        # Wait for all in-flight HTTP requests to complete
        await asyncio.sleep(0.2)
        async with tasks_lock:
            pending = list(tasks)
        if pending:
            await asyncio.gather(*pending)

    total_time = time.monotonic() - t0

    metrics = [r for r in results if r is not None]
    return RunResult(n_requests=len(requests), metrics=metrics, total_time_s=total_time)


# ---------------------------------------------------------------------------
# Workload loading
# ---------------------------------------------------------------------------

def load_requests(
    workload: str,
    n: int,
    num_groups: int = 5,
    system_prompt_len: int = 2048,
    context_len: int = 2048,
    workload_kwargs: dict | None = None,
) -> list[dict]:
    gen_fn = WORKLOAD_GENERATORS[workload]
    import inspect
    sig = inspect.signature(gen_fn)
    kwargs: dict = {}
    if "num_groups" in sig.parameters:
        kwargs["num_groups"] = num_groups
    if "system_prompt_len" in sig.parameters:
        kwargs["system_prompt_len"] = system_prompt_len
    if "context_len" in sig.parameters:
        kwargs["context_len"] = context_len
    if workload_kwargs:
        kwargs.update(workload_kwargs)
    prompts = gen_fn(n, **kwargs)
    return to_request_dicts(prompts)


# ---------------------------------------------------------------------------
# Flush helper
# ---------------------------------------------------------------------------

def flush_cache(base_url: str) -> bool:
    """Flush server KV cache. Returns True on success."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url}/flush_cache", method="POST")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Poisson-arrival benchmark client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:30000",
        help="Server base URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--backend", type=str, default="sglang", choices=["sglang", "vllm"],
        help="Server backend type (default: sglang)",
    )
    parser.add_argument(
        "--workload", type=str, default="shared_system_prompts",
        help=f"Workload name. Available: {', '.join(WORKLOAD_GENERATORS.keys())}",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, required=True, help="Mean arrival rate (req/s)")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent requests")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max output tokens per request")
    parser.add_argument("--num-groups", type=int, default=5, help="Number of prefix groups")
    parser.add_argument("--system-prompt-len", type=int, default=2048, help="System prompt length in tokens")
    parser.add_argument("--context-len", type=int, default=2048, help="Max context length")
    parser.add_argument("--stream", action="store_true", default=True, help="Use streaming (default: True)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--flush-before", action="store_true", help="Flush server cache before benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Poisson process")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument(
        "--workload-kwargs", type=str, default=None,
        help="JSON dict of extra kwargs for workload generator",
    )
    args = parser.parse_args()

    stream = not args.no_stream
    wk = json.loads(args.workload_kwargs) if args.workload_kwargs else None

    print(f"Loading workload: {args.workload} (n={args.n}, groups={args.num_groups})")
    requests = load_requests(
        args.workload, args.n,
        num_groups=args.num_groups,
        system_prompt_len=args.system_prompt_len,
        context_len=args.context_len,
        workload_kwargs=wk,
    )
    print(f"  {len(requests)} requests loaded")

    if args.flush_before:
        print(f"Flushing cache at {args.base_url}...")
        if flush_cache(args.base_url):
            print("  Flushed.")
            time.sleep(0.5)
        else:
            print("  Warning: flush failed (server may not support it)")

    print(f"Sending {len(requests)} requests to {args.base_url} "
          f"(rate={args.arrival_rate} req/s, max_concurrent={args.max_concurrent}, "
          f"backend={args.backend}, stream={stream})")

    result = asyncio.run(run_poisson(
        base_url=args.base_url,
        requests=requests,
        arrival_rate=args.arrival_rate,
        max_concurrent=args.max_concurrent,
        max_new_tokens=args.max_new_tokens,
        stream=stream,
        backend=args.backend,
        seed=args.seed,
    ))

    print(f"\nResults:")
    print(result.summary())

    if args.output_json:
        out = {
            "workload": args.workload,
            "n_requests": result.n_requests,
            "arrival_rate": args.arrival_rate,
            "total_time_s": result.total_time_s,
            "throughput_req_s": result.request_throughput,
            "output_tok_s": result.output_token_throughput,
            "cache_hit_rate": result.cache_hit_rate,
            "mean_ttft_ms": result.mean_ttft_ms,
            "p99_ttft_ms": result.p99_ttft_ms,
            "mean_e2e_ms": result.mean_e2e_latency_ms,
            "p99_e2e_ms": result.p99_e2e_latency_ms,
            "metrics": [
                {
                    "request_id": m.request_id,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "cached_tokens": m.cached_tokens,
                    "e2e_latency_s": m.e2e_latency_s,
                    "ttft_s": m.ttft_s,
                }
                for m in result.metrics
            ],
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
