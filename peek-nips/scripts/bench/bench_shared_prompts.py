#!/usr/bin/env python3
"""Shared-system-prompts benchmark client with Poisson arrivals.

Generates a workload of N requests split across G groups (uniform). Every
request in group g shares the same system prompt of `prefix_tokens` tokens;
each request has a unique user question appended. Dispatches requests at a
Poisson rate and measures per-request TTFT, ITL, E2E, and cache hit stats.

Metrics reported (JSON summary + stdout):
  - request_throughput (req/s actually completed over wall clock)
  - token throughput: input, output, total (tok/s)
  - TTFT (time-to-first-token): p50 / p95 / p99 / mean
  - ITL (inter-token latency, excl. TTFT): p50 / p95 / p99 / max / mean
  - TPOT (per-output-token): same percentiles. Same quantity as ITL in most
    serving systems; reported for continuity with bench_serving.
  - E2E latency: p50 / p95 / p99 / mean
  - cache hit rate: sum(cached_tokens) / sum(prompt_tokens) across all reqs
  - prefill tokens: total prompt tokens, total cached, total actually prefilled
  - goodput under SLO: % of requests meeting (TTFT, TPOT, E2E) SLO jointly
  - running concurrency stats: mean in-flight, max in-flight
  - errored / timed-out request counts
  - warmup exclusions (first N excluded from metrics)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import aiohttp


# ---------------------------------------------------------------------------
# Workload generation -- LooGLE-backed (paper-comparable) with synthetic fallback
# ---------------------------------------------------------------------------


def _try_load_loogle(max_docs: int):
    """Load LooGLE longdep_qa directly via the datasets library.

    Returns list[{context, title, questions}] on success, None on failure
    (no network, dataset cache missing, etc). Self-contained -- does not
    import from superkv to avoid sys.path conflicts with our peek package.
    """
    import json as _json
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"[bench] HF datasets unavailable ({e}); falling back to synthetic.")
        return None
    try:
        try:
            rows = load_dataset("bigainlco/LooGLE", "longdep_qa", split="test")
        except TypeError:
            rows = load_dataset(
                "bigainlco/LooGLE", "longdep_qa", split="test", trust_remote_code=True
            )
    except Exception as e:
        print(f"[bench] LooGLE download failed ({e}); falling back to synthetic.")
        return None
    # LooGLE longdep_qa is one row per question. Group by doc_id so each
    # "document" becomes a cluster of (shared context + list of real questions).
    by_doc: dict = {}
    order: list = []
    for row in rows:
        did = row.get("doc_id") or row.get("id")
        if did is None:
            continue
        if did not in by_doc:
            by_doc[did] = {
                "context": row.get("context", "") or row.get("input", ""),
                "title": row.get("title", ""),
                "questions": [],
            }
            order.append(did)
        q = row.get("question") or row.get("query") or row.get("output") or ""
        if q:
            by_doc[did]["questions"].append(q)
    docs = [by_doc[did] for did in order if by_doc[did]["context"] and by_doc[did]["questions"]]
    docs = docs[:max_docs]
    if not docs:
        print("[bench] LooGLE returned no usable documents; falling back to synthetic.")
        return None
    return docs


def _try_load_repobench(max_docs: int, prefix_tokens: int):
    """Load RepoBench (Python, cross-file-first) and return docs grouped by repo.

    RepoBench buckets examples by token count under `level` ('2k','4k','8k',
    '12k','16k','24k','32k','64k','128k'). We pick the bucket nearest to our
    prefix_tokens target so per-doc context is naturally close to the right
    size before tokenizer truncation.

    Returns list[{context, title, questions}] on success, None on failure.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"[bench] HF datasets unavailable ({e}); falling back to synthetic.")
        return None
    try:
        rows = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first")
    except Exception as e:
        print(f"[bench] RepoBench download failed ({e}); falling back to synthetic.")
        return None
    # Pick the level bucket nearest to our prefix_tokens target.
    levels_avail = ["2k", "4k", "8k", "12k", "16k", "24k", "32k", "64k", "128k"]
    levels_tok = {"2k": 2000, "4k": 4000, "8k": 8000, "12k": 12000, "16k": 16000,
                  "24k": 24000, "32k": 32000, "64k": 64000, "128k": 128000}
    target_level = min(levels_avail, key=lambda lvl: abs(levels_tok[lvl] - prefix_tokens))
    candidates = [r for r in rows if r.get("level") == target_level]
    if not candidates:
        print(f"[bench] RepoBench level={target_level} empty; falling back to synthetic.")
        return None
    # Group by repo (one example per repo to maximize prefix diversity).
    by_repo: dict = {}
    order: list = []
    for row in candidates:
        repo = row.get("repo_name", "?")
        if repo in by_repo:
            continue
        # Reconstruct a "document" prefix: imports + cross-file snippets + in-file context.
        parts = []
        imp = row.get("import_statement") or ""
        if imp:
            parts.append(imp)
        ctx_list = row.get("context") or []
        if ctx_list:
            parts.append("# Cross-file context:")
            for snip in ctx_list:
                if isinstance(snip, dict):
                    parts.append(f"\n# from {snip.get('path','?')}:\n{snip.get('snippet','')}")
        cropped = row.get("cropped_code") or ""
        if cropped:
            parts.append(f"\n# Current file:\n{cropped}")
        doc_text = "\n".join(parts)
        if len(doc_text) < 500:  # skip degenerate examples
            continue
        by_repo[repo] = {
            "context": doc_text,
            "title": f"{repo}/{row.get('file_path','?')}",
            "questions": list(_REPOBENCH_QUESTIONS),
        }
        order.append(repo)
        if len(by_repo) >= max_docs:
            break
    docs = [by_repo[r] for r in order][:max_docs]
    if not docs:
        print(f"[bench] RepoBench produced no usable docs at level={target_level}; falling back.")
        return None
    print(f"[bench] loaded {len(docs)} RepoBench Python repos (level={target_level}).")
    return docs


_REPOBENCH_QUESTIONS = [
    "Explain what this code does at a high level.",
    "Identify potential bugs or edge cases in this code.",
    "Suggest concrete improvements to this code.",
    "Add docstrings and type annotations to this code.",
    "Refactor the most complex function in this code.",
    "What inputs does the main function expect, and what does it return?",
    "Trace the data flow through this code, step by step.",
    "Find performance bottlenecks and propose optimizations.",
    "Write unit tests covering the main functionality.",
    "Explain how this code interacts with the cross-file dependencies.",
]


def _get_tokenizer(model_name: str):
    """Load the model's tokenizer -- used to truncate LooGLE docs precisely to
    `prefix_tokens`. Returns None if transformers isn't importable."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        print(f"[bench] transformers unavailable ({e}); cannot use LooGLE precisely.")
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"[bench] tokenizer load failed ({e}); falling back to synthetic.")
        return None


def _synthetic_group_system_prompt(group_id: int, target_tokens: int) -> str:
    """Fallback deterministic per-group prompt of approximately target_tokens.
    Qwen tokenizer ≈ 4 chars/token for English prose."""
    header = (
        f"You are Agent-{group_id}, a specialist in domain {group_id}. "
        f"All responses must cite the domain-{group_id} reference manual. "
    )
    filler_chunk = (
        f"Domain-{group_id} context: the following guidelines apply to every "
        f"question in this domain. Use precise terminology, include references, "
        f"prefer structured responses, and respect the domain-{group_id} "
        f"formatting conventions which differ from other domains. "
    )
    target_chars = max(len(header) + 50, target_tokens * 4)
    out = header
    while len(out) < target_chars:
        out += filler_chunk
    return out


_SYNTHETIC_QUESTIONS = [
    "Summarize the key points in this domain.",
    "List three common pitfalls and how to avoid them.",
    "Provide a worked example with step-by-step reasoning.",
    "Compare two alternative approaches.",
    "Explain the methodology in plain language.",
    "What are the underlying assumptions?",
    "Draft a one-paragraph action plan.",
    "Give a concrete, runnable example.",
    "Identify the key risks and mitigations.",
    "Propose improvements to the current approach.",
]


def _build_group_prompts(
    groups: int, prefix_tokens: int, model_name: str, dataset: str,
):
    """Return (group_prompts, group_questions) where
      group_prompts[g] is the text of group g's system prompt (~prefix_tokens
        tokens when encoded by the model's tokenizer), and
      group_questions[g] is a list of candidate user questions for that group.

    dataset='loogle' uses real LooGLE long documents + their bundled questions;
    dataset='synthetic' uses the fallback generator. 'auto' tries LooGLE first
    and falls back to synthetic on any failure.
    """
    docs = None
    tokenizer = None
    if dataset == "repobench":
        docs = _try_load_repobench(max_docs=groups, prefix_tokens=prefix_tokens)
        tokenizer = _get_tokenizer(model_name) if docs else None
    elif dataset in ("loogle", "auto"):
        docs = _try_load_loogle(max_docs=groups)
        tokenizer = _get_tokenizer(model_name) if docs else None

    if docs and tokenizer is not None:
        # Real LooGLE path: truncate each doc's context to prefix_tokens tokens.
        group_prompts = []
        group_questions = []
        n_real = len(docs)
        for g in range(groups):
            doc = docs[g % n_real]
            ctx = doc["context"]
            ids = tokenizer.encode(ctx, add_special_tokens=False)
            if len(ids) >= prefix_tokens:
                trimmed = tokenizer.decode(ids[:prefix_tokens])
            else:
                # Doc is shorter than requested prefix; repeat + salt with group id
                # so each group keeps a unique prefix (otherwise low-G with
                # short docs would produce identical prefixes across groups).
                trimmed = (ctx + f"\n\n[Domain-{g}]\n") * max(
                    1, (prefix_tokens * 4) // max(1, len(ctx))
                )
                tids = tokenizer.encode(trimmed, add_special_tokens=False)
                trimmed = tokenizer.decode(tids[:prefix_tokens])
            group_prompts.append(trimmed)
            qs = doc.get("questions") or []
            group_questions.append(qs if qs else list(_SYNTHETIC_QUESTIONS))
        print(
            f"[bench] loaded {n_real} LooGLE docs; "
            f"using {groups} group prompts (~{prefix_tokens} tokens each)."
        )
        return group_prompts, group_questions

    # Synthetic fallback: deterministic filler, shared questions.
    if dataset == "loogle":
        raise RuntimeError(
            "dataset='loogle' was requested but LooGLE / tokenizer loading failed. "
            "Run with dataset='auto' to permit synthetic fallback."
        )
    if dataset == "repobench":
        raise RuntimeError(
            "dataset='repobench' was requested but RepoBench / tokenizer loading failed."
        )
    print("[bench] using synthetic shared-prefix workload (no LooGLE).")
    group_prompts = [_synthetic_group_system_prompt(g, prefix_tokens) for g in range(groups)]
    group_questions = [list(_SYNTHETIC_QUESTIONS) for _ in range(groups)]
    return group_prompts, group_questions


@dataclass
class WorkloadRequest:
    req_id: int
    group_id: int
    system_prompt: str
    user_message: str
    max_tokens: int              # ceiling actually sent to sglang (hard stop)
    target_tokens: int           # target actual decode length (communicated via prompt)
    arrival_time_s: float        # seconds after t=0


def _zipf_group_assignments(
    rng: random.Random, n: int, num_groups: int, alpha: float
) -> List[int]:
    """Assign n requests to groups following a Zipf distribution.

    Group 0 is hottest, group num_groups-1 is coldest. Weight of group k
    is proportional to 1/(k+1)^alpha. alpha=1.0 gives the classic 80/20
    skew; higher alpha = more skewed (colder tail gets almost no traffic).
    """
    weights = [1.0 / ((k + 1) ** alpha) for k in range(num_groups)]
    # Randomly shuffle group index -> weight mapping so "hot" groups aren't
    # literally group 0, 1, 2... (avoids arbitrary correlation with tree node_ids).
    perm = list(range(num_groups))
    rng.shuffle(perm)
    shuffled_weights = [weights[perm.index(k)] for k in range(num_groups)]
    # Cumulative distribution for sampling.
    total = sum(shuffled_weights)
    cum = []
    s = 0.0
    for w in shuffled_weights:
        s += w / total
        cum.append(s)
    assignments: List[int] = []
    for _ in range(n):
        r = rng.random()
        lo, hi = 0, num_groups - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        assignments.append(lo)
    return assignments


def generate_workload(
    n: int,
    groups: int,
    prefix_tokens: int,
    max_tokens: int,
    rate_req_per_s: float,
    seed: int,
    model_name: str,
    dataset: str,
    distribution: str = "uniform",
    zipf_alpha: float = 1.0,
) -> List[WorkloadRequest]:
    """Group-assignment + Poisson inter-arrival schedule.

    distribution: 'uniform' (round-robin random) or 'zipf' (hot groups see
    much more traffic -- Zipf-weighted sampling with exponent zipf_alpha).

    Both group assignment and arrival schedule are fully deterministic given
    (seed, n, groups, rate, distribution, zipf_alpha). Same seed replays the
    same workload across policies -> fair comparison.
    """
    rng = random.Random(seed)
    group_prompts, group_questions = _build_group_prompts(
        groups=groups, prefix_tokens=prefix_tokens,
        model_name=model_name, dataset=dataset,
    )
    # Group assignments per request.
    if distribution == "zipf":
        group_assignments = _zipf_group_assignments(rng, n, groups, zipf_alpha)
    else:
        group_assignments = [rng.randrange(groups) for _ in range(n)]
    requests: List[WorkloadRequest] = []
    t = 0.0
    for i in range(n):
        g = group_assignments[i]
        qs = group_questions[g]
        q = qs[rng.randrange(len(qs))]
        if rate_req_per_s > 0:
            # Inter-arrival ~ Exp(rate); cumulative arrival times form Poisson.
            t += rng.expovariate(rate_req_per_s)
        requests.append(
            WorkloadRequest(
                req_id=i,
                group_id=g,
                system_prompt=group_prompts[g],
                user_message=q,
                max_tokens=max_tokens,
                target_tokens=max_tokens,
                arrival_time_s=t if rate_req_per_s > 0 else 0.0,
            )
        )
    return requests


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@dataclass
class ReqMetric:
    req_id: int
    group_id: int
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    # Wall-clock timestamps (time.time(), seconds) to join with server-side
    # phase-timing dump. dispatch_ts is when the client actually sent the
    # request (post-semaphore, post-arrival-delay); first_token_ts and
    # last_token_ts are wall-clock moments for the first and last streamed
    # content tokens. response_id is sglang's internal rid observed from
    # the OpenAI stream -- matches server-side req.rid for cross-process join.
    dispatch_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    response_id: Optional[str] = None
    itl_samples_ms: List[float] = field(default_factory=list)
    error: Optional[str] = None


async def dispatch_one(
    req: WorkloadRequest,
    endpoint: str,
    model: str,
    client: aiohttp.ClientSession,
    in_flight_tracker: dict,
) -> ReqMetric:
    # Build user message. If the per-req target length differs from the
    # hard ceiling (hide-max-tokens mode), append a length-hint instruction
    # so the model actually varies its output. Qwen follows "in about N
    # words" reasonably well -- not exact, which is what we want: the
    # sglang new_token_ratio estimator must have genuine uncertainty.
    user_text = req.user_message
    if req.target_tokens != req.max_tokens:
        # ~1 token ≈ 0.75 English words.
        target_words = max(10, int(req.target_tokens * 0.75))
        user_text = (
            f"{req.user_message}\n\n"
            f"Please respond in approximately {target_words} words."
        )
    messages = []
    if req.system_prompt:
        if os.environ.get("BENCH_NO_SYSTEM_ROLE"):
            user_text = f"{req.system_prompt}\n\n{user_text}"
        else:
            messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    t_start = time.perf_counter()
    dispatch_wall = time.time()
    in_flight_tracker["cur"] += 1
    in_flight_tracker["max"] = max(in_flight_tracker["max"], in_flight_tracker["cur"])
    first_token_t: Optional[float] = None
    last_token_t: Optional[float] = None
    first_token_wall: float = 0.0
    last_token_wall: float = 0.0
    itl_samples: List[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    response_id: Optional[str] = None
    err: Optional[str] = None
    try:
        async with client.post(
            endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=int(os.environ.get("BENCH_HTTP_TIMEOUT_S", 1800)))
        ) as resp:
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for chunk in resp.content:
                    if not chunk:
                        continue
                    for line in chunk.decode("utf-8", errors="ignore").splitlines():
                        if not line.startswith("data: "):
                            continue
                        body = line[len("data: "):].strip()
                        if body == "[DONE]":
                            continue
                        try:
                            obj = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        # Server-side rid (sglang's req.rid) appears in every
                        # chunk's top-level 'id'. Used to join with the
                        # server-side phase-timing dump.
                        if response_id is None:
                            response_id = obj.get("id")
                        # Streamed content tokens.
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                now = time.perf_counter()
                                now_wall = time.time()
                                if first_token_t is None:
                                    first_token_t = now
                                    first_token_wall = now_wall
                                elif last_token_t is not None:
                                    itl_samples.append((now - last_token_t) * 1000)
                                last_token_t = now
                                last_token_wall = now_wall
                        # Usage arrives in the final chunk.
                        usage = obj.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0) or 0
                            completion_tokens = usage.get("completion_tokens", 0) or 0
                            details = usage.get("prompt_tokens_details") or {}
                            cached_tokens = (
                                details.get("cached_tokens")
                                or usage.get("cached_tokens")
                                or 0
                            )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        in_flight_tracker["cur"] -= 1
    t_end = time.perf_counter()
    return ReqMetric(
        req_id=req.req_id,
        group_id=req.group_id,
        ttft_ms=(first_token_t - t_start) * 1000 if first_token_t else -1.0,
        total_ms=(t_end - t_start) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        dispatch_ts=dispatch_wall,
        first_token_ts=first_token_wall,
        last_token_ts=last_token_wall,
        response_id=response_id,
        itl_samples_ms=itl_samples,
        error=err,
    )


async def run_sweep(args) -> dict:
    workload = generate_workload(
        n=args.n,
        groups=args.groups,
        prefix_tokens=args.prefix_tokens,
        max_tokens=args.max_tokens,
        rate_req_per_s=args.rate,
        seed=args.seed,
        model_name=args.model,
        dataset=args.dataset,
        distribution=args.distribution,
        zipf_alpha=args.zipf_alpha,
    )
    if args.decode_mix:
        mix = []
        for part in args.decode_mix.split(","):
            pct_s, len_s = part.strip().split(":")
            mix.append((int(pct_s), int(len_s)))
        if sum(p for p, _ in mix) != 100:
            raise ValueError(f"--decode-mix percentages must sum to 100, got {mix}")
        # Production pattern: target length is a property of the GROUP (the
        # type of workload -- chat vs RAG vs CoT), not a per-request roll.
        # Reqs sharing a system prompt come from the same caller and have
        # similar output-length distributions. Assigning one target per group
        # gives peek's per-cluster decode predictor real signal to learn from.
        rng = random.Random(args.seed ^ 0xD3C0DE)
        n_groups = max(args.groups, 1)
        group_lengths: List[int] = []
        for pct, ln in mix:
            group_lengths.extend([ln] * ((pct * n_groups) // 100))
        while len(group_lengths) < n_groups:
            group_lengths.append(mix[-1][1])
        rng.shuffle(group_lengths)
        # group_id -> target_tokens map.
        group_target = {g: group_lengths[g] for g in range(n_groups)}
        for req in workload:
            ln = group_target[req.group_id]
            req.target_tokens = ln
            if args.hide_max_tokens:
                # Uniform loose ceiling -> sglang must estimate via new_token_ratio.
                req.max_tokens = args.max_tokens
            else:
                req.max_tokens = ln
        hist: dict = {}
        for ln in group_lengths:
            hist[ln] = hist.get(ln, 0) + 1
        print(
            f"[bench] decode_mix applied per-group: groups_by_target={sorted(hist.items())} "
            f"hide_max_tokens={args.hide_max_tokens} "
            f"uniform_ceiling={args.max_tokens if args.hide_max_tokens else 'per-req'}"
        )
    print(
        f"[bench] label={args.label} n={args.n} groups={args.groups} "
        f"prefix={args.prefix_tokens} decode={args.max_tokens} rate={args.rate} "
        f"concurrency_cap={args.concurrency} warmup={args.warmup_reqs}"
    )
    metrics: List[ReqMetric] = []
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
                    # Wait until the request's scheduled arrival time.
                    elapsed = time.perf_counter() - wall_start
                    if req.arrival_time_s > elapsed:
                        await asyncio.sleep(req.arrival_time_s - elapsed)
                if sem is not None:
                    async with sem:
                        m = await dispatch_one(
                            req, args.endpoint, args.model, client, in_flight
                        )
                else:
                    m = await dispatch_one(
                        req, args.endpoint, args.model, client, in_flight
                    )
                metrics.append(m)
            tasks.append(asyncio.create_task(_fire()))
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall_start

    # Post-process.
    return summarize(metrics, wall, args, in_flight_max=in_flight["max"])


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return -1.0
    xs_sorted = sorted(xs)
    k = int(p * (len(xs_sorted) - 1))
    return xs_sorted[k]


def _load_server_phases(path_or_glob: str) -> dict:
    """Read the server-side phase-timing dump ({rid: {arrive_ts, pick_ts}}).

    Accepts either a literal file path or a glob pattern with '{pid}' (or a
    shell-style `*`) -- sglang runs several subprocesses and each writes its
    own PID-suffixed file. We read all of them and merge; the scheduler
    process is the only one that actually populates rids, so there's no
    conflict across files.

    Returns {} on any failure -- caller falls back to client-only metrics.
    """
    import glob as _glob
    if "{pid}" in path_or_glob:
        pattern = path_or_glob.replace("{pid}", "*")
    elif "*" in path_or_glob:
        pattern = path_or_glob
    else:
        pattern = path_or_glob
    paths = _glob.glob(pattern) if ("*" in pattern) else [pattern]
    merged: dict = {}
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


def summarize(
    metrics: List[ReqMetric], wall_s: float, args, in_flight_max: int
) -> dict:
    # Drop warmup: first `warmup_reqs` requests (by req_id) are excluded
    # from metrics, but their tokens count for wall-clock throughput (they
    # consumed GPU time). Arrival-order exclusion is deterministic.
    eff = [m for m in metrics if m.req_id >= args.warmup_reqs]
    ok = [m for m in eff if m.error is None and m.ttft_ms > 0]
    errored = [m for m in eff if m.error is not None]
    timed_out_or_bad = len(eff) - len(ok) - len(errored)

    ttfts = [m.ttft_ms for m in ok]
    totals = [m.total_ms for m in ok]
    # ITL: flatten all inter-token samples. TPOT: (total - ttft) / (n_tok - 1).
    itls_all: List[float] = []
    tpots: List[float] = []
    for m in ok:
        itls_all.extend(m.itl_samples_ms)
        if m.completion_tokens >= 2:
            tpots.append((m.total_ms - m.ttft_ms) / (m.completion_tokens - 1))

    prompt_total = sum(m.prompt_tokens for m in ok)
    completion_total = sum(m.completion_tokens for m in ok)
    cached_total = sum(m.cached_tokens for m in ok)
    prefilled_actual = max(0, prompt_total - cached_total)

    # Goodput under SLO -- joint (all three must hold per-request).
    def passes(m: ReqMetric) -> bool:
        if m.error is not None or m.ttft_ms <= 0:
            return False
        tpot = (m.total_ms - m.ttft_ms) / max(1, m.completion_tokens - 1)
        return (
            m.ttft_ms <= args.ttft_slo_ms
            and tpot <= args.tpot_slo_ms
            and m.total_ms <= args.e2e_slo_ms
        )

    slo_ok = sum(1 for m in ok if passes(m))
    slo_rate = slo_ok / max(1, len(eff))
    # Goodput in req/s: successful-under-SLO completions per wall second.
    goodput_req_per_s = slo_ok / wall_s if wall_s > 0 else 0

    # ------------------------------------------------------------------
    # Phase decomposition -- join per-request client data with server-side
    # arrive/pick timings dumped by peek's patch_hook.
    #
    #   server_queue_wait = pick_ts - arrive_ts
    #   server_prefill    = first_token_ts - pick_ts
    #   decode            = last_token_ts - first_token_ts (client only)
    #   client_queue      = dispatch_ts - scheduled_arrival (client-only;
    #                       time spent waiting for the concurrency semaphore)
    #
    # arrive_ts from the server is when the rid FIRST appears in the sync
    # loop's waiting_queue snapshot. That's ~ a few ms after sglang's HTTP
    # ingress accepted the request. dispatch_ts from the client is when
    # aiohttp returned from .post() entry -- the two differ by network +
    # tokenizer pre-processing. We treat server arrive_ts as the authority
    # for server-side queue start.
    server_phases = _load_server_phases(args.phase_dump_path)
    q_samples, p_samples, d_samples, cq_samples = [], [], [], []
    matched = 0
    # schedule_times indexed by req_id (req.arrival_time_s is seconds since
    # the wall_start, captured during dispatch setup).
    for m in ok:
        q_ms = p_ms = d_ms = cq_ms = None
        # Client-side queue wait (semaphore delay): time between when the
        # request was SCHEDULED to arrive and when it actually dispatched.
        # Requires that we recorded arrival_time_s on workload gen AND
        # stored wall_start; we pass them through for consistency.
        # For now, we skip this here and compute post-hoc if needed.
        if m.first_token_ts > 0 and m.last_token_ts > 0:
            d_ms = (m.last_token_ts - m.first_token_ts) * 1000
        info = server_phases.get(m.response_id) if m.response_id else None
        if info and "arrive_ts" in info and "pick_ts" in info:
            q_ms = (info["pick_ts"] - info["arrive_ts"]) * 1000
            if m.first_token_ts > 0:
                p_ms = (m.first_token_ts - info["pick_ts"]) * 1000
            matched += 1
        if q_ms is not None:
            q_samples.append(q_ms)
        if p_ms is not None:
            p_samples.append(p_ms)
        if d_ms is not None:
            d_samples.append(d_ms)

    def _phase(xs):
        if not xs:
            return {"samples": 0, "mean": -1, "p50": -1, "p95": -1, "p99": -1}
        return {
            "samples": len(xs),
            "mean": statistics.fmean(xs),
            "p50": _pct(xs, 0.50),
            "p95": _pct(xs, 0.95),
            "p99": _pct(xs, 0.99),
        }

    return {
        "label": args.label,
        "args": {
            k: v for k, v in vars(args).items()
            if not k.startswith("_") and not callable(v)
        },
        "wall_clock_s": wall_s,
        "counts": {
            "submitted": len(metrics),
            "after_warmup": len(eff),
            "ok": len(ok),
            "errored": len(errored),
            "bad_ttft": timed_out_or_bad,
            "slo_met": slo_ok,
        },
        "throughput": {
            "request_per_s": len(ok) / wall_s if wall_s > 0 else 0,
            "input_tok_per_s": prompt_total / wall_s if wall_s > 0 else 0,
            "output_tok_per_s": completion_total / wall_s if wall_s > 0 else 0,
            "total_tok_per_s": (
                (prompt_total + completion_total) / wall_s if wall_s > 0 else 0
            ),
            "goodput_req_per_s": goodput_req_per_s,
            "slo_attainment_pct": 100 * slo_rate,
        },
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) if ttfts else -1,
            "p50": _pct(ttfts, 0.50),
            "p95": _pct(ttfts, 0.95),
            "p99": _pct(ttfts, 0.99),
        },
        "itl_ms": {
            "mean": statistics.fmean(itls_all) if itls_all else -1,
            "p50": _pct(itls_all, 0.50),
            "p95": _pct(itls_all, 0.95),
            "p99": _pct(itls_all, 0.99),
            "max": max(itls_all) if itls_all else -1,
            "samples": len(itls_all),
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpots) if tpots else -1,
            "p50": _pct(tpots, 0.50),
            "p95": _pct(tpots, 0.95),
            "p99": _pct(tpots, 0.99),
        },
        "e2e_ms": {
            "mean": statistics.fmean(totals) if totals else -1,
            "p50": _pct(totals, 0.50),
            "p95": _pct(totals, 0.95),
            "p99": _pct(totals, 0.99),
        },
        "cache": {
            "hit_rate_pct": (
                100.0 * cached_total / prompt_total if prompt_total else 0.0
            ),
            "prompt_tokens_total": prompt_total,
            "cached_tokens_total": cached_total,
            "prefilled_tokens_total": prefilled_actual,
        },
        "concurrency": {
            "max_in_flight": in_flight_max,
            "cap": args.concurrency,
        },
        "phases": {
            "matched_rids": matched,
            "total_ok": len(ok),
            "server_queue_wait_ms": _phase(q_samples),
            "server_prefill_ms":    _phase(p_samples),
            "decode_ms":            _phase(d_samples),
        },
        "per_request": [asdict(m) for m in metrics] if args.save_per_request else [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:30000/v1/chat/completions")
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--n", type=int, default=1000, help="Total requests.")
    p.add_argument("--groups", type=int, default=100, help="G: number of prefix groups.")
    p.add_argument("--prefix-tokens", type=int, default=2048)
    p.add_argument(
        "--dataset", choices=("loogle", "synthetic", "auto", "repobench"), default="auto",
        help="loogle = real long docs (paper-comparable); repobench = real Python repo "
             "code-context (RepoBench cross-file-first); synthetic = deterministic filler; "
             "auto = try loogle then fall back to synthetic.",
    )
    p.add_argument(
        "--distribution", choices=("uniform", "zipf"), default="uniform",
        help="Group sampling distribution: uniform (default) or zipf.",
    )
    p.add_argument(
        "--zipf-alpha", type=float, default=1.0,
        help="Zipf exponent when --distribution=zipf. 1.0 = classic 80/20, higher = more skewed.",
    )
    p.add_argument("--max-tokens", type=int, default=128, help="Decode length target.")
    p.add_argument(
        "--decode-mix", default="",
        help="Mixed decode lengths: 'pct:len,pct:len,...'. "
             "Example: '20:512,60:4096,20:8192' -> 20%% 512, 60%% 4096, 20%% 8192. "
             "Overrides --max-tokens when set. Percentages must sum to 100.",
    )
    p.add_argument(
        "--hide-max-tokens", action="store_true",
        help="Send uniform max_tokens=--max-tokens to sglang regardless of "
             "--decode-mix target. Target length is communicated to the model "
             "via a 'respond in ~N words' prompt instruction instead. This "
             "makes sglang's new_token_ratio estimator uncertain, exercising "
             "KV-budget admission control the way production traffic does.",
    )
    p.add_argument(
        "--rate", type=float, default=0.0,
        help="Poisson arrival rate req/s. 0 = closed-loop (fire all at t=0, gated by --concurrency).",
    )
    p.add_argument(
        "--concurrency", type=int, default=0,
        help="Max in-flight requests (closed-loop cap). 0 = uncapped.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-reqs", type=int, default=50)
    p.add_argument("--ttft-slo-ms", type=float, default=2000)
    p.add_argument("--tpot-slo-ms", type=float, default=100)
    p.add_argument("--e2e-slo-ms", type=float, default=60000)
    p.add_argument("--output", default="/tmp/bench_shared_prompts.json")
    p.add_argument("--label", default="run")
    p.add_argument(
        "--save-per-request", action="store_true",
        help="Embed per-request metrics in the JSON (large).",
    )
    p.add_argument(
        "--phase-dump-path", default="/tmp/peek_phases_{pid}.json",
        help="Where the server's peek patch_hook dumps per-rid phase timings. "
             "Accepts {pid} or '*' -- client reads all matching files and merges "
             "(sglang runs several subprocesses; only the scheduler one has "
             "real data, but each writes its own PID-suffixed file).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    summary = asyncio.run(run_sweep(args))
    # Pretty-print key metrics to stdout.
    print("\n[summary]")
    for section in ("throughput", "ttft_ms", "itl_ms", "tpot_ms", "e2e_ms", "cache", "phases", "concurrency", "counts"):
        print(f"  {section}:")
        for k, v in summary[section].items():
            if isinstance(v, float):
                print(f"    {k:<24s} {v:>12.3f}")
            elif isinstance(v, dict):
                print(f"    {k:<24s} "
                      + " ".join(f"{kk}={vv}" for kk, vv in v.items()))
            else:
                print(f"    {k:<24s} {v}")
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
