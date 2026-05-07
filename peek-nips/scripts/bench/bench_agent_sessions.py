#!/usr/bin/env python3
"""Agent-serving benchmark driver against a running sglang server.

Simulates N concurrent chat sessions, each with:
  * a system prompt (one of K "agent types" -- shared by ~ sessions/K users)
  * a target number of turns
  * response-conditional turn timing (user_think_time between turns)

Measures per-turn:
  * TTFT (time to first token)
  * end-to-end latency
  * prompt / completion tokens (reported by sglang)

Summary metrics:
  * p50 / p95 / p99 TTFT across turns
  * p50 / p95 / p99 end-to-end latency across turns
  * total throughput (tokens/sec)
  * per-turn cache hit inference (= prompt_tokens - cached_tokens reported)
  * total wall clock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import aiohttp


@dataclass
class TurnMetric:
    session_id: str
    agent_type: int
    turn_num: int
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: Optional[int]  # sglang reports "cached_tokens" in usage
    error: Optional[str] = None


async def run_session(
    session_id: str,
    agent_type: int,
    system_prompt: str,
    user_messages: List[str],
    n_turns: int,
    think_time_mean: float,
    max_tokens: int,
    endpoint: str,
    client: aiohttp.ClientSession,
    metrics: List[TurnMetric],
    seed: int,
    idle_after_turn: int = -1,
    idle_gap_s: float = 0.0,
) -> None:
    rng = random.Random(seed)
    conversation = [{"role": "system", "content": system_prompt}]

    for turn_num in range(n_turns):
        conversation.append({"role": "user", "content": user_messages[turn_num]})

        payload = {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "messages": conversation,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
        }

        t_start = time.perf_counter()
        first_token_t: Optional[float] = None
        output_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens: Optional[int] = None
        err = None

        try:
            async with client.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    err = f"HTTP {resp.status}: {await resp.text()}"
                else:
                    async for chunk in resp.content:
                        if not chunk:
                            continue
                        for line in chunk.decode("utf-8", errors="ignore").splitlines():
                            if not line.startswith("data: "):
                                continue
                            payload_str = line[len("data: "):]
                            if payload_str.strip() == "[DONE]":
                                continue
                            try:
                                obj = json.loads(payload_str)
                            except json.JSONDecodeError:
                                continue
                            # First chunk with content = TTFT marker
                            choices = obj.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    if first_token_t is None:
                                        first_token_t = time.perf_counter()
                                    output_text += content
                            # Usage only shows up in the final chunk
                            usage = obj.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                # sglang reports cached_tokens under prompt_tokens_details
                                details = usage.get("prompt_tokens_details") or {}
                                cached_tokens = details.get("cached_tokens")
                                if cached_tokens is None:
                                    cached_tokens = usage.get("cached_tokens")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        t_end = time.perf_counter()
        ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else -1.0
        total_ms = (t_end - t_start) * 1000

        metrics.append(
            TurnMetric(
                session_id=session_id,
                agent_type=agent_type,
                turn_num=turn_num,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                error=err,
            )
        )

        if err is not None:
            return

        conversation.append({"role": "assistant", "content": output_text})

        if turn_num < n_turns - 1:
            gap = rng.expovariate(1 / max(0.1, think_time_mean))
            # Extended idle window after a specific turn (bursty+idle scenario).
            if idle_after_turn >= 0 and turn_num == idle_after_turn:
                gap += idle_gap_s
            await asyncio.sleep(gap)


def _make_system_prompt(agent_idx: int, target_tokens: int) -> str:
    """Build a ~target_tokens-long system prompt that varies per agent type.
    Qwen tokenizer averages ~0.75 tokens per word for English prose; we aim
    high by using lots of filler to hit the token target."""
    header = f"You are Agent-{agent_idx}, a specialized assistant for domain {agent_idx}. "
    filler_chunk = (
        "You have access to the following tools: search_docs, execute_code, "
        "fetch_data, send_email, schedule_meeting, analyze_file, translate_text. "
        f"Domain-{agent_idx}-specific context: when responding, always follow "
        f"the domain-{agent_idx} style guide and cite the relevant subsection "
        f"number from the domain-{agent_idx} reference. "
    )
    # Roughly 20 tokens per filler_chunk; repeat until target
    return header + (filler_chunk * max(1, target_tokens // 20))


def _make_user_messages(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    questions = [
        "Please summarize the latest findings in this domain.",
        "What are the three most important concepts here?",
        "Explain the methodology in detail.",
        "Compare the two approaches mentioned above.",
        "What are the key risks we should consider?",
        "Draft a plan for next steps.",
        "Provide a bulleted list of action items.",
        "What would you recommend to improve this?",
        "Give me a concrete example.",
        "Identify the assumptions being made.",
    ]
    return [questions[rng.randrange(len(questions))] for _ in range(n)]


def summarize(metrics: List[TurnMetric], slos: Optional[dict] = None) -> dict:
    ok = [m for m in metrics if m.error is None and m.ttft_ms > 0]
    errored = [m for m in metrics if m.error is not None]
    slos = slos or {"ttft_ms": 2000, "itl_ms": 100, "e2e_ms": 60000}

    def pct(xs, p):
        if not xs:
            return -1.0
        xs_sorted = sorted(xs)
        k = int(p * (len(xs_sorted) - 1))
        return xs_sorted[k]

    def itl_of(m):
        if m.completion_tokens <= 1:
            return -1.0
        return (m.total_ms - m.ttft_ms) / max(1, m.completion_tokens - 1)

    ttfts = [m.ttft_ms for m in ok]
    totals = [m.total_ms for m in ok]
    itls = [itl_of(m) for m in ok if itl_of(m) > 0]
    prompt_tokens = sum(m.prompt_tokens for m in ok)
    completion_tokens = sum(m.completion_tokens for m in ok)
    cached_tokens = sum(m.cached_tokens or 0 for m in ok if m.cached_tokens is not None)
    n_turns = len(ok)

    # SLO attainment
    slo_ttft = sum(1 for m in ok if m.ttft_ms <= slos["ttft_ms"]) / max(1, n_turns)
    slo_itl = sum(1 for m in ok if 0 < itl_of(m) <= slos["itl_ms"]) / max(1, n_turns)
    slo_e2e = sum(1 for m in ok if m.total_ms <= slos["e2e_ms"]) / max(1, n_turns)
    slo_all = sum(
        1 for m in ok
        if m.ttft_ms <= slos["ttft_ms"]
        and 0 < itl_of(m) <= slos["itl_ms"]
        and m.total_ms <= slos["e2e_ms"]
    ) / max(1, n_turns)

    return {
        "n_turns_ok": n_turns,
        "n_errored": len(errored),
        "ttft_p50_ms": pct(ttfts, 0.50),
        "ttft_p95_ms": pct(ttfts, 0.95),
        "ttft_p99_ms": pct(ttfts, 0.99),
        "total_p50_ms": pct(totals, 0.50),
        "total_p95_ms": pct(totals, 0.95),
        "total_p99_ms": pct(totals, 0.99),
        "itl_p50_ms": pct(itls, 0.50),
        "itl_p95_ms": pct(itls, 0.95),
        "itl_p99_ms": pct(itls, 0.99),
        "prompt_tokens_total": prompt_tokens,
        "cached_tokens_total": cached_tokens,
        "cache_hit_rate_pct": (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0,
        "completion_tokens_total": completion_tokens,
        "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else -1,
        "mean_total_ms": statistics.fmean(totals) if totals else -1,
        "slo_attainment_pct": {
            "ttft": 100 * slo_ttft, "itl": 100 * slo_itl,
            "e2e": 100 * slo_e2e, "all": 100 * slo_all,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--agent-types", type=int, default=5)
    parser.add_argument("--system-prompt-tokens", type=int, default=3000)
    parser.add_argument("--think-time-mean", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/workspace/peek/bench_results.json")
    parser.add_argument("--label", default="run")
    parser.add_argument(
        "--idle-fraction", type=float, default=0.0,
        help="Fraction of sessions that have an extended idle gap mid-conversation",
    )
    parser.add_argument(
        "--idle-after-turn", type=int, default=1,
        help="After which turn (0-indexed) idle sessions pause",
    )
    parser.add_argument(
        "--idle-gap-s", type=float, default=20.0,
        help="Extra seconds added to think-time after idle-after-turn for idle sessions",
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=0.0,
        help="Sessions/sec Poisson arrival rate. 0 means all sessions start at t=0.",
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=2000)
    parser.add_argument("--e2e-slo-ms", type=float, default=60000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    agent_prompts = [
        _make_system_prompt(i, args.system_prompt_tokens)
        for i in range(args.agent_types)
    ]

    # Skew: agent 0 gets 40%, agent 1 gets 30%, rest split the remaining 30%.
    weights = [0.40, 0.30] + [
        0.30 / max(1, args.agent_types - 2)
    ] * max(0, args.agent_types - 2)
    weights = weights[: args.agent_types]

    metrics: List[TurnMetric] = []

    print(
        f"[bench] label={args.label} sessions={args.sessions} turns={args.turns} "
        f"agents={args.agent_types} sys_tokens≈{args.system_prompt_tokens} "
        f"think={args.think_time_mean}s max_tok={args.max_tokens}"
    )
    wall_start = time.perf_counter()

    # Precompute Poisson session start times (0 = all at t=0).
    if args.arrival_rate > 0:
        start_times = []
        t = 0.0
        for _ in range(args.sessions):
            start_times.append(t)
            t += rng.expovariate(args.arrival_rate)
    else:
        start_times = [0.0] * args.sessions

    conn = aiohttp.TCPConnector(limit=args.sessions * 2)
    async with aiohttp.ClientSession(connector=conn) as client:
        tasks = []
        n_idle = int(args.sessions * args.idle_fraction)
        for i in range(args.sessions):
            agent_idx = rng.choices(range(args.agent_types), weights=weights, k=1)[0]
            msgs = _make_user_messages(args.turns, seed=args.seed * 1000 + i)
            is_idle_session = (i < n_idle)
            start_delay = start_times[i]

            async def _delayed(i=i, agent_idx=agent_idx, msgs=msgs,
                                is_idle_session=is_idle_session,
                                start_delay=start_delay):
                if start_delay > 0:
                    await asyncio.sleep(start_delay)
                await run_session(
                    session_id=f"sess-{i:03d}",
                    agent_type=agent_idx,
                    system_prompt=agent_prompts[agent_idx],
                    user_messages=msgs,
                    n_turns=args.turns,
                    think_time_mean=args.think_time_mean,
                    max_tokens=args.max_tokens,
                    endpoint=args.endpoint,
                    client=client,
                    metrics=metrics,
                    seed=args.seed * 1000 + i,
                    idle_after_turn=args.idle_after_turn if is_idle_session else -1,
                    idle_gap_s=args.idle_gap_s if is_idle_session else 0.0,
                )

            tasks.append(asyncio.create_task(_delayed()))
        await asyncio.gather(*tasks)

    wall = time.perf_counter() - wall_start
    slos = {"ttft_ms": args.ttft_slo_ms, "itl_ms": 100, "e2e_ms": args.e2e_slo_ms}
    summary = summarize(metrics, slos=slos)
    summary["wall_clock_s"] = wall
    summary["throughput_toks_per_s"] = (
        (summary["prompt_tokens_total"] + summary["completion_tokens_total"]) / wall
    )
    slo_all_pct = summary["slo_attainment_pct"]["all"] / 100.0
    summary["goodput_req_per_s"] = (slo_all_pct * summary["n_turns_ok"]) / wall if wall > 0 else 0

    print("\n[summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<28s} {v:>10.2f}")
        else:
            print(f"  {k:<28s} {v}")

    out = {
        "label": args.label,
        "args": vars(args),
        "summary": summary,
        "turns": [asdict(m) for m in metrics],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[bench] wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
