#!/usr/bin/env python3
"""W2 vllm-side bench harness: replays Mooncake conversation_trace as a
multi-round agentic workload against a vllm OpenAI-compatible server.

Mirrors the parts of sglang.bench_serving.get_mooncake_request_over_time
that matter for peek's measurements (4-round burst, inter-turn gap,
optional cross-session shared system prompt) without depending on sglang.

Prompt construction
-------------------
Each Mooncake record carries `hash_ids` (list of ints): the canonical hash
of each block in that session's prefix. To make vllm's prefix cache hit
when two sessions share leading `hash_ids`, we map each hash_id
deterministically to a ~512-token block of text. Concatenating those
blocks yields a per-session prompt whose token-prefix tree mirrors the
hash_ids tree -- exactly the shape the trace encodes.

This is faithful to peek's purpose (measure scheduler + eviction under
realistic prefix sharing) but does not preserve the literal user/assistant
text from the trace (Mooncake does not provide it; sglang's reference
also fakes the text).

Output
------
Single JSON file matching W1's schema (`wall_clock_s`, `throughput.*`,
`ttft_ms.*`, `tpot_ms.*`, `cache.hit_rate_pct`, etc.) so the W1
aggregator can pick it up unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp


# ----- prompt synthesis ------------------------------------------------

# Common English stopwords tokenize to ~0.88 tokens/word in Mistral/Llama
# BPE. A 580-word random sequence ≈ 510 tokens -- matches the Mooncake
# block convention (~512 tokens/block). Seeding the RNG with the hash_id
# makes the per-block sequence deterministic and distinct from the first
# token, so identical hash_ids across sessions hit vllm's prefix cache and
# different hash_ids do not.
_STOPWORD_VOCAB = (
    "the of to and in is a it be that for on with as this by not at or "
    "from but they we he his she you have are had do can all so if "
    "my me your an one would about who which their will more no other "
    "into when out up than them then now its over also some what only "
    "first many time these like our could been new"
).split()
_WORDS_PER_BLOCK = 470
_BLOCK_CACHE: dict[int, str] = {}


def block_text(hash_id: int) -> str:
    """Deterministic ~512-token text snippet for one block, keyed by hash."""
    cached = _BLOCK_CACHE.get(hash_id)
    if cached is not None:
        return cached
    rng = random.Random(hash_id)
    # Lead with a unique 4-5 token marker so two different hash_ids never
    # share leading tokens by chance -- the marker alone separates them in
    # vllm's prefix tree.
    marker = f"block-{hash_id:08d}-start"
    body = " ".join(rng.choice(_STOPWORD_VOCAB) for _ in range(_WORDS_PER_BLOCK))
    text = marker + " " + body
    _BLOCK_CACHE[hash_id] = text
    return text


def session_base_prompt(hash_ids: list[int]) -> str:
    """Concatenated block text for a session's full prefix chain."""
    return "\n".join(block_text(h) for h in hash_ids)


def round_user_query(round_idx: int, hash_ids: list[int]) -> str:
    """Per-round user query: the session's full prefix on round 0, then
    a short delta on subsequent rounds (so prefix accumulates monotonically
    across rounds, matching the agentic burst shape)."""
    if round_idx == 0:
        return session_base_prompt(hash_ids)
    # For rounds 1..N-1 we want the assistant context + a small new query
    # to drive incremental prefill. Use a deterministic short query.
    rng = random.Random(hash_ids[0] * 1000 + round_idx if hash_ids else round_idx)
    n = 16 + rng.randrange(48)  # 16-64 words
    return " ".join(rng.choice(_STOPWORD_VOCAB) for _ in range(n))


# ----- metrics ---------------------------------------------------------

@dataclass
class TurnRec:
    session_id: int
    round_idx: int
    arrive_ts: float = 0.0
    first_token_ts: float = 0.0
    finish_ts: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    error: Optional[str] = None
    itls: list[float] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_ts - self.arrive_ts) * 1000.0 if self.first_token_ts else -1.0

    @property
    def e2e_ms(self) -> float:
        return (self.finish_ts - self.arrive_ts) * 1000.0 if self.finish_ts else -1.0

    @property
    def tpot_ms(self) -> float:
        if self.completion_tokens <= 1 or not self.first_token_ts:
            return -1.0
        decode_s = self.finish_ts - self.first_token_ts
        return decode_s * 1000.0 / max(1, self.completion_tokens - 1)


# ----- request driver --------------------------------------------------

async def call_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    rec: TurnRec,
    request_timeout_s: float,
) -> str:
    """Stream a single chat-completions call, populate rec timings, return
    the assistant text (used to build the next round's chat history)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec.arrive_ts = time.perf_counter()
    out_chunks: list[str] = []
    last_token_ts = 0.0
    try:
        async with session.post(endpoint, json=payload,
                                timeout=aiohttp.ClientTimeout(total=request_timeout_s)) as resp:
            if resp.status != 200:
                body = await resp.text()
                rec.error = f"HTTP {resp.status}: {body[:200]}"
                rec.finish_ts = time.perf_counter()
                return ""
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                # usage chunk (final, with stream_options.include_usage)
                if chunk.get("usage"):
                    u = chunk["usage"]
                    rec.prompt_tokens = int(u.get("prompt_tokens", 0))
                    rec.completion_tokens = int(u.get("completion_tokens", 0))
                    pt_details = u.get("prompt_tokens_details") or {}
                    rec.cached_tokens = int(pt_details.get("cached_tokens", 0))
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if not rec.first_token_ts:
                        rec.first_token_ts = now
                    else:
                        rec.itls.append((now - last_token_ts) * 1000.0)
                    last_token_ts = now
                    out_chunks.append(content)
        rec.finish_ts = time.perf_counter() if last_token_ts == 0.0 else last_token_ts
    except asyncio.TimeoutError:
        rec.error = "timeout"
        rec.finish_ts = time.perf_counter()
    except Exception as e:
        rec.error = f"exc: {type(e).__name__}: {str(e)[:120]}"
        rec.finish_ts = time.perf_counter()
    return "".join(out_chunks)


async def run_session(
    session_idx: int,
    record: dict,
    sem: asyncio.Semaphore,
    http: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    num_rounds: int,
    output_len: int,
    inter_turn_gap_median_ms: float,
    inter_turn_gap_sigma: float,
    shared_system_prompt: str,
    arrive_after_s: float,
    request_timeout_s: float,
    rng: random.Random,
    out_recs: list[TurnRec],
) -> None:
    if arrive_after_s > 0:
        await asyncio.sleep(arrive_after_s)
    hash_ids = record.get("hash_ids", []) or []
    chat: list[dict] = []
    if shared_system_prompt:
        chat.append({"role": "system", "content": shared_system_prompt})

    for r in range(num_rounds):
        chat.append({"role": "user", "content": round_user_query(r, hash_ids)})
        rec = TurnRec(session_id=session_idx, round_idx=r)
        async with sem:
            assistant = await call_chat(http, endpoint, model, chat,
                                        output_len, rec, request_timeout_s)
        out_recs.append(rec)
        if rec.error:
            return  # session aborts on first error
        chat.append({"role": "assistant", "content": assistant})

        if inter_turn_gap_median_ms > 0 and r < num_rounds - 1:
            mu = math.log(inter_turn_gap_median_ms / 1000.0)
            gap_s = rng.lognormvariate(mu, inter_turn_gap_sigma)
            await asyncio.sleep(gap_s)


# ----- summary ---------------------------------------------------------

def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return -1.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(recs: list[TurnRec], wall: float, args: argparse.Namespace) -> dict:
    ok = [r for r in recs if r.error is None and r.first_token_ts]
    err = [r for r in recs if r.error is not None]
    no_token = [r for r in recs if r.error is None and not r.first_token_ts]
    err_samples: list[str] = []
    seen_err: set[str] = set()
    for r in err + no_token:
        msg = r.error or "no_token_returned"
        key = msg.split(":", 1)[0]
        if key not in seen_err and len(err_samples) < 5:
            err_samples.append(f"sess={r.session_id} round={r.round_idx} prompt_tok={r.prompt_tokens}: {msg[:200]}")
            seen_err.add(key)
    ttfts = [r.ttft_ms for r in ok]
    e2es = [r.e2e_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    itls = [v for r in ok for v in r.itls]
    prompt_tot = sum(r.prompt_tokens for r in ok)
    cached_tot = sum(r.cached_tokens for r in ok)
    completion_tot = sum(r.completion_tokens for r in ok)

    # SLO accounting (matching W1's approach)
    ttft_slo = args.ttft_slo_ms
    tpot_slo = args.tpot_slo_ms
    e2e_slo = args.e2e_slo_ms
    slo_met = sum(1 for r in ok
                  if r.ttft_ms <= ttft_slo
                  and r.tpot_ms <= tpot_slo
                  and r.e2e_ms <= e2e_slo)

    summary = {
        "label": args.label,
        "args": vars(args),
        "wall_clock_s": wall,
        "counts": {
            "submitted": len(recs),
            "ok": len(ok),
            "errored": len(err),
            "slo_met": slo_met,
        },
        "throughput": {
            "request_per_s": len(ok) / wall if wall > 0 else 0.0,
            "input_tok_per_s": prompt_tot / wall if wall > 0 else 0.0,
            "output_tok_per_s": completion_tot / wall if wall > 0 else 0.0,
            "total_tok_per_s": (prompt_tot + completion_tot) / wall if wall > 0 else 0.0,
            "goodput_req_per_s": slo_met / wall if wall > 0 else 0.0,
            "slo_attainment_pct": 100.0 * slo_met / max(1, len(ok)),
        },
        "ttft_ms": {
            "mean": statistics.mean(ttfts) if ttfts else -1.0,
            "p50": percentile(ttfts, 0.5),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "tpot_ms": {
            "mean": statistics.mean(tpots) if tpots else -1.0,
            "p50": percentile(tpots, 0.5),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
        },
        "itl_ms": {
            "mean": statistics.mean(itls) if itls else -1.0,
            "p50": percentile(itls, 0.5),
            "p95": percentile(itls, 0.95),
            "p99": percentile(itls, 0.99),
            "samples": len(itls),
        },
        "e2e_ms": {
            "mean": statistics.mean(e2es) if e2es else -1.0,
            "p50": percentile(e2es, 0.5),
            "p95": percentile(e2es, 0.95),
            "p99": percentile(e2es, 0.99),
        },
        "cache": {
            "prompt_tokens_total": prompt_tot,
            "cached_tokens_total": cached_tot,
            "hit_rate_pct": (100.0 * cached_tot / prompt_tot) if prompt_tot else 0.0,
        },
        "concurrency": {"max_in_flight": args.concurrency, "cap": args.concurrency},
        "error_samples": err_samples,
    }
    return summary


# ----- main ------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    # Load trace
    sessions: list[dict] = []
    with open(args.dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    if not sessions:
        print(f"[bench] no sessions in {args.dataset_path}", flush=True)
        return 1
    # Sort by timestamp and take the first N -- matches sglang's mooncake
    # loader behavior (cluster the earliest N arrivals, instead of sampling
    # randomly across the full ~59-min trace which would bloat wall-clock
    # without changing the workload mix).
    sessions.sort(key=lambda r: r.get("timestamp", 0))
    sessions = sessions[: args.num_prompts]

    # Optional shared system prompt
    shared_sys = ""
    if args.shared_system_prompt_path and Path(args.shared_system_prompt_path).exists():
        shared_sys = Path(args.shared_system_prompt_path).read_text()

    # Arrival schedule: take the trace's relative timestamps (ms),
    # divided by slowdown_factor to compress / dilate inter-arrival.
    base_ts = sessions[0].get("timestamp", 0)
    arrivals = [(s.get("timestamp", 0) - base_ts) / 1000.0 / max(args.slowdown_factor, 1e-6)
                for s in sessions]

    sem = asyncio.Semaphore(args.concurrency)
    out_recs: list[TurnRec] = []

    print(f"[bench] label={args.label} sessions={len(sessions)} rounds={args.num_rounds} "
          f"output_len={args.output_len} concurrency={args.concurrency} "
          f"shared_sys={'yes' if shared_sys else 'no'}",
          flush=True)
    print(f"[bench] inter-turn-gap: LogNormal(median={args.inter_turn_gap_median_ms}ms, "
          f"sigma={args.inter_turn_gap_sigma})",
          flush=True)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.concurrency * 2)) as http:
        t0 = time.perf_counter()
        per_session_rng = [random.Random(args.seed * 100003 + i) for i in range(len(sessions))]
        tasks = [
            asyncio.create_task(run_session(
                i, sessions[i], sem, http, args.endpoint, args.model,
                args.num_rounds, args.output_len,
                args.inter_turn_gap_median_ms, args.inter_turn_gap_sigma,
                shared_sys, arrivals[i], args.request_timeout_s,
                per_session_rng[i], out_recs))
            for i in range(len(sessions))
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    summary = summarize(out_recs, wall, args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] wall={wall:.1f}s  ok={summary['counts']['ok']}/{summary['counts']['submitted']}  "
          f"err={summary['counts']['errored']}  "
          f"req/s={summary['throughput']['request_per_s']:.2f}  "
          f"hit%={summary['cache']['hit_rate_pct']:.1f}  "
          f"ttft_p50={summary['ttft_ms']['p50']:.0f}ms  "
          f"e2e_p99={summary['e2e_ms']['p99']:.0f}ms  ->  {args.output}",
          flush=True)
    return 0 if summary['counts']['errored'] == 0 else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://127.0.0.1:30000/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-path", required=True,
                   help="path to mooncake conversation_trace_le6k.jsonl")
    p.add_argument("--num-prompts", type=int, default=30,
                   help="number of sessions to run from the trace (after shuffle)")
    p.add_argument("--num-rounds", type=int, default=4)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--slowdown-factor", type=float, default=1.0,
                   help="divide trace inter-arrival by this; >1 compresses arrivals")
    p.add_argument("--inter-turn-gap-median-ms", type=float, default=50.0)
    p.add_argument("--inter-turn-gap-sigma", type=float, default=0.5)
    p.add_argument("--shared-system-prompt-path", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    p.add_argument("--tpot-slo-ms", type=float, default=200.0)
    p.add_argument("--e2e-slo-ms", type=float, default=60000.0)
    p.add_argument("--label", default="run")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
