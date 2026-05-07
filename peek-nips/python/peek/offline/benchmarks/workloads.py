# Copyright 2026 Anonymous Authors
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

"""Real workload generators for Peek benchmarks.

Uses standard LLM evaluation datasets (ShareGPT, MMLU, LooGLE,
HumanEval, MBPP) when a tokenizer and the ``datasets`` library are
available.  Falls back to synthetic token generation when they are not.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peek.offline.prompt import PromptRequest

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"
_VOCAB_SIZE = 32_000  # Llama-2 vocab

_tokenizer_cache: dict[str, Any] = {}


def get_tokenizer() -> Any:
    """Return the Llama-2 tokenizer, or *None* if ``transformers`` is missing."""
    if "tok" in _tokenizer_cache:
        return _tokenizer_cache["tok"]
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        _tokenizer_cache["tok"] = tok
        return tok
    except Exception:
        warnings.warn(
            f"Could not load tokenizer {TOKENIZER_NAME!r}. "
            "Falling back to synthetic token generation.",
            stacklevel=2,
        )
        _tokenizer_cache["tok"] = None
        return None


# ---------------------------------------------------------------------------
# ShareGPT dataset
# ---------------------------------------------------------------------------

_SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)


def _data_dir(data_dir: str = ".data") -> Path:
    return Path(__file__).resolve().parent / data_dir


def load_sharegpt(
    data_dir: str = ".data",
    max_samples: int | None = None,
) -> list[dict]:
    """Load ShareGPT V3 conversations.

    Auto-downloads to ``benchmarks/<data_dir>/`` on first run.
    Returns list of conversation dicts with ``conversations`` key.
    """
    ddir = _data_dir(data_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    fpath = ddir / "ShareGPT_V3_unfiltered_cleaned_split.json"

    if not fpath.exists():
        print(f"Downloading ShareGPT V3 to {fpath} ...")
        try:
            import urllib.request

            urllib.request.urlretrieve(_SHAREGPT_URL, str(fpath))
            print("Download complete.")
        except Exception as exc:
            warnings.warn(f"ShareGPT download failed: {exc}", stacklevel=2)
            return []

    with open(fpath) as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


# ---------------------------------------------------------------------------
# HuggingFace datasets loaders (MMLU, LooGLE, HumanEval, MBPP)
# ---------------------------------------------------------------------------

def _load_dataset_safe(
    path: str,
    name: str | None = None,
    split: str = "test",
) -> list[dict] | None:
    """Load a HuggingFace dataset, returning *None* on any failure."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        try:
            ds = load_dataset(path, name, split=split)
        except TypeError:
            # Older datasets library may need trust_remote_code
            ds = load_dataset(path, name, split=split, trust_remote_code=True)
        return list(ds)
    except Exception as exc:
        warnings.warn(
            f"Could not load dataset {path!r} (name={name!r}, split={split!r}): {exc}",
            stacklevel=2,
        )
        return None


def _format_mmlu_question(row: dict, include_answer: bool = True) -> str:
    """Format a single MMLU row as 'Q: ... A. ... B. ... Answer: X'."""
    choices = row.get("choices", [])
    labels = "ABCD"
    lines = [f"Q: {row['question']}"]
    for i, c in enumerate(choices):
        lbl = labels[i] if i < len(labels) else str(i)
        lines.append(f"  {lbl}. {c}")
    if include_answer:
        answer_idx = row.get("answer", 0)
        if isinstance(answer_idx, int) and answer_idx < len(labels):
            lines.append(f"Answer: {labels[answer_idx]}")
        else:
            lines.append(f"Answer: {answer_idx}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def load_mmlu(
    num_subjects: int = 10,
    max_per_subject: int = 50,
) -> dict[str, dict[str, list[dict]]] | None:
    """Load MMLU dev (for few-shot exemplars) and test (for questions).

    Returns ``{subject: {"dev": [...], "test": [...]}}`` for the top-N
    subjects (by number of test examples), or *None* on failure.
    """
    dev_rows = _load_dataset_safe("cais/mmlu", "all", split="auxiliary_train")
    test_rows = _load_dataset_safe("cais/mmlu", "all", split="test")
    if dev_rows is None or test_rows is None:
        # Try individual subject loading as fallback
        dev_rows = _load_dataset_safe("cais/mmlu", "all", split="dev")
        if dev_rows is None or test_rows is None:
            return None

    # Group by subject
    from collections import defaultdict

    by_subject_dev: dict[str, list[dict]] = defaultdict(list)
    by_subject_test: dict[str, list[dict]] = defaultdict(list)

    for row in dev_rows:
        subj = row.get("subject", "unknown")
        by_subject_dev[subj].append(row)

    for row in test_rows:
        subj = row.get("subject", "unknown")
        by_subject_test[subj].append(row)

    # Pick top-N subjects by test count
    subjects_by_count = sorted(
        by_subject_test.keys(),
        key=lambda s: len(by_subject_test[s]),
        reverse=True,
    )[:num_subjects]

    result: dict[str, dict[str, list[dict]]] = {}
    for subj in subjects_by_count:
        result[subj] = {
            "dev": by_subject_dev.get(subj, [])[:10],  # keep up to 10 dev exemplars
            "test": by_subject_test[subj][:max_per_subject],
        }
    return result


def load_loogle(max_docs: int = 20) -> list[dict] | None:
    """Load LooGLE longdep_qa split.

    Returns list of ``{context, title, questions}`` dicts grouped by
    document, or *None* on failure.
    """
    rows = _load_dataset_safe("bigainlco/LooGLE", "longdep_qa", split="test")
    if rows is None:
        return None

    docs: list[dict] = []
    for row in rows[:max_docs]:
        context = row.get("input", "") or row.get("context", "")
        title = row.get("title", "")
        # Questions can be in 'qa_pairs' (JSON string) or 'output'
        qa_pairs_raw = row.get("qa_pairs", "")
        questions: list[str] = []
        if qa_pairs_raw:
            try:
                qa_pairs = json.loads(qa_pairs_raw) if isinstance(qa_pairs_raw, str) else qa_pairs_raw
                if isinstance(qa_pairs, list):
                    for qa in qa_pairs:
                        q = qa.get("Q", "") or qa.get("question", "")
                        if q:
                            questions.append(q)
            except (json.JSONDecodeError, TypeError):
                pass
        if not questions:
            output = row.get("output", "")
            if output:
                questions.append(output)
        if context and questions:
            docs.append({"context": context, "title": title, "questions": questions})

    return docs if docs else None


def load_humaneval_mbpp() -> list[dict] | None:
    """Load HumanEval + MBPP prompts.

    Returns list of ``{prompt, source}`` dicts, or *None* on failure.
    """
    results: list[dict] = []

    he_rows = _load_dataset_safe("openai/openai_humaneval", None, split="test")
    if he_rows is not None:
        for row in he_rows:
            prompt = row.get("prompt", "")
            if prompt:
                results.append({"prompt": prompt, "source": "humaneval"})

    mbpp_rows = _load_dataset_safe(
        "google-research-datasets/mbpp", "sanitized", split="test",
    )
    if mbpp_rows is not None:
        for row in mbpp_rows:
            text = row.get("prompt", "") or row.get("text", "")
            code = row.get("code", "")
            # Build a prompt: description + function signature
            if text:
                prompt = text
                if code:
                    # Extract first line (signature) from solution
                    first_line = code.split("\n")[0]
                    if first_line.strip().startswith("def "):
                        prompt = f"# {text}\n{first_line}\n"
                results.append({"prompt": prompt, "source": "mbpp"})

    return results if results else None


# ---------------------------------------------------------------------------
# Helpers: synthetic fallback tokens
# ---------------------------------------------------------------------------

# ShareGPT token-length statistics (empirical from full dataset):
#   mean ~161, median ~87, p95 ~580, max ~2048 (after truncation)
_SHAREGPT_LEN_PARAMS = {"mu": 4.4, "sigma": 1.1}  # log-normal fit


def _sample_length(rng: random.Random, min_len: int = 4, max_len: int = 512) -> int:
    """Sample a prompt length from a log-normal distribution."""
    raw = int(rng.lognormvariate(**_SHAREGPT_LEN_PARAMS))
    return max(min_len, min(raw, max_len))


def _random_tokens(rng: random.Random, length: int) -> list[int]:
    """Generate random token IDs in [1, VOCAB_SIZE)."""
    return [rng.randint(1, _VOCAB_SIZE - 1) for _ in range(length)]


def _tokenize_or_fake(
    text: str,
    tokenizer: Any | None,
    rng: random.Random,
    max_len: int = 512,
) -> list[int]:
    """Tokenize *text* if a tokenizer is available, else fake tokens."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
        if len(ids) >= 4:
            return ids
    length = _sample_length(rng)
    return _random_tokens(rng, length)


def _deterministic_prefix(seed: str, length: int) -> list[int]:
    """Deterministic pseudo-random prefix from a seed string."""
    h = hashlib.sha256(seed.encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    return _random_tokens(rng, length)


def _zipf_group_assignments(
    n: int,
    num_groups: int,
    alpha: float = 1.0,
    rng: random.Random | None = None,
) -> list[int]:
    """Assign *n* requests to groups following a Zipf distribution.

    Alpha=1.0 gives the classic 80/20 skew.  Higher alpha = more skew.
    Returns list of group indices (0..num_groups-1).
    """
    if rng is None:
        rng = random.Random(99)
    # Compute unnormalised Zipf weights: w_k = 1 / k^alpha
    weights = [1.0 / (k + 1) ** alpha for k in range(num_groups)]
    total = sum(weights)
    cdf = []
    cumulative = 0.0
    for w in weights:
        cumulative += w / total
        cdf.append(cumulative)

    assignments: list[int] = []
    for _ in range(n):
        u = rng.random()
        for g, c in enumerate(cdf):
            if u <= c:
                assignments.append(g)
                break
        else:
            assignments.append(num_groups - 1)
    return assignments


def _sample_prefix_lengths(
    num_groups: int,
    base_len: int,
    sigma: float = 0.3,
    rng: random.Random | None = None,
) -> list[int]:
    """Sample per-group prefix lengths from a log-normal distribution.

    Centred on *base_len* with multiplicative spread *sigma*.
    sigma=0.3 gives roughly ±30% variation.  Returns list of int lengths.
    """
    if rng is None:
        rng = random.Random(100)
    mu = math.log(base_len)
    lengths = []
    for _ in range(num_groups):
        raw = int(rng.lognormvariate(mu, sigma))
        # Clamp to [base_len//4, base_len*4] to avoid degenerate extremes
        raw = max(base_len // 4, min(raw, base_len * 4))
        lengths.append(raw)
    return lengths


# ---------------------------------------------------------------------------
# PromptRequest factory
# ---------------------------------------------------------------------------

@dataclass
class _PromptFactory:
    """Stateful factory to produce PromptRequests with unique IDs."""
    _counter: int = 0

    def make(
        self,
        token_ids: list[int],
        *,
        group: str = "",
        metadata: dict | None = None,
    ) -> PromptRequest:
        self._counter += 1
        md = metadata or {}
        if group:
            md["group"] = group
        return PromptRequest(
            id=f"req-{self._counter}",
            token_ids=token_ids,
            metadata=md,
        )


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def shared_system_prompts(
    n: int,
    num_groups: int = 5,
    system_prompt_len: int = 2048,
    question_len: int = 128,
    tokenizer: Any | None = ...,
    group_distribution: str = "uniform",
    zipf_alpha: float = 1.0,
    prefix_len_sigma: float = 0.0,
) -> list[PromptRequest]:
    """Long-context QA using LooGLE documents as shared prefixes.

    ~90%+ prefix sharing: each request shares a long document context with
    its group, then has a unique question appended.

    Parameters
    ----------
    group_distribution : ``"uniform"`` (round-robin, default) or ``"zipf"``
        (Zipf-distributed group popularity with exponent *zipf_alpha*).
    zipf_alpha : float
        Zipf exponent when *group_distribution* is ``"zipf"``.
        1.0 = classic 80/20, higher = more skew.
    prefix_len_sigma : float
        If > 0, per-group prefix lengths are sampled from LogNormal centred
        on *system_prompt_len* with this spread.  0.0 (default) = all groups
        use exactly *system_prompt_len*.

    Falls back to synthetic deterministic prefixes when *tokenizer* is None
    or the LooGLE dataset is unavailable.
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(42)
    factory = _PromptFactory()

    # Per-group prefix lengths (fixed or variable)
    if prefix_len_sigma > 0:
        group_lengths = _sample_prefix_lengths(
            num_groups, system_prompt_len, sigma=prefix_len_sigma,
            rng=random.Random(101),
        )
    else:
        group_lengths = [system_prompt_len] * num_groups

    # Try loading real LooGLE data
    loogle_docs = load_loogle(max_docs=num_groups) if tokenizer is not None else None

    system_prompts: list[list[int]] = []
    group_questions: list[list[list[int]]] = []  # per-group list of tokenized questions

    if loogle_docs and tokenizer is not None:
        n_real_docs = len(loogle_docs)
        for g in range(num_groups):
            sp_len = group_lengths[g]
            if g < n_real_docs:
                # Use real LooGLE document
                doc = loogle_docs[g]
                context_text = doc["context"]
                sp = tokenizer.encode(context_text, add_special_tokens=False)[:sp_len]
            else:
                # Beyond available docs: create unique prefix by prepending
                # a group-specific salt to a recycled document
                doc = loogle_docs[g % n_real_docs]
                header = f"[Tenant {g}] System context:\n"
                header_tokens = tokenizer.encode(header, add_special_tokens=False)
                doc_tokens = tokenizer.encode(doc["context"], add_special_tokens=False)
                sp = (header_tokens + doc_tokens)[:sp_len]
            system_prompts.append(sp)
            # Tokenize the real questions for this document
            qs: list[list[int]] = []
            for q_text in doc["questions"]:
                q_tokens = tokenizer.encode(q_text, add_special_tokens=False)[:question_len]
                if len(q_tokens) >= 4:
                    qs.append(q_tokens)
            if not qs:
                qs.append(_random_tokens(rng, question_len))
            group_questions.append(qs)
    else:
        # Synthetic fallback
        for g in range(num_groups):
            sp = _deterministic_prefix(f"system-{g}", group_lengths[g])
            system_prompts.append(sp)
            group_questions.append([])  # empty -> use random tokens below

    # Group assignment: uniform (round-robin) or Zipf
    if group_distribution == "zipf":
        assignments = _zipf_group_assignments(
            n, num_groups, alpha=zipf_alpha, rng=random.Random(102),
        )
    else:
        assignments = [i % num_groups for i in range(n)]

    # Track per-group request count for question cycling
    group_counters: list[int] = [0] * num_groups

    prompts: list[PromptRequest] = []
    for i in range(n):
        g = assignments[i]
        qs = group_questions[g]
        q_idx = group_counters[g]
        group_counters[g] += 1
        if qs:
            question = qs[q_idx % len(qs)]
        else:
            question = _random_tokens(rng, question_len)
        token_ids = system_prompts[g] + question
        prompts.append(factory.make(token_ids, group=f"sys-{g}"))
    return prompts


def few_shot_mmlu(
    n: int,
    num_subjects: int = 10,
    num_shots: int = 5,
    tokenizer: Any | None = ...,
) -> list[PromptRequest]:
    """5-shot MMLU evaluation prompts using real MMLU data.

    Very high prefix sharing: per-subject prefix with header + 5 dev
    exemplars, unique test question per request.

    Falls back to synthetic pattern when *tokenizer* is None or the
    MMLU dataset is unavailable.
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(43)
    factory = _PromptFactory()

    # Try loading real MMLU data
    mmlu_data = load_mmlu(num_subjects=num_subjects) if tokenizer is not None else None

    shot_prefixes: list[list[int]] = []
    subject_questions: list[list[list[int]]] = []  # per-subject tokenized test Qs
    subject_names: list[str] = []

    if mmlu_data and tokenizer is not None:
        for subj, splits in mmlu_data.items():
            subject_names.append(subj)
            # Build prefix: header + num_shots dev exemplars
            header = (
                f"The following are multiple choice questions (with answers) "
                f"about {subj.replace('_', ' ')}.\n\n"
            )
            prefix_text = header
            dev_rows = splits["dev"][:num_shots]
            for row in dev_rows:
                prefix_text += _format_mmlu_question(row, include_answer=True) + "\n\n"
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            shot_prefixes.append(prefix_tokens)

            # Tokenize test questions
            qs: list[list[int]] = []
            for row in splits["test"]:
                q_text = _format_mmlu_question(row, include_answer=False)
                q_tokens = tokenizer.encode(q_text, add_special_tokens=False)
                if len(q_tokens) >= 4:
                    qs.append(q_tokens)
            if not qs:
                qs.append(_random_tokens(rng, 40))
            subject_questions.append(qs)
    else:
        # Synthetic fallback
        for s in range(num_subjects):
            subject_names.append(f"subject_{s}")
            shots: list[int] = []
            for k in range(num_shots):
                shots.extend(_deterministic_prefix(f"mmlu-{s}-shot-{k}", 60))
            shot_prefixes.append(shots[:600])
            subject_questions.append([])  # empty -> use random tokens below

    actual_subjects = len(shot_prefixes)
    prompts: list[PromptRequest] = []
    for i in range(n):
        s = i % actual_subjects
        qs = subject_questions[s]
        if qs:
            question = qs[i // actual_subjects % len(qs)]
        else:
            question = _random_tokens(rng, 40)
        token_ids = shot_prefixes[s] + question
        prompts.append(factory.make(token_ids, group=f"mmlu-{subject_names[s]}"))
    return prompts


def multi_turn_chat(
    n: int,
    turns_per_conv: int = 6,
    context_len: int = 2048,
    tokenizer: Any | None = ...,
) -> list[PromptRequest]:
    """Multi-turn chat using ShareGPT conversations.

    Turn N shares all of turns 0..N-1 as prefix, giving moderate-high
    prefix sharing within each conversation.

    Filters for conversations with at least *turns_per_conv* turns when
    real data is available.
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(44)
    factory = _PromptFactory()

    sharegpt_data = load_sharegpt(max_samples=2000) if tokenizer else []
    # Filter for conversations with enough turns
    if sharegpt_data:
        long_convs = [
            c for c in sharegpt_data
            if len(c.get("conversations", [])) >= turns_per_conv
        ]
        # Fall back to all data if not enough long conversations
        if len(long_convs) >= 10:
            sharegpt_data = long_convs

    prompts: list[PromptRequest] = []
    conv_idx = 0

    while len(prompts) < n:
        context: list[int] = []
        num_turns = min(turns_per_conv, n - len(prompts))

        for turn in range(num_turns):
            if sharegpt_data and conv_idx < len(sharegpt_data):
                conv = sharegpt_data[conv_idx % len(sharegpt_data)]
                turns_list = conv.get("conversations", [])
                if turn < len(turns_list):
                    text = turns_list[turn].get("value", "")
                    turn_tokens = _tokenize_or_fake(text, tokenizer, rng, max_len=512)
                else:
                    turn_tokens = _random_tokens(rng, rng.randint(30, 200))
            else:
                turn_tokens = _random_tokens(rng, rng.randint(30, 200))

            context = context + turn_tokens
            if len(context) > context_len:
                context = context[:context_len]

            prompts.append(factory.make(
                list(context),
                group=f"conv-{conv_idx}",
                metadata={"turn": turn},
            ))
            if len(prompts) >= n:
                break

        conv_idx += 1

    return prompts[:n]


def code_completion(
    n: int,
    max_prompt_len: int = 512,
    tokenizer: Any | None = ...,
) -> list[PromptRequest]:
    """Code completion using HumanEval + MBPP prompts.

    Low-moderate prefix sharing: real function signatures and docstrings.

    Falls back to synthetic template-based generation when *tokenizer* is
    None or datasets are unavailable.
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(45)
    factory = _PromptFactory()

    # Try loading real code datasets
    code_data = load_humaneval_mbpp() if tokenizer is not None else None

    if code_data and tokenizer is not None:
        prompts: list[PromptRequest] = []
        for i in range(n):
            entry = code_data[i % len(code_data)]
            token_ids = tokenizer.encode(
                entry["prompt"], add_special_tokens=False,
            )[:max_prompt_len]
            if len(token_ids) < 4:
                token_ids = _random_tokens(rng, _sample_length(rng, max_len=128))
            prompts.append(factory.make(
                token_ids,
                group="code",
                metadata={"source": entry.get("source", "unknown")},
            ))
        return prompts

    # Synthetic fallback
    _TEMPLATES = [
        "def {name}({args}):\n    \"\"\"",
        "class {name}:\n    def __init__(self, {args}):\n        ",
        "async def {name}({args}) -> {ret}:\n    ",
        "def {name}({args}) -> {ret}:\n    # ",
    ]
    _NAMES = [
        "sort_array", "find_max", "merge_lists", "binary_search",
        "parse_json", "validate_email", "compute_hash", "flatten_tree",
        "build_graph", "cache_lookup", "encode_base64", "decode_token",
        "transform_data", "filter_results", "aggregate_scores",
    ]
    _ARGS = ["arr", "data", "items, key", "s, pattern", "node, depth", "x, y, z"]
    _RETS = ["list", "int", "bool", "str", "dict", "None"]

    prompts = []
    for i in range(n):
        tmpl = rng.choice(_TEMPLATES)
        text = tmpl.format(
            name=rng.choice(_NAMES),
            args=rng.choice(_ARGS),
            ret=rng.choice(_RETS),
        )
        token_ids = _tokenize_or_fake(text, tokenizer, rng, max_len=128)
        if rng.random() < 0.2 and prompts:
            donor = rng.choice(prompts)
            shared_len = rng.randint(4, min(20, len(donor.token_ids)))
            token_ids = donor.token_ids[:shared_len] + token_ids
        prompts.append(factory.make(token_ids[:max_prompt_len], group="code"))
    return prompts


def single_turn_diverse(
    n: int,
    max_prompt_len: int = 1024,
    tokenizer: Any | None = ...,
) -> list[PromptRequest]:
    """One-off unique instructions from ShareGPT first turns.

    Low prefix sharing (worst case for KV cache).
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(46)
    factory = _PromptFactory()

    sharegpt_data = load_sharegpt(max_samples=5000) if tokenizer else []
    prompts: list[PromptRequest] = []
    for i in range(n):
        if sharegpt_data and i < len(sharegpt_data):
            conv = sharegpt_data[i]
            turns = conv.get("conversations", [])
            text = turns[0].get("value", "") if turns else ""
            token_ids = _tokenize_or_fake(text, tokenizer, rng, max_len=max_prompt_len)
        else:
            token_ids = _random_tokens(rng, _sample_length(rng))
        prompts.append(factory.make(token_ids, group="diverse"))
    rng.shuffle(prompts)
    return prompts


def rag_chunked_overlap(
    n: int,
    num_groups: int = 5,
    doc_len: int = 8192,
    chunk_len: int = 4096,
    question_len: int = 128,
    overlap_ratio: float = 0.8,
    tokenizer: Any | None = ...,
    group_distribution: str = "uniform",
    zipf_alpha: float = 1.0,
) -> list[PromptRequest]:
    """RAG workload with partial prefix overlap within each document group.

    Unlike ``shared_system_prompts`` (100% prefix alignment), this simulates
    real retrieval-augmented generation where each request shares a common
    document prefix but retrieves a *different passage* for the tail.

    The first ``chunk_len * overlap_ratio`` tokens are shared across all
    requests in a group (the document header / shared context).  The
    remaining ``chunk_len * (1 - overlap_ratio)`` tokens are drawn from
    different regions of the document, simulating different retrieved
    passages.  This produces a shared *prefix* (cacheable) plus a
    divergent *suffix* (unique per request).

    Parameters
    ----------
    overlap_ratio : float
        Fraction of ``chunk_len`` that is a shared prefix.  0.8 means the
        first 80% of tokens are identical within a group, and the last 20%
        differ per request.
    group_distribution : ``"uniform"`` or ``"zipf"``
    """
    if tokenizer is ...:
        tokenizer = None
    rng = random.Random(50)
    factory = _PromptFactory()

    shared_len = int(chunk_len * overlap_ratio)
    suffix_len = chunk_len - shared_len

    # Build per-group documents (source material for shared prefix + suffix pool)
    loogle_docs = load_loogle(max_docs=num_groups) if tokenizer is not None else None
    documents: list[list[int]] = []

    for g in range(num_groups):
        if loogle_docs and tokenizer is not None and g < len(loogle_docs):
            doc = loogle_docs[g]
            doc_tokens = tokenizer.encode(doc["context"], add_special_tokens=False)
            if len(doc_tokens) < doc_len:
                doc_tokens = doc_tokens + _random_tokens(rng, doc_len - len(doc_tokens))
            documents.append(doc_tokens[:doc_len])
        else:
            documents.append(_deterministic_prefix(f"rag-doc-{g}", doc_len))

    # Per-group: shared prefix (first shared_len tokens of document)
    # and suffix pool (remaining tokens, sliced into suffix_len chunks)
    group_prefixes: list[list[int]] = []
    group_suffix_pools: list[list[list[int]]] = []
    for g in range(num_groups):
        group_prefixes.append(documents[g][:shared_len])
        # Build diverse suffixes from the rest of the document
        remaining = documents[g][shared_len:]
        suffixes: list[list[int]] = []
        for start in range(0, len(remaining) - suffix_len + 1, suffix_len):
            suffixes.append(remaining[start : start + suffix_len])
        if not suffixes:
            suffixes.append(_random_tokens(rng, suffix_len))
        group_suffix_pools.append(suffixes)

    # Group assignment
    if group_distribution == "zipf":
        assignments = _zipf_group_assignments(
            n, num_groups, alpha=zipf_alpha, rng=random.Random(103),
        )
    else:
        assignments = [i % num_groups for i in range(n)]

    group_counters: list[int] = [0] * num_groups

    prompts: list[PromptRequest] = []
    for i in range(n):
        g = assignments[i]
        idx = group_counters[g]
        group_counters[g] += 1

        prefix = group_prefixes[g]
        suffix = group_suffix_pools[g][idx % len(group_suffix_pools[g])]
        question = _random_tokens(rng, question_len)
        token_ids = prefix + suffix + question

        prompts.append(factory.make(
            token_ids,
            group=f"rag-{g}",
            metadata={"shared_prefix_len": shared_len, "suffix_len": len(suffix)},
        ))
    return prompts


def mixed_traffic(
    n: int,
    shared_fraction: float = 0.7,
    system_prompt_len: int = 2048,
    num_groups: int = 5,
    tokenizer: Any | None = ...,
    group_distribution: str = "uniform",
    zipf_alpha: float = 1.0,
) -> list[PromptRequest]:
    """Realistic production mix: 70% shared system prompt + 30% diverse."""
    if tokenizer is ...:
        tokenizer = None

    n_shared = int(n * shared_fraction)
    n_diverse = n - n_shared
    shared = shared_system_prompts(
        n_shared, num_groups=num_groups, system_prompt_len=system_prompt_len,
        tokenizer=tokenizer, group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
    )
    diverse = single_turn_diverse(n_diverse, tokenizer=tokenizer)

    # Re-number IDs to avoid collisions between the two sub-generators
    # (each creates its own _PromptFactory starting at counter=0)
    for i, p in enumerate(shared + diverse):
        p.id = f"mix-{i + 1}"

    combined = shared + diverse
    rng = random.Random(47)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Arrival patterns
# ---------------------------------------------------------------------------

def poisson_arrivals(
    prompts: list[PromptRequest],
    rate: float,
) -> list[list[PromptRequest]]:
    """Split prompts into arrival waves via Poisson process.

    *rate* is the expected number of requests per wave (batch interval).
    Returns list of waves, each a list of PromptRequests.
    """
    rng = random.Random(48)
    waves: list[list[PromptRequest]] = []
    idx = 0
    while idx < len(prompts):
        count = max(1, rng.poisson(rate) if hasattr(rng, "poisson") else int(rng.expovariate(1.0 / rate)))
        count = min(count, len(prompts) - idx)
        waves.append(prompts[idx : idx + count])
        idx += count
    return waves


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_request_dicts(prompts: list[PromptRequest]) -> list[dict]:
    """Convert PromptRequests to request dicts: {id, token_ids, ...}."""
    return [
        {"id": p.id, "token_ids": p.token_ids, **p.metadata}
        for p in prompts
    ]


# Backward-compatible alias
to_sglang_dicts = to_request_dicts


def generate_cached_prefixes(
    prompts: list[PromptRequest],
    coverage: float = 0.3,
    prefix_sample_len: int = 64,
) -> list[tuple[int, ...]]:
    """Sample prefixes from *prompts* to simulate an existing KV cache.

    Returns a list of tuples (prefix token sequences) representing cached
    KV state, covering approximately *coverage* fraction of unique prefixes.
    """
    rng = random.Random(49)
    seen: set[tuple[int, ...]] = set()
    for p in prompts:
        plen = min(prefix_sample_len, len(p.token_ids))
        if plen >= 4:
            seen.add(tuple(p.token_ids[:plen]))

    all_prefixes = list(seen)
    rng.shuffle(all_prefixes)
    k = max(1, int(len(all_prefixes) * coverage))
    return all_prefixes[:k]


# ---------------------------------------------------------------------------
# All workload names for iteration
# ---------------------------------------------------------------------------

WORKLOAD_GENERATORS = {
    "shared_system_prompts": shared_system_prompts,
    "few_shot_mmlu": few_shot_mmlu,
    "multi_turn_chat": multi_turn_chat,
    "code_completion": code_completion,
    "single_turn_diverse": single_turn_diverse,
    "mixed_traffic": mixed_traffic,
    "rag_chunked_overlap": rag_chunked_overlap,
}
