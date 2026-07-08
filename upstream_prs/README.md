# Upstream PRs — queue-aware KV cache eviction

Standalone, upstream-ready contributions extracted from the PEEK project's
offline queue-aware eviction. Each is a self-contained feature with **no
dependency on the `peek` package** — the scoring that PEEK delegated to
`peek.offline.scheduler` is inlined into the engine. Authored by **Bing Xie**;
no other tooling is attributed anywhere in the commits, PR bodies, or code.

| PR | Target | Files | Size |
|---|---|---|---|
| [`sglang/`](sglang/) | sgl-project/sglang @ v0.5.14 | 5 | +94 / -1 |
| [`vllm/`](vllm/) | vllm-project/vllm @ v0.24.0 | 5 | +130 / -1 |

Both target the **latest** release (not PEEK's pinned 0.5.9 / 0.19.1) and apply
cleanly with `git apply`. Each folder has the patch, an upstream-style CPU-only
test, a `PR.md` (title + body), and a `README.md` with apply / commit / DCO
steps.

## Scope

These PRs cover **queue-aware eviction only**. PEEK's cache-aware *scheduling*
reorder (the `PeekEngine` / DFS-reorder hooks, with profiling and tuning
constants) is intentionally excluded — it is more invasive and belongs in a
separate RFC + PR so each change stays small and independently reviewable.

## Not done yet (your call)

- Opening the actual PRs on GitHub (kept fully local per your instruction).
- The two remaining PEEK pieces discussed: the tiny `bench_serving` icebreaker
  patch, and the online Rust-backed cLPM scheduler (needs an RFC first).
