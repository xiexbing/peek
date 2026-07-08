# vLLM PR: queue-aware eviction

Self-contained, `peek`-free change that adds queue-aware KV cache eviction to
vLLM v1. Authored by the **PEEK paper authors** — Bing Xie (@xiexbing), Zhipeng
Wang, Masahiro Tanaka, Zhen Zheng. No build/AI tooling is credited in the
commit, PR body, or code.

## Contents

- `0001-queue-aware-eviction.patch` — the change (against vLLM **v0.24.0**, 5 files, +139/-1)
- `test_queue_aware_eviction.py` — CPU-only test (goes to `tests/v1/core/`)
- `PR.md` — PR title + body

## Apply to a fresh vLLM checkout

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout -b queue-aware-eviction v0.24.0      # then rebase onto main before opening
git apply --index /path/to/0001-queue-aware-eviction.patch
cp /path/to/test_queue_aware_eviction.py tests/v1/core/test_queue_aware_eviction.py
git add tests/v1/core/test_queue_aware_eviction.py
```

## Commit (authors = PEEK paper authors, DCO sign-off)

```bash
git -c user.name="Bing Xie" -c user.email="xiexbing@gmail.com" \
    commit -s -m "[Core] Add queue-aware KV cache eviction

Protect KV-cache blocks whose prefix is needed by a request in the waiting
queue: evict unprotected blocks first, then the cheapest-to-recompute
protected blocks. Opt-in via --enable-queue-aware-eviction; default behavior
is unchanged.

Reference: PEEK (Xie et al., 2026), https://github.com/xiexbing/peek

Co-authored-by: Zhipeng Wang <REPLACE-WITH-EMAIL>
Co-authored-by: Masahiro Tanaka <REPLACE-WITH-EMAIL>
Co-authored-by: Zhen Zheng <REPLACE-WITH-EMAIL>"
```

`-s` adds the `Signed-off-by: Bing Xie <xiexbing@gmail.com>` line vLLM's DCO
check requires. The `Co-authored-by:` trailers credit the other PEEK paper
authors as co-authors of the PR — **replace each `<REPLACE-WITH-EMAIL>` with the
person's real email** so GitHub links them. Note: some maintainers ask every
listed author to add their own `Signed-off-by:` line for DCO; if so, each
co-author appends one before the PR is opened.

## Before opening the PR

- [ ] Rebase the branch onto current `main` and re-run `git apply` cleanly
- [ ] `pytest tests/v1/core/test_queue_aware_eviction.py -v`
- [ ] `pre-commit run --all-files` (vLLM uses ruff/isort/mypy/yapf)
- [ ] Confirm `vllm serve --help` shows `--enable-queue-aware-eviction`
- [ ] Read `docs/contributing`; large features may warrant an RFC issue first
- [ ] Paste `PR.md` as the PR description
