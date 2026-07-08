# vLLM PR: queue-aware eviction

Self-contained, `peek`-free change that adds queue-aware KV cache eviction to
vLLM v1. Authored by **Bing Xie** (@xiexbing). No third-party tooling is
credited in the commit, PR body, or code.

## Contents

- `0001-queue-aware-eviction.patch` — the change (against vLLM **v0.24.0**, 5 files, +130/-1)
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

## Commit (author = Bing Xie, DCO sign-off, no other attribution)

```bash
git -c user.name="Bing Xie" -c user.email="xiexbing@gmail.com" \
    commit -s -m "[Core] Add queue-aware KV cache eviction

Protect KV-cache blocks whose prefix is needed by a request in the waiting
queue: evict unprotected blocks first, then the cheapest-to-recompute
protected blocks. Opt-in via --enable-queue-aware-eviction; default behavior
is unchanged."
```

`-s` adds the `Signed-off-by: Bing Xie <xiexbing@gmail.com>` line vLLM's DCO
check requires. Do **not** add any co-author trailer.

## Before opening the PR

- [ ] Rebase the branch onto current `main` and re-run `git apply` cleanly
- [ ] `pytest tests/v1/core/test_queue_aware_eviction.py -v`
- [ ] `pre-commit run --all-files` (vLLM uses ruff/isort/mypy/yapf)
- [ ] Confirm `vllm serve --help` shows `--enable-queue-aware-eviction`
- [ ] Read `docs/contributing`; large features may warrant an RFC issue first
- [ ] Paste `PR.md` as the PR description
