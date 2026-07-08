# sglang PR: queue-aware eviction

Self-contained, `peek`-free change that adds a `queue-aware` radix-cache
eviction policy to sglang. Authored by **Bing Xie** (@xiexbing). No third-party
tooling is credited in the commit, PR body, or code.

## Contents

- `0001-queue-aware-eviction.patch` — the change (against sglang **v0.5.14**, 5 files, +94/-1)
- `test_queue_aware_eviction.py` — CPU-only test (goes to `test/srt/`)
- `PR.md` — PR title + body

## Apply to a fresh sglang checkout

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
git checkout -b queue-aware-eviction v0.5.14      # then rebase onto main before opening
git apply --index /path/to/0001-queue-aware-eviction.patch
cp /path/to/test_queue_aware_eviction.py test/srt/test_queue_aware_eviction.py
git add test/srt/test_queue_aware_eviction.py
```

## Commit (author = Bing Xie, DCO sign-off, no other attribution)

```bash
git -c user.name="Bing Xie" -c user.email="xiexbing@gmail.com" \
    commit -s -m "feat(radix): add queue-aware KV cache eviction policy

Protect radix-cache blocks that requests in the waiting queue will reuse:
evict unreferenced blocks first, then the cheapest-to-recompute referenced
blocks. Opt-in via --radix-eviction-policy queue-aware; default behavior is
unchanged."
```

`-s` adds the `Signed-off-by: Bing Xie <xiexbing@gmail.com>` line the sglang
DCO check requires. Do **not** add any co-author trailer.

## Before opening the PR

- [ ] Rebase the branch onto current `main` and re-run `git apply` cleanly
- [ ] `python -m pytest test/srt/test_queue_aware_eviction.py -v`
- [ ] `pre-commit run --all-files` (sglang uses black/isort/ruff)
- [ ] Read `CONTRIBUTING.md`; PR title follows their conventional-commit style
- [ ] Paste `PR.md` as the PR description
