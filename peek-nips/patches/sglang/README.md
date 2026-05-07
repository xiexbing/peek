# sglang patches for W4 (Mooncake conversation_trace replay)

Two small additions to `sglang.bench_serving.get_mooncake_request_over_time`,
both env-controlled and **no-op when their env vars are unset** (so applying
this patch is safe and behavior is identical to upstream sglang otherwise).

| env var | what it does | used by |
|---|---|---|
| `PEEK_AGENT_INTER_TURN_MEDIAN_MS` (default 0) | LogNormal-distributed sleep between turns within a Mooncake session -- models tool-chain / RAG inter-turn delay | W4 |
| `PEEK_AGENT_INTER_TURN_SIGMA` (default 0.5) | LogNormal sigma for the gap above | W4 |
| `PEEK_SHARED_SYSTEM_PROMPT_PATH` (default empty) | path to a text file whose contents are prepended as a `system` message to every session | W4 (`agentic_shared`) |

Pinned against **sglang 0.5.9**.

The W4 SGLang driver (`benchmarks/w4/run_w4_sglang.sh`) **requires** this
patch -- without it, the `PEEK_AGENT_INTER_TURN_*` and
`PEEK_SHARED_SYSTEM_PROMPT_PATH` env vars the driver sets are silently
ignored, and the run does not match the paper's W4 protocol.

## Apply

The patch is applied automatically by `scripts/install_peek_sglang.sh`
(unless `SKIP_BENCH_SERVING_PATCH=1` is set). To apply it manually:

```bash
cd <peek-repo-root>
SGLANG_BENCH_SERVING="$(python3 -c 'import sglang.bench_serving as m; print(m.__file__)')"
patch -p0 "$SGLANG_BENCH_SERVING" < patches/sglang/bench_serving.patch
```

To verify the patch landed:

```bash
grep -c "PEEK PATCH" "$SGLANG_BENCH_SERVING"   # expect: 6
```

## Revert

```bash
patch -R -p0 "$SGLANG_BENCH_SERVING" < patches/sglang/bench_serving.patch
```

## Why a patch and not a wrapper?

`get_mooncake_request_over_time` is an `async generator` invoked deep inside
`sglang.bench_serving.main()` -- replacing it cleanly via monkey-patch from
outside requires duplicating sglang's argument-parsing surface and is fragile
across sglang versions. A vendored patch is small (~30 lines), explicit, and
easy to audit.
