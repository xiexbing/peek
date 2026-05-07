# W4 data

This directory bundles the data files PEEK W4 (paper §4.4) needs:

- **`conversation_trace_le6k.jsonl`** (629 KB) -- Mooncake `conversation_trace`
  filtered to sessions with cumulative token count ≤ 6 k. Pre-filtered for
  speed; the unfiltered trace is roughly 5x larger and adds runtime without
  changing PEEK-vs-baseline ratios.
- **`shared_system_prompt.txt`** (6.5 KB, 1402 tokens) -- agentic system
  prompt prepended to every session in the `agentic_shared` variant
  (paper §4.4). Modelled on Cursor / Copilot / Claude Code patterns.

The full Mooncake trace is the `conversation_trace.jsonl` from
[FAST'25 Mooncake release](https://github.com/kvcache-ai/Mooncake);
download it there and filter as you like if you want the unfiltered
workload.

## License

Mooncake `conversation_trace` is released under the Mooncake repo's
license; check the upstream README before redistributing. The
`shared_system_prompt.txt` is original to PEEK and Apache 2.0 with the
rest of this repo.
