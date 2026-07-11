# TARGET 12 Prompt Archive

This directory contains historical TARGET 12 execution prompts.

For new Codex threads, do not use this archive as the primary project map.
Prefer the root-level route and release baseline:

- `prompts/target.md`
- `prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md`
- `prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md`

TARGET 12 is release-complete at `v0.0.0`. Use archived files only when:

- the TARGET 12 closeout references a specific historical prompt;
- a milestone artifact needs exact commands, contracts, or stop rules;
- a regression depends on an old metadata, SWA, chunked-prefill, C128, CUDA
  graph, or recipe decision;
- future long-context algorithm research needs the TARGET 12.61 attribution
  harness and source-parity record.

Important historical groups:

- `TARGET_12.4` through `12.47`: SGLang-style in-graph metadata preparation
  and promotion;
- `TARGET_12.49` through `12.53`: release bundle, SWA independent lifecycle,
  and HC temporary cleanup;
- `TARGET_12.54` through `12.595`: chunked prefill, bounded indexer, C128
  one-surface metadata, and 1M capability;
- `TARGET_12.597` through `12.606`: max-sequence semantics, CUDA graph reserve,
  padding correctness, recipe selection, and release promotion;
- `TARGET_12.61`: long-context TTFT owner attribution and future algorithmic
  research handoff.

Some archived prompts describe blockers or planned work that were resolved by
later TARGET 12 stages. Treat their status as historical. The root TARGET 12
summary, the v0.0.0 baseline, and later milestone reports are authoritative.

The files here are historical source material, not active todos.
