# TARGET 11 Prompt Archive

This directory contains historical MTP speculative decoding TARGET 11 prompts.

For new Codex threads, do not use this archive as the primary project map.
Prefer the root-level route files:

- `prompts/target.md`
- `prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/TARGET_09_dsv4_sm80_low_precision_research.md`

TARGET 11 is paused and will not be restarted for `v0.0.0`.  The root TARGET
11 report contains the authoritative postmortem and future restart contract.
The archived files preserve the MTP investigation trail, including
target-verify parity, C128/SWA/KV state lifecycle checks, row-shape-sensitive
kernel probes, and the final pivot decision.

Use archived files only when:

- consulting a future, post-release MTP restart on a new dedicated branch;
- consulting exact historical commands, debug env vars, or stop rules;
- comparing future SGLang-aligned target-verify work against the failed local
  owner-chase route.  Do not use `dsv4-mtp-paused-reference` as the base of the
  future implementation; it is an oracle and evidence source only.

The files here are historical source material, not active todos.
