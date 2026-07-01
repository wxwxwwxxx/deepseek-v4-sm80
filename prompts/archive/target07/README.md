# TARGET 07 Fine-Grained Prompt Archive

This directory contains old fine-grained TARGET 07 prompt files.

For new Codex threads, do not use this archive as the primary reference by
default.  Prefer the current root-level files:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.10_dsv4_sm80_foundation_history.md`
- `prompts/TARGET_07.20_dsv4_sm80_moe_history.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- the active todo target, currently
  `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`

Use this archive only when:

- a summarized history file points to a specific original target for exact
  command details;
- a milestone artifact references an old prompt by its original filename;
- a debugging task needs the exact original scope, stop conditions, or
  thread-local assumptions.

The files here are historical source material, not the active project map.
