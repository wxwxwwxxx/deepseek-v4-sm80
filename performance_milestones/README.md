# Performance Milestones

This directory records performance milestones for the DeepSeek V4 sm80/A100
work. Each milestone should keep:

- a short conclusion document;
- copied lightweight summaries, configs, matrices, and derived CSV reports;
- symlinks to large raw artifacts under `/tmp` when the raw files are too large
  to copy into the repository.

Large symlinked artifacts are not durable if `/tmp` is cleaned. Copy them to a
persistent store before deleting the original run directory.

## Milestones

| Milestone | Summary |
| --- | --- |
| [`v1_moe`](v1_moe/README.md) | First exact grouped MoE E2E gate and Nsight Systems profile evidence. |
