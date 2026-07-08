# Debug Harnesses

This directory is for reusable, tracked debug harnesses.

`performance_milestones/` is intentionally ignored and should keep reports,
large raw outputs, Nsight files, and one-off experiment artifacts.  If a script
becomes a reusable tool for future targets, move or copy the maintained version
here and make new targets reference this directory.

Rules:

- Keep reusable scripts under a topic subdirectory, for example `debug/mtp/`.
- Write large outputs to `performance_milestones/<target>/raw/` or `/tmp`, not
  under `debug/`.
- Prefer stable command-line interfaces over target-specific hardcoding.
- Keep scripts importable from the repository root and from their own path.
- If a milestone-local script is superseded by a tracked debug harness, update
  future targets to use the tracked harness.

Current tracked harnesses:

- `debug/mtp/run_matrix.py`: TP8 baseline/MTP exactness matrix runner used by
  TARGET 11 MTP correctness work.
- `debug/mtp/analyze_state_parity.py`: state/KV parity analyzer with the
  online C128 main-state planner rule from TARGET 11.251.
