# Poison And Quarantine Tests

All runs use TP8, page size `256`, `--num-pages 128`, `--max-tokens 8`,
`dsv4_sm80_a100_victory_marlin_release`, and no CUDA graph unless noted.

## Matrix

| Test | Env pattern | Raw artifact | Status | Replay / eager | Prompt 0 token ids |
| --- | --- | --- | --- | ---: | --- |
| Hidden ref poison | `zero` | `raw/text_smoke_hidden_ref_poison_zero.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| Hidden ref poison | `nan` | `raw/text_smoke_hidden_ref_poison_nan.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| Freed-block quarantine | all released bytes, `zero` | `raw/text_smoke_release_quarantine_all_zero.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| Freed-block quarantine | `6.375GiB`, `zero` | `raw/text_smoke_release_quarantine_6p375gib.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| Freed-block quarantine | `3.1875GiB`, `zero` | `raw/text_smoke_release_quarantine_3p1875gib.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| Freed-block quarantine | `3.1875GiB`, `deterministic` | `raw/text_smoke_release_quarantine_3p1875gib_deterministic.dsv4_sm80_a100_victory_marlin_release.json` | pass | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |

## Hidden-Ref Poison

Hidden-ref poison keeps references to the original raw expert tensors so their storage is
not returned to the allocator, then overwrites their contents after Marlin prebuild and cache
signature validation.

Both zero and NaN poison pass.  This rules out a necessary stale read of the original raw
expert tensor contents in this smoke.  If the grouped/raw path were silently being used after
release, these poison runs should have corrupted output.

## Freed-Block Quarantine

Freed-block quarantine physically releases the raw tensor attributes, then immediately
allocates dummy tensors to hold allocator blocks that would otherwise be reused.

All quarantine runs pass, including the smaller requested pressure point of `3.1875GiB/rank`
and both zero and deterministic patterns.  This shows that the important variable is block
ownership/lifetime, not the dummy tensor value.

## Pressure Sweep Result

The requested `3.1875GiB` to `6.375GiB` pressure interval did not produce a failing point:
both endpoints passed.  The smallest tested quarantine size, `3.1875GiB/rank`, is already
enough to protect the text smoke.

This does not mean `3.1875GiB` is a promotion policy.  It means the repro is sensitive to
which freed blocks are captured by early DSV4 KV/component owners.  A production fix should
use a clear ownership boundary or arena policy, not a dummy quarantine.
