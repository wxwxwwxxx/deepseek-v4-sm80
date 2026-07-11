# TARGET 11: DSV4 SM80 MTP Speculative Decoding Pause Report

## Status

Paused and archived.  MTP will not be restarted for `v0.0.0`.

TARGET 11 investigated DeepSeek V4 MTP speculative decoding as a possible
throughput lever after TARGET 08 prefix/SWA stabilization and TARGET 10
communication promotion.  The release conclusion remains no-go: the explored
high-performance flattened target-verify runtime did not establish the
required target distribution contract against normal target decode.

The MTP implementation, opt-ins, debug harnesses, and detailed investigation
history have been preserved on the local branch:

```text
dsv4-mtp-paused-reference
```

The tagged `v0.0.0` release and current `dsv4-sglang-based` branch must not
contain active MTP runtime code or MTP opt-ins.  They may still skip checkpoint
tensors named `mtp.*` during weight loading, because that was the pre-MTP
behavior and keeps the non-MTP model path clean.

## Final Verdict

Do not ship MTP, restart MTP, or continue polishing the paused implementation
for `v0.0.0`.

The latest pivot report is:

```text
performance_milestones/target11_mtp_target_verify_pivot_feasibility/README.md
```

Its final classification was:

```text
pause_mtp_for_release
```

Key evidence:

- Teacher-forced target-verify replay replaced draft proposals with baseline
  future tokens and blocked verify-produced emission/commit.  Visible emitted
  tokens passed only because commit was blocked, but the row-depth oracle failed
  with `9/174` target-verify top1 mismatches.
- The first teacher-forced mismatch was already large enough to flip output:
  target-verify top1 `582` versus normal target oracle top1 `223`, with max
  logit delta about `2.14`.
- Canonical replay commit, where committed state is replayed through normal
  target decode after target-verify decisions, still failed `bs=2/4/5/6`.
  This means the decisions/logits produced by target verify were already
  non-equivalent before commit-state source mattered.
- Local owner repairs around `q_wqb`, shared expert, SWA, C128, MoE aggregate,
  and all-reduce could improve individual anchors but did not produce a bounded
  path to full strict token exactness.

Interpretation: the blocker is not one isolated cache-copy bug.  The old
investigation also mixed two different problem classes: real target-verify
state/lifecycle defects and deterministic row-shape-dependent numerical
differences.  The postmortem below supersedes the broader interpretation that
every late mismatch necessarily proved another cache or commit-state bug.

## v0.0.0 Postmortem Re-evaluation

The release decision is unchanged, but the likely root mechanism is now more
specific.

```text
v0.0.0 MTP status:                  disabled and paused
restart on the v0.0.0 release line: no
dsv4-mtp-paused-reference usage:    read-only oracle/history
future restart:                     conditional, contract-first, post-release
```

### Reclassified Evidence

- The serial depth-by-depth target verifier from TARGET 11.25 produced exact
  token ids for `bs=1/2/4` and draft lengths `2/4`.  It was too slow to be a
  useful serving implementation, but proves that the model and MTP checkpoint
  are not intrinsically unusable.
- The rollback-only flattened verifier later passed exactness at smaller batch
  sizes before accepted-KV commit was enabled.  The more complete path reached
  `bs=1/2/4/5` and failed at `bs=6`, which is consistent with row-shape and
  previously untested path changes rather than one universal bad token.
- Direct probes found deterministic, reproducible row-shape differences.  For
  the same real input and same cached BF16 `q_wqb` weight, a normal four-row
  projection versus target `3+1`/one-row shapes differed by up to
  `6.103515625e-05`.  Shared expert one-row versus three/four-row execution
  differed by up to `0.0001220703125`.
- These differences were not run-to-run nondeterminism.  They are backend
  accumulation/rounding differences that can become a BF16 ULP after q/RoPE,
  grow over many layers, perturb MoE routing, and eventually produce a much
  larger logit delta or top1 flip.  Therefore the final `~2.14` logit delta is
  not evidence that the initial error was already large.
- Replacing one surface with a per-row oracle made that local anchor exact, but
  moved or regressed another case.  This explains much of the historical
  whack-a-mole pattern: several GEMM/MoE surfaces changed row shape together.
- Normal non-MTP decode itself did not provide batch invariance in the old
  probes: the same request could choose token `671` or `9628` under different
  batch shapes while remaining text-sane.  Requiring MTP to be token-identical
  to one no-spec batch shape is therefore stricter than the numerical contract
  the release engine currently guarantees.

### Root-Cause Confidence

1. **High confidence:** deterministic row-shape/BF16 numerical
   non-invariance is a dominant cause of the late token flips.
2. **Medium confidence:** mini's flattened target-verify architecture still
   has contract debt.  It overloaded `phase="decode"`, kept request-count
   `Batch.size`, and supplied `request_count * verify_width` token rows to
   operators.  Target-only parent splitting, MoE branches, metadata, and raw
   `lm_head.linear`/argmax therefore did not form one dedicated, proven verify
   mode equivalent to SGLang's implementation.
3. **Established historical defects:** KV/SWA/C128 publication and accepted
   commit lifecycle bugs were real and required fixes.  Canonical replay and
   the later row-shape probes indicate they are not a sufficient explanation
   for the final remaining mismatch.

This means the old owner-by-owner patch sequence should not be resumed.  It
also means there is no honest finite-patch promise for a fast, flattened,
strictly token-exact verifier.  There is, however, a bounded way to decide
whether a future production-tolerant MTP implementation is viable.

### SGLang Correctness Reference

SGLang uses a dedicated target-verify mode and metadata contract, frozen-KV
draft attention, its normal logits processing surface, and explicit accepted
path movement.  Its registered frozen-KV MTP test gates end-task quality and
acceptance efficiency, including GSM8K score of at least `0.65` and average
acceptance length of at least `1.5`; it does not require token identity with a
separate no-spec run under every batch shape.

This is a practical product-quality gate, not proof that SGLang's verifier is
bitwise or distribution-exact under every BF16 batch shape.

That distinction matters.  Classical exact speculative sampling assumes the
verifier evaluates the same target distribution as ordinary decode.  Practical
BF16 row-shape changes may violate bit/token identity without necessarily
causing unacceptable model quality.  Any future relaxed implementation must be
labelled and validated as production-tolerant or approximate, not claimed to
be lossless merely because text looks sane.

## What Was Learned

The model and checkpoint do support MTP:

- `/models/DeepSeek-V4-Flash/config.json` has
  `num_nextn_predict_layers = 1`.
- The checkpoint contains `mtp.0.*` weights.
- SGLang has a relevant frozen-KV MTP implementation.

Important SGLang references for any future restart:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

The MTP draft model is roughly one next-token/nextn transformer block plus
shared embedding/head surfaces.  It is not a full copy of the target verifier.
Draft tokens are allowed to be wrong.  Under a strict/lossless contract, the
target verifier is not allowed to be non-equivalent to normal target decode;
under a deliberately relaxed contract, the residual difference must instead
pass the explicit quality and distribution gates above.

## Main Non-Equivalences Found

- Mini's target-verify row construction was ad hoc: row depths, parent rows,
  active/padded masks, recursive parent splitting, and commit rows were built
  by mini-specific code rather than one proven SGLang-equivalent verify mode.
- Its logical request count and flattened token-row count were both visible to
  different operators.  This dual-size contract changed GEMM and MoE row
  shapes relative to normal target decode.
- Several kernels or backend paths are row-shape sensitive under target verify.
  `q_wqb` and shared-expert oracles fixed individual anchors but were not enough
  to make the full strict-exactness matrix pass.
- Attention/KV/SWA/C4/C128 metadata lifetimes were repeatedly exposed as
  symptoms.  Canonical replay showed these are not sufficient root fixes while
  target-verify decisions are already wrong.
- The inspected mini target-verify path used raw `lm_head.linear`/argmax
  surfaces.  A future restart must align the normal logits processor/sampler
  contract as well.
- The final pivot did not cover a position/page-offset control where
  `position != full_loc % 256`; any future contract port must include that
  case.

## Archive Map

Fine-grained TARGET 11 prompts are archived under:

```text
prompts/archive/target11/
```

Important historical child prompts:

- `prompts/archive/target11/TARGET_11.1_dsv4_sm80_mtp_weight_oracle.md`
- `prompts/archive/target11/TARGET_11.2_dsv4_sm80_mtp_spec_runtime_v1.md`
- `prompts/archive/target11/TARGET_11.25_dsv4_sm80_mtp_frozen_kv_verify_runtime.md`
- `prompts/archive/target11/TARGET_11.29_dsv4_sm80_mtp_target_verify_contract_port.md`
- `prompts/archive/target11/TARGET_11.9_dsv4_sm80_mtp_sglang_aligned_target_verify_runtime_mode.md`
- `prompts/archive/target11/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md`
- `prompts/archive/target11/TARGET_11.261_dsv4_sm80_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle.md`
- `prompts/archive/target11/TARGET_11.262_dsv4_sm80_mtp_target_verify_pivot_feasibility.md`

Use these files only as historical evidence.  New release or performance work
should not start from an archived MTP child prompt.

## Future Restart Conditions

If MTP is restarted after `v0.0.0`, create a new branch from the then-current
stable release, for example `mtp-v2-contract-port`.  Do not merge or continue
the paused branch wholesale.  Use `dsv4-mtp-paused-reference` only for model
loading code, serial oracles, debug harnesses, and historical evidence.

The future route must begin by choosing its numerical product contract:

- **Strict mode:** verifier probabilities/top1 must satisfy the chosen no-spec
  oracle contract.  The serial verifier is the correctness oracle, even if it
  proves too slow for promotion.
- **Production-tolerant mode:** small deterministic BF16 differences are
  allowed only after KL/logit-margin, quality, acceptance-length, and E2E
  throughput gates pass.  This mode cannot be advertised as bit-exact or
  lossless.

Minimum restart gates:

1. Define every verify row semantically: request id, draft depth, row type,
   token-in, token-scored, absolute position, full location, page offset,
   parent row, KV/SWA/C4/C128 read/write surfaces, and commit eligibility.
2. Port or faithfully reproduce SGLang's dedicated target-verify mode,
   frozen-KV metadata preparation, logits processor, and accepted-path
   movement before changing kernels for performance.
3. Run teacher-forced target verify with acceptance, commit, sampling, and CUDA
   graph disabled.  Compare it first with the serial verifier on the same
   semantic rows using allclose, KL divergence, top1 margin, and per-layer
   hidden-state evidence.  Require exact top1 only if strict mode was chosen.
4. Use canonical target state for accepted commits, or prove the target-verify
   state is equivalent.
5. Align the logits processor/sampler contract before evaluating sampling
   behavior.
6. Cover `bs=1/2/4/6/16`, C4/C128/SWA boundaries, and a page-offset case where
   `position != full_loc % 256` before enabling online commit.
7. Evaluate real prompts/datasets for KL, quality, average accepted length, and
   throughput.  Only then enable CUDA graph and optimize performance.

### Bounded Future Feasibility Experiments

Before committing to another full implementation, run only these three
decision experiments:

1. **Serial versus flattened teacher-forced verifier:** disable commit and use
   identical semantic rows.  Locate the first numerical divergence and record
   whether it is explained by row-shape backend selection.
2. **SGLang runtime oracle:** run no-spec and frozen-KV MTP with the same DSV4
   checkpoint, prompts, and batch matrix.  Measure its own token invariance,
   quality, KL, acceptance length, and throughput rather than assuming it is
   bit-identical.
3. **Minimal contract port:** on the current stable mini runtime, port only the
   dedicated SGLang-aligned TARGET_VERIFY metadata/logits boundary.  Do not add
   draft acceptance, commit, or graph capture until the teacher-forced matrix
   is classified.

Stop after these experiments if a dedicated contract still cannot distinguish
semantic/state errors from row-shape numerical differences, or if the relaxed
quality/acceptance/throughput case is not compelling.  This is the bounded
GO/NO-GO gate for any post-release restart.

## Completed Post-Cleanup Outcome

MTP code was removed from `dsv4-sglang-based`, and subsequent non-MTP soak,
TARGET 12 release work, and the `v0.0.0` tag established the supported baseline.
The old checklist is retained below only as historical context.

Minimum gate:

```text
1. Python/unit smoke for DSV4 config, cache, wrappers, and fallback tests.
2. Text sanity smoke with page size 256 and no MTP environment variables.
3. TP8 macro on the promoted TARGET 10 preset:
   dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
4. Include at least:
   - historical_4096_128_bs4
   - historical_4096_1024_bs4
   - serving_mixed_112req_wave16
   - prefix_multi_112req_wave16
5. Record the result as the post-MTP-cleanup baseline milestone.
```

The baseline should be compared against TARGET 08.30 and TARGET 10.27, not
against any MTP run.
