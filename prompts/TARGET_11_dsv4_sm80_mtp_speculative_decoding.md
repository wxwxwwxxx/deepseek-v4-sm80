# TARGET 11: DSV4 SM80 MTP Speculative Decoding Pause Report

## Status

Paused and archived for release.

TARGET 11 investigated DeepSeek V4 MTP speculative decoding as a possible
throughput lever after TARGET 08 prefix/SWA stabilization and TARGET 10
communication promotion.  The current conclusion is no-go for the release
branch: mini's target-verify runtime is not equivalent to normal target decode
under the same committed prefix/state.

The MTP implementation, opt-ins, debug harnesses, and detailed investigation
history have been preserved on the local branch:

```text
dsv4-mtp-paused-reference
```

The current `dsv4-sglang-based` branch should not contain active MTP runtime
code or MTP opt-ins.  It may still skip checkpoint tensors named `mtp.*` during
weight loading, because that was the pre-MTP behavior and keeps the non-MTP
model path clean.

## Final Verdict

Do not ship or continue polishing the current MTP implementation.

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
  path to full exactness.

Interpretation: the blocker is not one isolated cache-copy bug.  The current
flattened target-verify runtime contract differs materially from the normal
target decode contract and from SGLang's frozen-KV target-verify design.

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
Draft tokens are allowed to be wrong.  The target verifier is not allowed to be
non-equivalent to normal target decode.

## Main Non-Equivalences Found

- Mini's target-verify row construction is ad hoc: row depths, parent rows,
  active/padded masks, recursive parent splitting, and commit rows are built by
  mini-specific code rather than a proven SGLang-equivalent contract.
- Several kernels or backend paths are row-shape sensitive under target verify.
  q/wq_b and shared expert oracles fixed individual anchors but were not enough
  to make the full exactness matrix pass.
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

If MTP is restarted later, do not resume the local owner chase.  Start from a
SGLang-aligned target-verify contract.

Minimum restart gates:

1. Define every verify row semantically: request id, draft depth, row type,
   token-in, token-scored, absolute position, full location, page offset,
   parent row, KV/SWA/C4/C128 read/write surfaces, and commit eligibility.
2. Port or faithfully reproduce SGLang's frozen-KV target-verify metadata
   preparation before changing kernels for performance.
3. Run teacher-forced target-verify replay first.  It must match normal
   no-spec target decode row logits/top1 before draft acceptance is enabled.
4. Use canonical target state for accepted commits, or prove the target-verify
   state is equivalent.
5. Align the logits processor/sampler contract before evaluating sampling
   behavior.
6. Include batch sizes beyond `bs=4` and a page-offset case where
   `position != full_loc % 256`.
7. Only after greedy target-verify equivalence passes should graph capture,
   acceptance-rate optimization, and throughput work begin.

Approximate or non-strict MTP can be a research-only opt-in in a future branch,
but it should not be described as lossless speculative decoding.

## Post-Cleanup Soak Recommendation

After removing MTP code from `dsv4-sglang-based`, run a non-MTP soak to confirm
that the release baseline is intact.

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
