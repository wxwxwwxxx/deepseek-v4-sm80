# TARGET 11: DSV4 SM80 MTP Speculative Decoding

## Status

Planned after TARGET 08 prefix/SWA stabilization and TARGET 10 communication
promotion.  TARGET 09 low-precision research is deferred for now, so this
target explores a different throughput lever: DeepSeek V4 MTP speculative
decoding.

## Feasibility Verdict

MTP looks feasible, but it is not a small model-loader feature.

The checkpoint and config support MTP:

- `/models/DeepSeek-V4-Flash/config.json` has
  `num_nextn_predict_layers = 1`.
- The checkpoint contains `mtp.0.*` tensors, including MTP attention,
  projection, expert, embedding/head, and normalization weights.
- mini currently skips these weights in
  `python/minisgl/models/weight.py` by ignoring names that start with `mtp.`.

SGLang has the most relevant reference implementation:

- `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
- `/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py`
- `/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- `/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- `/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`

The SGLang design is not "load MTP weights and sample extra tokens".  It is a
frozen-target-KV MTP draft path with a target verification path, plus DSV4
attention/compression metadata handling.

## Current Mini Baseline

Use the latest promoted exact/prefix route as the comparison baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL default threshold32m
--page-size 256
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

If a TARGET 08 child has promoted SWA independent lifecycle or Marlin WNA16
release presets after this file was written, use that newer promoted baseline.

## Design Direction

Follow SGLang first:

1. top-k 1 / single-draft-chain MTP before any tree MTP;
2. frozen target KV for the draft model, not a separate draft KV cache;
3. target verification keeps generation exact under greedy decoding;
4. DSV4 C4/C128/compressed metadata and online C128 MTP handling must be
   source-parity checked before graph promotion;
5. graph capture is a later promotion step, not part of the first correctness
   proof.

Do not start by inventing a mini-only speculative decoding protocol.  First map
the SGLang behavior, then adapt the minimum subset that fits mini-sglang.

## Important SGLang Behaviors To Preserve Or Explain

- MTP weights are loaded with an `is_nextn` path.  SGLang remaps `mtp.0.*` into
  a nextn decoder layer and shared MTP head/embedding fields.
- Current DSV4 MTP support is top-k 1 oriented.  SGLang's DeepSeek V4 attention
  backend asserts top-k in `[0, 1]` for MTP.
- The draft worker reads the target model's KV allocator and `req_to_token`
  mapping read-only.  It does not own a separate draft KV pool.
- Draft position handling uses the target sequence state.  SGLang utilities set
  frozen-KV MTP positions from `seq_lens - 1`.
- Target verification must update normal target KV/cache state only for the
  accepted target path, not for rejected speculative tokens.
- DSV4 C128 online compression state has special MTP handling in
  `online_c128_mtp.py`.  Mini must not assume the regular decode metadata is
  automatically valid for target verify.

## Split Plan

Run these in order.

| Stage | Prompt | Purpose |
| --- | --- | --- |
| TARGET 11.1 | `prompts/TARGET_11.1_dsv4_sm80_mtp_weight_oracle.md` | Load and run the MTP layer/head behind an opt-in, without changing scheduler semantics.  Build a one-step oracle and memory ledger. |
| TARGET 11.2 | `prompts/TARGET_11.2_dsv4_sm80_mtp_spec_runtime_v1.md` | Add a conservative greedy top-k 1 speculative runtime with target verification and exact-output gates. |
| TARGET 11.25 | `prompts/TARGET_11.25_dsv4_sm80_mtp_frozen_kv_verify_runtime.md` | Turn the V1 sidecar into a real frozen-KV draft plus multi-token verify/accept runtime, still eager/no-graph. |
| TARGET 11.27 | `prompts/TARGET_11.27_dsv4_sm80_mtp_flattened_verify_temp_kv.md` | Replace the sequential verifier with a flattened/temp-KV verify path so MTP can reduce target passes while preserving exact ownership. |
| TARGET 11.28 | `prompts/TARGET_11.28_dsv4_sm80_mtp_accepted_kv_commit_root_cause.md` | Root-cause why accepted flattened verify KV commit changes later greedy output; required after the 11.27 rollback-only no-go. |
| TARGET 11.29 | `prompts/TARGET_11.29_dsv4_sm80_mtp_target_verify_contract_port.md` | Port the explicit target-verify metadata/front-chain/C128 pending contract needed before accepted-KV commit can be exact. |
| TARGET 11.295 | `prompts/TARGET_11.295_dsv4_sm80_mtp_online_c128_lifecycle_port.md` | Port or fail-closed the online C128 MTP pending/write/commit lifecycle that blocks accepted target-verify commit. |
| TARGET 11.296 | `prompts/TARGET_11.296_dsv4_sm80_mtp_row0_logits_parity_after_commit.md` | After C128 lifecycle is ready but exactness still drifts, prove and fix row0 target-verify logits parity after accepted commit. |
| TARGET 11.3 | `prompts/TARGET_11.3_dsv4_sm80_mtp_attention_graph_perf.md` | After accepted-KV commit is exact and useful eager target-pass reduction is proven, align DSV4 attention/compression metadata and graph replay with SGLang, then profile throughput. |

## Correctness Contract

For greedy decoding, speculative MTP must be exact:

```text
baseline greedy output == MTP speculative greedy output
```

If this exactness fails, stop performance work and fix verification, token
acceptance, KV/cache updates, or metadata ownership.  Do not accept "mostly
similar" text as a passing result.

For sampling modes, defer exact distributional testing until after greedy
correctness and throughput are proven.  The first implementation may be greedy
only.

## Speculative Stats Glossary

Keep MTP stats precise.  Draft acceptance, target correction, and target-verify
row commit are related but not identical.

```text
draft_tokens_proposed:
    tokens produced by the MTP draft path.
draft_tokens_verified:
    draft tokens compared against target-verify outputs.
draft_tokens_accepted:
    draft tokens whose value matched target verify and became visible output.
draft_tokens_rejected:
    draft tokens whose value did not match target verify.
target_correction_tokens:
    target-model tokens emitted at the first rejection point.
target_verify_rows:
    rows computed by the target verify forward; these may include accepted
    draft rows, correction rows, and bonus/tail rows.
target_verify_rows_committed:
    target-verify rows made visible in target KV/component/state.
accepted_kv_copied_tokens:
    historical mini stat for committed target-verify rows.  Do not interpret it
    as accepted draft token count unless a child target proves the row category.
```

It is normal for MTP draft tokens to be wrong:

```text
draft token != target verify token
```

It is not acceptable for target verification to diverge from the baseline target
model under the same committed prefix/state:

```text
normal target decode row0 logits != target-verify row0 logits
```

## Promotion Gates

Do not promote MTP by default unless all of these are true:

1. Greedy exactness passes on short, medium, long, prefix-hit, no-hit, and
   batched mixed prompts.
2. Text smoke shows no乱码, NaN, repeated garbage, or early EOS regression.
3. Acceptance is high enough to amortize draft overhead.  A weak first stop-line
   is average accepted tokens per target verification above `1.2`; a better
   target is above `1.5`.
4. `4096/1024/bs4` throughput improves over the current promoted exact baseline.
5. Serving-style workloads do not regress heavily when acceptance is low.
6. CUDA graph replay remains healthy after graph support is added.
7. Prefix cache and SWA/component ownership invariants still pass.

## Stop Lines

- If TARGET 11.1 cannot load MTP weights cleanly or one-step outputs are NaN or
  obviously corrupt, do not touch scheduler/runtime.
- If TARGET 11.2 cannot prove greedy exactness, do not proceed to graph/perf.
- If acceptance is too low to pay for draft overhead, keep MTP opt-in or close
  the target as negative evidence.
- If DSV4 target-verify metadata is not SGLang-equivalent, stop at TARGET
  11.29 and fix that contract before continuing.
- If online C128 MTP pending/write/commit handling is not SGLang-equivalent,
  stop at TARGET 11.295 and fix or fail-closed that lifecycle before continuing
  to graph/perf.
- If accepted commit is enabled but target-verify row0 logits diverge from
  normal target decode, stop at TARGET 11.296 and fix row0 parity before
  continuing to graph/perf.

## Deliverables

Each child target should write results under:

```text
performance_milestones/target11_*/
```

Include:

- source-parity notes against SGLang;
- command lines and env flags;
- text smoke/correctness results;
- memory ledger for extra MTP weights/state;
- acceptance and throughput metrics when runtime exists;
- stop-line decision and next target recommendation.
