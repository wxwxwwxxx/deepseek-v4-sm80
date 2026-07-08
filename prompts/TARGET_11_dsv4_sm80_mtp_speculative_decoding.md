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
| TARGET 11.297 | `prompts/TARGET_11.297_dsv4_sm80_mtp_target_verify_batch_shape_hidden_parity.md` | After the visible row0 token drift is fixed but logits still differ, bisect normal one-row decode vs flattened target-verify row0 hidden parity. |
| TARGET 11.298 | `prompts/TARGET_11.298_dsv4_sm80_mtp_wo_a_projection_batch_shape_parity.md` | Fix or prove the layer0 `wo_a` projection batch-shape owner found by 11.297. |
| TARGET 11.299 | `prompts/TARGET_11.299_dsv4_sm80_mtp_multi_request_verify_contract.md` | Fix or prove the multi-request target-verify row/depth and mixed verify-length contract after `wo_a` parity is closed. |
| TARGET 11.5 | `prompts/TARGET_11.5_dsv4_sm80_mtp_bs4_accepted_commit_state_parity.md` | After 11.299 fixes row/depth and mixed-length contract issues, isolate and repair the remaining `bs=4` accepted-commit state drift. |
| TARGET 11.6 | `prompts/TARGET_11.6_dsv4_sm80_mtp_bs5_exposure_state_parity.md` | After 11.5 fixes `bs=4`, classify and repair the new `bs=5` exposure failure, starting with a normal-target batch-shape oracle. |
| TARGET 11.7 | `prompts/TARGET_11.7_dsv4_sm80_mtp_bs6_path_census_contract_closure.md` | After 11.6 fixes `bs=5`, enumerate the MTP target-verify path matrix and close the smallest remaining `bs=6` exposure failure. |
| TARGET 11.8 | `prompts/TARGET_11.8_dsv4_sm80_mtp_target_verify_runtime_contract_unification.md` | After 11.7 proves per-batch patching is unsafe, write and implement or plan a unified target-verify runtime contract. |
| TARGET 11.9 | `prompts/TARGET_11.9_dsv4_sm80_mtp_sglang_aligned_target_verify_runtime_mode.md` | After 11.8 stops at a contract/no-go for local patching, implement one SGLang-aligned target-verify runtime mode and prove eager exactness through `bs=1/2/4/5/6`. |
| TARGET 11.10 | `prompts/TARGET_11.10_dsv4_sm80_mtp_target_verify_layer0_attention_kv_producer_parity.md` | After 11.9 makes the SGLang-shaped runtime explicit but non-exact, fix or precisely no-go the first owner at layer0 target-verify attention/KV producer parity. |
| TARGET 11.11 | `prompts/TARGET_11.11_dsv4_sm80_mtp_attn_wo_b_projection_reduce_parity.md` | After 11.10 fixes attention/KV parity and exposes `layer0.final_attention_output`, fix or precisely no-go `attn.wo_b` row-parallel projection/all-reduce parity. |
| TARGET 11.12 | `prompts/TARGET_11.12_dsv4_sm80_mtp_rank_local_downstream_parity_census.md` | After 11.11 closes bs=1 `wo_b` parity but the matrix still fails, census rank-local downstream owners such as indexer FP8, MoE, and later-layer attention before the next fix. |
| TARGET 11.13 | `prompts/TARGET_11.13_dsv4_sm80_mtp_operator_parity_framework_q_norm_rope_pilot.md` | After 11.12 ranks q/RoPE as the top rank-local owner, build a reusable operator-parity framework and use q_norm_rope as the first same-kernel/micro-allclose pilot. |
| TARGET 11.14 | `prompts/TARGET_11.14_dsv4_sm80_mtp_q_wqb_q_lora_precision_boundary_parity.md` | After 11.13 shows q_norm_rope only amplifies a non-bit-exact `q_wqb_output`, use the operator framework to find and fix/no-go the upstream q_lora/q_norm/wq_b precision boundary. |
| TARGET 11.15 | `prompts/TARGET_11.15_dsv4_sm80_mtp_moe_output_subboundary_parity.md` | After 11.14 closes the q/wq_b boundary and exposes exact-input MoE drift, split `moe_input -> moe_output` into router/topk, routed expert, shared expert, aggregation, and reduce sub-boundaries. |
| TARGET 11.16 | `prompts/TARGET_11.16_dsv4_sm80_mtp_moe_post_reduce_parity.md` | After 11.15 proves router/topk, routed expert, shared expert, and pre-reduce aggregation are exact, fix or precisely no-go the MoE post-experts reduce/all-reduce boundary. |
| TARGET 11.17 | `prompts/TARGET_11.17_dsv4_sm80_mtp_moe_pre_reduce_drifting_rank_parity.md` | After 11.16 proves post-reduce drift is propagated from rank0/rank7 local pre-reduce aggregate drift, find and fix/no-go the drifting-rank MoE sub-boundary. |
| TARGET 11.18 | `prompts/TARGET_11.18_dsv4_sm80_mtp_post_moe_downstream_owner_census.md` | After 11.17 fixes MoE pre-reduce but the matrix still fails, rank downstream owners such as layer21 `attention_wo_b` and layer32 `indexer_query_fp8_values` before choosing the next fix. |
| TARGET 11.19 | `prompts/TARGET_11.19_dsv4_sm80_mtp_accepted_commit_lifecycle_state_parity.md` | After 11.18 ranks accepted-commit lifecycle/post-commit state above local operator fixes, find the first event/state component that makes Mini self-consistent but baseline-divergent. |
| TARGET 11.20 | `prompts/TARGET_11.20_dsv4_sm80_mtp_row_depth_committed_state_baseline_parity.md` | After 11.19 narrows the owner to bs4 uid0 event4 depth1 committed correction-row state, compare MTP committed rows against baseline greedy state hashes. |
| TARGET 11.21 | `prompts/TARGET_11.21_dsv4_sm80_mtp_target_verify_row_depth_producer_parity.md` | After 11.20 proves logical ownership is correct but `swa.layer1` row values differ, find the layer0-to-layer1 producer boundary that first diverges. |
| TARGET 11.22 | `prompts/TARGET_11.22_dsv4_sm80_mtp_layer0_wo_a_projection_contract_parity.md` | After 11.21 identifies layer0 `attention_wo_a_output` as the first producer mismatch, align target-verify `wo_a` projection with baseline/SGLang semantics. |
| TARGET 11.23 | `prompts/TARGET_11.23_dsv4_sm80_mtp_layer0_wo_b_projection_reduce_parity.md` | After 11.22 closes `wo_a` and exposes layer0 `wo_b` local/post-all-reduce drift, fix or precisely no-go the `wo_b` projection/reduce contract. |
| TARGET 11.24 | `prompts/TARGET_11.24_dsv4_sm80_mtp_post_layer1_logits_owner_census.md` | After 11.23 closes layer0->layer1 for the old anchor but bs2/bs6 still drift, bisect from layer1 through final logits/sampler/commit to find the next owner. |
| TARGET 11.241 | `prompts/TARGET_11.241_dsv4_sm80_mtp_layer2_attention_committed_kv_state_owner.md` | After 11.24 proves the next first boundary is layer2 attention output, split current Q compute from consumed committed KV/SWA/C128/page metadata, with bs6 as a full-matrix lifecycle guard. |
| TARGET 11.242 | `prompts/TARGET_11.242_dsv4_sm80_mtp_layer2_swa_commit_state_producer_owner.md` | After 11.241 proves layer2 consumed SWA cache values are non-equivalent, trace the bad SWA locs through producer, store, snapshot restore, and accepted commit. |
| TARGET 11.243 | `prompts/TARGET_11.243_dsv4_sm80_mtp_target_verify_layer2_input_producer_parity.md` | After 11.242 proves the bad SWA store inputs are already non-equivalent, trace target-verify writer rows from embedding through layer2 input and classify the first producer boundary. |
| TARGET 11.244 | `prompts/TARGET_11.244_dsv4_sm80_mtp_target_verify_layer0_moe_output_subboundary_parity.md` | After 11.243 proves the first producer boundary is layer0 MoE output, split router, routed expert, shared expert, aggregation, and reduce for the current target-verify rows. |
| TARGET 11.245 | `prompts/TARGET_11.245_dsv4_sm80_mtp_target_verify_layer0_moe_row_shape_precision_contract.md` | After 11.244 proves target-verify layer0 MoE is row-shape/precision sensitive, define the SGLang-aligned execution contract and test row-stable or shape-stable oracles. |
| TARGET 11.246 | `prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md` | After 11.245 proves normal-shape-compatible MoE microbatching is the only exact oracle under Mini's backend, implement the runtime path and validate the full exactness matrix. |
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
- If visible token exactness passes but row0 full logits still differ enough to
  threaten stability, stop at TARGET 11.297 and find the first layer/submodule
  owner before continuing to graph/perf.
- If the first row0 hidden-parity owner is `wo_a` projection batch shape, stop
  at TARGET 11.298 and fix/prove that projection path before continuing to
  graph/perf.
- If bs=1 target verify is exact but bs=2/4 exposes row/depth or mixed
  verify-length contract failures, stop at TARGET 11.299 and fix/prove the
  multi-request target-verify contract before continuing to graph/perf.
- If TARGET 11.299 proves row/depth and mixed-length handling but `bs=4` still
  diverges after accepted/correction row commit, stop at TARGET 11.5 and
  identify the first non-equivalent committed state owner before continuing to
  graph/perf.
- If TARGET 11.5 fixes `bs=1/2/4` but light exposure finds `bs=5+` failures,
  stop at TARGET 11.6 and first determine whether normal target decode is
  batch-shape sensitive for the failing prefix before continuing state-parity
  repair or graph/perf.
- If TARGET 11.6 fixes `bs=1/2/4/5` but `bs=6+` still exposes new failures,
  stop at TARGET 11.7 and build a source/runtime path census before applying
  another local correctness fix.
- If TARGET 11.7 finds that force_torch/separate-KV, force_torch/fused-KV, and
  splitk/fused-KV each fix different cases while regressing others, stop at
  TARGET 11.8 and unify the target-verify runtime contract before any more
  per-batch repairs.
- If TARGET 11.8 writes the unified contract but concludes that no existing
  local flag combination can implement it safely, stop at TARGET 11.9 and port
  one SGLang-aligned target-verify runtime mode before graph/perf work.
- If TARGET 11.9 cannot prove eager exactness through `bs=1/2/4/5/6` with
  accepted commit enabled, do not start TARGET 11.3; write the next correctness
  target around the first non-batch-special-case owner.
- If TARGET 11.9's first owner is
  `layer0.merged_attention_output_before_wo` under `sglang_prefill_extend`, stop
  at TARGET 11.10 and fix/prove target-verify attention/KV producer parity
  before any graph/perf or C128 boundary work.
- If TARGET 11.10 fixes attention/KV parity but the new first owner is
  `layer0.final_attention_output`, stop at TARGET 11.11 and fix/prove
  `attn.wo_b` projection/all-reduce parity before graph/perf or C128 boundary
  work.
- If TARGET 11.11 closes bs=1 `attn.wo_b` parity but the expanded matrix still
  fails with rank-local downstream owners such as indexer FP8, MoE, or later
  attention, stop at TARGET 11.12 and build a rank-local owner census before
  applying another local fix.
- If TARGET 11.12 shows multiple independent downstream owners and ranks
  q/RoPE as the earliest common rank-local owner, stop at TARGET 11.13 and build
  an operator-parity framework with q_norm_rope as the pilot before fixing MoE
  or indexer.
- If TARGET 11.13 shows q_norm_rope uses the same kernel in normal decode and
  target verify and only amplifies a non-bit-exact `q_wqb_output`, stop at
  TARGET 11.14 and identify the upstream q_lora/q_norm/wq_b precision boundary
  before fixing MoE or indexer.
- If TARGET 11.14 closes the q/wq_b boundary but the next owner is `moe_output`
  with exact `moe_input`, stop at TARGET 11.15 and split the MoE output
  sub-boundaries before fixing indexer FP8 or later attention.
- If TARGET 11.15 proves the first output-significant MoE owner is
  `expert_aggregate_before_reduce exact -> expert_reduce_output drift`, stop at
  TARGET 11.16 and fix/prove the post-experts reduce/all-reduce contract before
  fixing indexer FP8, later attention, or graph/perf.
- If TARGET 11.16 proves the post-reduce drift is caused by all-reduce SUM
  propagating rank0/rank7 local `expert_aggregate_before_reduce` drift, stop at
  TARGET 11.17 and debug the drifting ranks' MoE pre-reduce sub-boundaries
  before changing communication policy or fixing indexer FP8.
- If TARGET 11.17 fixes the scoped MoE pre-reduce owner but the matrix still
  fails with post-MoE downstream owners such as layer21 `attention_wo_b` and
  layer32 `indexer_query_fp8_values`, stop at TARGET 11.18 and build an
  event/layer/rank owner census before choosing the next local fix.
- If TARGET 11.18 shows that direct `attention_wo_b` and indexer FP8 fixes are
  lower priority than accepted-commit lifecycle/post-commit state divergence,
  stop at TARGET 11.19 and find the first event/state component that diverges
  from baseline greedy before applying more local operator fixes.
- If TARGET 11.19 narrows the first concrete unclosed owner to bs4 uid0 event4
  committed depth1 correction-row state but lacks baseline-side state hashes,
  stop at TARGET 11.20 and compare row-depth committed state against baseline
  greedy before applying a runtime patch.
- If TARGET 11.20 proves row-depth ownership is equivalent but committed
  `swa.layer1` values differ between baseline greedy and MTP target verify,
  stop at TARGET 11.21 and trace the producer-side layer0-to-layer1 boundary
  before patching attention, indexer, C128, or graph/perf.
- If TARGET 11.21 proves the first output-significant producer mismatch is
  layer0 `attention_wo_a_output`, stop at TARGET 11.22 and align the target
  verify `wo_a` projection contract with baseline/SGLang before patching later
  attention, indexer, C128, or graph/perf.
- If TARGET 11.22 closes the `wo_a` contract but layer0 `wo_b` local or
  post-all-reduce output remains the first output-significant mismatch, stop at
  TARGET 11.23 and fix/prove the `wo_b` projection/reduce contract before
  patching indexer, C128, lifecycle ownership, or graph/perf.
- If TARGET 11.23 closes layer0 `wo_b`, layer0 post-MoE, layer1 input, and
  layer1 KV for the old anchor but the broad matrix still fails at bs2/bs6,
  stop at TARGET 11.24 and bisect from layer1 through final logits, sampler,
  and visible commit before reopening layer0, indexer, C128, lifecycle
  ownership, or graph/perf.
- If TARGET 11.24 proves hidden states are exact through layer2 input but first
  diverge at layer2 attention output, stop at TARGET 11.241 and split layer2
  attention into current Q compute versus consumed committed KV/SWA/C128/page
  metadata before patching logits, sampler, graph/perf, or low-precision paths.
- If TARGET 11.241 proves current Q, metadata, C4 cache, and attention backend
  dispatch are equivalent but consumed layer2 SWA cache values differ, stop at
  TARGET 11.242 and trace the bad SWA locs through producer, store, snapshot
  restore, accepted commit, and later read before patching attention kernels or
  downstream logits/sampler.
- If TARGET 11.242 proves SWA store, snapshot restore, commit, and later read
  preserve their inputs but the bad SWA rows are already non-equivalent at
  target-verify writer `layer2.input`, stop at TARGET 11.243 and trace the
  writer rows from embedding through layer2 input before patching SWA cache
  lifecycle again.
- If TARGET 11.243 proves row identity, embedding, layer0 attention, and
  `layer0.moe_input` are equivalent but bad rows first diverge at
  `layer0.moe_output`, stop at TARGET 11.244 and split layer0 MoE into router,
  routed expert, shared expert, aggregation, and reduce before reopening SWA or
  layer2 attention.
- If TARGET 11.244 proves router logits/topk ids are exact but target-verify
  layer0 MoE remains row-shape/precision sensitive across topk weights, expert
  backends, reduce, and final BF16 materialization, stop at TARGET 11.245 and
  define the SGLang-aligned target-verify MoE shape/precision contract before
  implementing a runtime fix.
- If TARGET 11.245 proves normal-shape-compatible microbatching is the only
  exact target-verify MoE oracle under Mini's current SM80 backend, stop at
  TARGET 11.246 and implement that microbatch contract as a real runtime path
  before graph/perf promotion.

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
