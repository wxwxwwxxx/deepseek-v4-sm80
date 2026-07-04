# TARGET 10.1: DSV4 SM80 Communication Path Parity vs vLLM

Status: complete census. No NCCL/PyNCCL/symmetric-memory/backend tuning, no low-precision rewrite, and no attention-kernel rewrite were made in this target.

## Inputs And Artifacts

- Mini source: `/workspace/mini-sglang`, system Python.
- vLLM source: `/workspace/vllm-dsv4-docker`, venv `/workspace/venvs/vllm-dsv4/bin/python`.
- Mini runtime source: Target08 promoted-prefix artifacts under `performance_milestones/target08_post_prefix_reprofile/`.
- vLLM runtime source: Target7 runner reused through `performance_milestones/vllm/scripts/run_vllm_deepseek_v4_matrix.py` and env setup from `performance_milestones/vllm/scripts/vllm_env.sh`.
- Generated summaries:
  - `summaries/mini_comm_census.json`
  - `summaries/vllm_comm_census_4096x128_bs4.json`
  - `raw/vllm_comm_probe_4096x128_bs4/communication_probe.json`
  - `raw/vllm_comm_probe_4096x128_bs4/reports/000_decode_throughput_bs8__vllm.json`

Key caveat: mini communication counters count Python-visible/capture calls. Decode CUDA graph replay does not increment those counters again. Mini owner-timing rows cover profile replay/capture ownership separately. vLLM runtime probing sees Python-visible logits all-gather, while compiled backbone all-reduces are inside torch.compile/CUDA graph/custom ops and are therefore marked as static fallback rows below.

## Static Communication Path Census

| Boundary | mini path | vLLM path | Parity | Severity |
|---|---|---|---|---|
| TP wrapper and accounting | `DistributedCommunicator.all_reduce/all_gather`, stats and owner labels in `python/minisgl/distributed/impl.py:68-95,125-158,167-184` | `tensor_model_parallel_*` into `GroupCoordinator` in `/workspace/vllm-dsv4-docker/vllm/distributed/communication_op.py:12-35` and `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py:494-580` | Same collective owner boundary class, different runtime implementation. vLLM default uses custom op path when enabled. | Match, with backend implementation difference |
| Vocab embedding | mini masks local vocab shard then `all_reduce`, label `dsv4.embedding_all_reduce`, `python/minisgl/models/deepseek_v4.py:755-764` | vLLM masks local shard then `tensor_model_parallel_all_reduce`, `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/vocab_parallel_embedding.py:470-490` | Same op and bf16 tensor family `[tokens,4096]`. | Match |
| Attention output `wo_b` | mini row-parallel `wo_b` all-reduce label `dsv4.attn.wo_b.row_parallel_projection_all_reduce`, `python/minisgl/models/deepseek_v4.py:1822-1840` | vLLM `DeepseekV4MLAModules.wo_b` is `RowParallelLinear`; reduce occurs in `RowParallelLinear.forward`, `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:681-714`, `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py:1538-1558` | Same per-layer row-parallel owner boundary and bf16 tensor family. | Match |
| MoE routed/shared reduction | mini current path uses reduce-once: routed and shared are local, then final `dsv4.v1_moe_reduce_once_all_reduce`, `python/minisgl/models/deepseek_v4.py:2379-2408`; runner variant also final-reduces in `:2287-2325` | vLLM SM80 forces standard FusedMoE, shared MLP is not separately reduced, and runner final output all-reduces in `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:618-631,683-690,736-752` and `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:337-379` | Boundary matches as reduce-once, but dtype does not: mini runtime is fp32, vLLM static source indicates bf16 output/all-reduce (`mxfp4.py:65-67`, `modular_kernel.py:1372-1378`). | High |
| Separate shared/routed all-reduce | mini has labels for `dsv4.shared_expert_all_reduce` and `dsv4.routed_expert_all_reduce`, but current reduce-once path suppresses them | vLLM standard SM80 path final-reduces combined output; no PCP dispatch/combine at PCP=1 | No extra shared/routed collectives on either current path. | Match |
| LM head logits | mini casts hidden/weight to fp32 then `all_gather`, label `dsv4.lm_head_all_gather`, `python/minisgl/models/deepseek_v4.py:766-775` | vLLM runtime observed `LogitsProcessor._gather_logits` as bf16 `all_gather`, `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/logits_processor.py:75-99` | Both all-gather full vocab in this run, but dtype differs and shape representation differs (`[B,16160]->[B*8,16160]` vs `[B,16160]->[B,129280]`). | Low to medium |
| Metadata/runtime overhead | mini has DSV4 component/page-table metadata work and direct graph metadata buffers | vLLM default runner does not expose a matching component-location metadata path in this probe | Not a collective mismatch; keep separate from backend experiments. | Informational |

## Mini Runtime Communication Census

Source: `summaries/mini_comm_census.json`, generated from Target08 promoted-prefix macro r01 reports and profile owner timing. Bytes are summed across TP ranks as recorded by mini stats. Owner timing is max-rank total from Target08 profile rows where available.

| scenario | label | op | dtype | shape -> output | count | bytes GiB | owner max ms | owner count | graph replay/eager |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `historical_4096_128_bs4` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `all_reduce` | `bfloat16` | `[16384,4096] -> [16384,4096]` | 688 | 86.000 | 2486.3 | 3784 | 127/0 `{'4':127}` |
| `historical_4096_128_bs4` | `dsv4.embedding_all_reduce` | `all_reduce` | `bfloat16` | `[16384,4096] -> [16384,4096]` | 16 | 2.000 | 1483.0 | 88 | 127/0 `{'4':127}` |
| `historical_4096_128_bs4` | `dsv4.lm_head_all_gather` | `all_gather` | `float32` | `[4,16160] -> [32,16160]` | 16 | 0.031 | 18.0 | 88 | 127/0 `{'4':127}` |
| `historical_4096_128_bs4` | `dsv4.v1_moe_reduce_once_all_reduce` | `all_reduce` | `float32` | `[16384,4096] -> [16384,4096]` | 688 | 172.000 | 810.7 | 3784 | 127/0 `{'4':127}` |
| `historical_4096_1024_bs4` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `all_reduce` | `bfloat16` | `[16384,4096] -> [16384,4096]` | 688 | 86.000 | N/A | N/A | 1023/0 `{'4':1023}` |
| `historical_4096_1024_bs4` | `dsv4.embedding_all_reduce` | `all_reduce` | `bfloat16` | `[16384,4096] -> [16384,4096]` | 16 | 2.000 | N/A | N/A | 1023/0 `{'4':1023}` |
| `historical_4096_1024_bs4` | `dsv4.lm_head_all_gather` | `all_gather` | `float32` | `[4,16160] -> [32,16160]` | 16 | 0.031 | N/A | N/A | 1023/0 `{'4':1023}` |
| `historical_4096_1024_bs4` | `dsv4.v1_moe_reduce_once_all_reduce` | `all_reduce` | `float32` | `[16384,4096] -> [16384,4096]` | 688 | 172.000 | N/A | N/A | 1023/0 `{'4':1023}` |
| `serving_mixed_112req_wave16` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `all_reduce` | `bfloat16` | `[2496,4096] -> [2496,4096]` | 2408 | 45.855 | 3351.6 | 5848 | 441/0 `{'1':112,'2':112,'4':56,'8':56,'16':105}` |
| `serving_mixed_112req_wave16` | `dsv4.embedding_all_reduce` | `all_reduce` | `bfloat16` | `[2496,4096] -> [2496,4096]` | 56 | 1.066 | 1492.4 | 136 | 441/0 `{'1':112,'2':112,'4':56,'8':56,'16':105}` |
| `serving_mixed_112req_wave16` | `dsv4.lm_head_all_gather` | `all_gather` | `float32` | `[16,16160] -> [128,16160]` | 56 | 0.432 | 11.6 | 136 | 441/0 `{'1':112,'2':112,'4':56,'8':56,'16':105}` |
| `serving_mixed_112req_wave16` | `dsv4.v1_moe_reduce_once_all_reduce` | `all_reduce` | `float32` | `[2496,4096] -> [2496,4096]` | 2408 | 91.711 | 1633.6 | 5848 | 441/0 `{'1':112,'2':112,'4':56,'8':56,'16':105}` |
| `prefix_multi_112req_wave16` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `all_reduce` | `bfloat16` | `[1024,4096] -> [1024,4096]` | 2064 | 16.125 | 3128.1 | 5848 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `all_reduce` | `bfloat16` | `[9216,4096] -> [9216,4096]` | 344 | 24.188 | 3128.1 | 5848 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.embedding_all_reduce` | `all_reduce` | `bfloat16` | `[1024,4096] -> [1024,4096]` | 48 | 0.375 | 1469.3 | 136 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.embedding_all_reduce` | `all_reduce` | `bfloat16` | `[9216,4096] -> [9216,4096]` | 8 | 0.562 | 1469.3 | 136 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.lm_head_all_gather` | `all_gather` | `float32` | `[16,16160] -> [128,16160]` | 56 | 0.432 | 10.8 | 136 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.v1_moe_reduce_once_all_reduce` | `all_reduce` | `float32` | `[1024,4096] -> [1024,4096]` | 2064 | 32.250 | 1567.7 | 5848 | 49/0 `{'16':49}` |
| `prefix_multi_112req_wave16` | `dsv4.v1_moe_reduce_once_all_reduce` | `all_reduce` | `float32` | `[9216,4096] -> [9216,4096]` | 344 | 48.375 | 1567.7 | 5848 | 49/0 `{'16':49}` |

### Mini Metadata And Runtime Overhead

These are non-collective owner rows from Target08 profile data. They matter for end-to-end parity but should not be treated as NCCL/backend mismatches.

| scenario | label | section | max-rank ms | count |
|---|---|---|---:|---:|
| `historical_4096_128_bs4` | `dsv4.prepare.prefill.attention_metadata` | `host` | 787.3 | 8 |
| `historical_4096_128_bs4` | `dsv4.prepare.decode.attention_metadata` | `host` | 346.4 | 1016 |
| `historical_4096_128_bs4` | `dsv4.metadata.decode.make_c4_sparse_indices` | `cuda` | 395.3 | 8 |
| `historical_4096_128_bs4` | `dsv4.metadata.decode.make_c128_indices` | `cuda` | 457.4 | 1024 |
| `historical_4096_128_bs4` | `dsv4.metadata.decode.make_component_page_tables` | `cuda` | 39.3 | 1024 |
| `historical_4096_128_bs4` | `dsv4.metadata.decode.make_write_locs` | `cuda` | 71.0 | 1024 |
| `serving_mixed_112req_wave16` | `dsv4.prepare.prefill.attention_metadata` | `host` | 700.5 | 56 |
| `serving_mixed_112req_wave16` | `dsv4.prepare.decode.attention_metadata` | `host` | 1336.5 | 3528 |
| `serving_mixed_112req_wave16` | `dsv4.metadata.decode.make_c4_sparse_indices` | `cuda` | 416.5 | 56 |
| `serving_mixed_112req_wave16` | `dsv4.metadata.decode.make_c128_indices` | `cuda` | 432.1 | 3584 |
| `serving_mixed_112req_wave16` | `dsv4.metadata.decode.make_component_page_tables` | `cuda` | 356.1 | 3584 |
| `serving_mixed_112req_wave16` | `dsv4.metadata.decode.make_write_locs` | `cuda` | 249.1 | 3584 |
| `prefix_multi_112req_wave16` | `dsv4.prepare.prefill.attention_metadata` | `host` | 876.9 | 56 |
| `prefix_multi_112req_wave16` | `dsv4.prepare.decode.attention_metadata` | `host` | 307.8 | 392 |
| `prefix_multi_112req_wave16` | `dsv4.metadata.decode.make_c4_sparse_indices` | `cuda` | 381.4 | 56 |
| `prefix_multi_112req_wave16` | `dsv4.metadata.decode.make_c128_indices` | `cuda` | 361.3 | 448 |
| `prefix_multi_112req_wave16` | `dsv4.metadata.decode.make_component_page_tables` | `cuda` | 327.3 | 448 |
| `prefix_multi_112req_wave16` | `dsv4.metadata.decode.make_write_locs` | `cuda` | 35.5 | 448 |

## vLLM Runtime Census And Static Fallback

Run: `decode_throughput_bs8`, prompt 4096, decode 128, batch 4, TP8, `max_num_batched_tokens=4096`, chunked prefill enabled, `enforce_eager=false`, `disable_custom_all_reduce=false`. The final probe passed with 512 output tokens and 6.2299 s elapsed.

Probe history:

- Attempt 1 failed during engine profile/compile because the first monkeypatch called `time.perf_counter()` inside `GroupCoordinator.all_reduce`; TorchDynamo reported an unsupported skipped function.
- Attempt 2 reached CUDA graph capture, then failed because V1 `collective_rpc` would not serialize Python function objects without pickle fallback.
- Final attempt set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` only for probe RPC reset/snapshot. The model communication path stayed default. Runtime probe then completed, but compiled backbone collectives bypassed Python recording, so backbone rows below are explicit static fallback rows.

Static fallback calculation for backbone rows uses runtime-observed 132 forward steps per rank from logits probe, 16,384 prompt token-rows plus 512 generated token-rows, hidden size 4096, 43 layers, TP8. It is a source-derived count/bytes census, not a direct Python runtime counter.

| label | source | op | dtype | shape family | count | bytes GiB | fallback reason |
|---|---|---|---|---|---:|---:|---|
| `vllm.embedding_all_reduce` | static source-derived | `all_reduce` | `bfloat16` | prefill chunks up to `[4096,4096]` plus decode batches up to `[4,4096]` | 1056 | 1.031 | TorchDynamo/CUDA graph bypassed Python `GroupCoordinator` probe for compiled model collectives |
| `vllm.attn.wo_b.row_parallel_projection_all_reduce` | static source-derived | `all_reduce` | `bfloat16` | same token-row family, repeated for 43 layers | 45408 | 44.344 | TorchDynamo/CUDA graph bypassed Python `GroupCoordinator` probe for compiled model collectives |
| `vllm.moe.reduce_once_all_reduce` | static source-derived | `all_reduce` | `bfloat16` | same token-row family, repeated for 43 layers | 45408 | 44.344 | TorchDynamo/CUDA graph bypassed Python probe; dtype inferred from MXFP4 bf16 activation support and hidden-state `empty_like` output |
| `vllm.logits_all_gather` | runtime probe | `all_gather` | `bfloat16` | `[1,16160] -> [1,129280]` | 16 | 0.004 | none |
| `vllm.logits_all_gather` | runtime probe | `all_gather` | `bfloat16` | `[2,16160] -> [2,129280]` | 16 | 0.008 | none |
| `vllm.logits_all_gather` | runtime probe | `all_gather` | `bfloat16` | `[3,16160] -> [3,129280]` | 24 | 0.017 | none |
| `vllm.logits_all_gather` | runtime probe | `all_gather` | `bfloat16` | `[4,16160] -> [4,129280]` | 1000 | 0.963 | none |

Observed vLLM runtime communication total from the Python-visible probe is 1056 all-gathers and 1.065 GB, all logits. Backbone all-reduces are present by source and graph registration, but not Python-visible during replay.

## Graph Replay / Eager State

Mini:

| scenario | graph enabled | captured/requested bs | replay/eager | replay by padded size |
|---|---|---|---:|---|
| `historical_4096_128_bs4` | true | `[1,2,4,8,16]` / `[1,2,4,8,16]` | 127/0 | `{'4':127}` |
| `historical_4096_1024_bs4` | true | `[1,2,4,8,16]` / `[1,2,4,8,16]` | 1023/0 | `{'4':1023}` |
| `serving_mixed_112req_wave16` | true | `[1,2,4,8,16]` / `[1,2,4,8,16]` | 441/0 | `{'1':112,'2':112,'4':56,'8':56,'16':105}` |
| `prefix_multi_112req_wave16` | true | `[1,2,4,8,16]` / `[1,2,4,8,16]` | 49/0 | `{'16':49}` |

vLLM:

- `enforce_eager=false`.
- `cudagraph_capture_sizes=[1,2,4]`, `max_cudagraph_capture_size=4`.
- Runtime stdout showed piecewise prefill/decode and full decode CUDA graph capture completed, then custom all-reduce graph address registration reached 522 addresses per rank.
- Runtime logits all-gather rows are marked `eager_or_python_capture_call`; compiled backbone all-reduces are inside torch.compile/CUDA graph/custom op replay and are static fallback in this census.

## Mismatch Classification

| Severity | Category | Finding | Impact |
|---|---|---|---|
| High | MoE reduce-once dtype/bytes | mini reduces MoE final output as fp32 (`dsv4.v1_moe_reduce_once_all_reduce`), while vLLM SM80 source path reduces bf16 hidden-state output. | Dominant byte delta: mini MoE reduce bytes are 2x attention bytes for the same shape. This is the largest path-level mismatch. |
| Medium | Runtime observability | vLLM backbone collectives are compiled/captured, so Python monkeypatch counters cannot directly observe them on the default path. | Count/bytes rows for vLLM backbone are static fallback. This is an instrumentation limitation, not evidence that collectives are absent. |
| Low to medium | LM head logits dtype/shape | mini all-gathers fp32 logits as `[B,16160] -> [B*8,16160]`; vLLM runtime all-gathers bf16 logits as `[B,16160] -> [B,129280]`. | Bytes are small relative to per-layer all-reduces, but the dtype mismatch is real. |
| Low | Backend implementation | mini uses its `DistributedCommunicator` plugin path over NCCL; vLLM default uses custom-op TP group path with custom all-reduce enabled. | Worth testing in Target10.2 after dtype mismatch is isolated. |
| Informational | Metadata/runtime overhead | mini DSV4 metadata/page-table work is visible and sometimes large; vLLM probe did not expose a directly matching metadata owner surface. | Keep separate from collective/backend parity conclusions. |

## Recommendation For TARGET 10.2

Recommendation: yes, enter TARGET 10.2, but keep it narrow.

The communication owner boundaries largely match vLLM: embedding all-reduce, attention `wo_b` row-parallel all-reduce, MoE reduce-once final all-reduce, no separate shared/routed reductions on the current SM80 path, and decode graph replay is enabled with zero mini eager decode. The blocker for a clean backend-only experiment is the high-severity MoE dtype/bytes mismatch: mini fp32 reduce-once versus vLLM bf16 static path.

Suggested 10.2 order:

1. First isolate MoE reduce dtype/bytes with a correctness gate. Do not mix this with backend toggles.
2. Then run backend experiments only on a fixed path: mini NCCL/plugin path versus vLLM default custom-op path, with graph replay enabled.
3. Keep logits dtype/all-gather as a small secondary item unless it shows measurable runtime impact.

