# TARGET 07.2: Communication and CUDA Graph

Status: partial. Communication observability and PyNCCL correctness are in
place. Guarded DSV4 decode CUDA graph now captures and replays `[1,2,4]`, but
the default DSV4 path remains exact eager and unchanged.

## Implemented

- Added semantic communication counters under `DistributedCommunicator`.
- Labeled DSV4 collectives:
  - `dsv4.embedding_all_reduce`
  - `dsv4.row_parallel_projection_all_reduce`
  - `dsv4.routed_expert_all_reduce`
  - `dsv4.shared_expert_all_reduce`
  - `dsv4.v1_moe_reduce_once_all_reduce`
  - `dsv4.lm_head_all_gather`
- Preserved the V1 late MoE reduce invariant. In V1, routed and shared outputs
  are summed locally and only `dsv4.v1_moe_reduce_once_all_reduce` fires.
- Added fp32 to the PyNCCL dtype map.
- Extended the PyNCCL direct test script to cover fp16, bf16, and fp32
  all-reduce/all-gather correctness.
- Added opt-in benchmark/text-smoke runtime variants:
  - `v1_moe_pynccl`
  - `v1_moe_graph`
  - `v1_moe_graph_hc`
  - `v1_moe_graph_sample`
  - `v1_moe_graph_pynccl`
- Added guarded DSV4 decode CUDA graph opt-in via `allow_dsv4_cuda_graph`.
  Default DSV4 graph behavior is unchanged and still disabled.
- Added graph capture diagnostics in `GraphRunner.capture_status`.
- Added graph replay coverage diagnostics in `GraphRunner.capture_status`:
  replay counts by actual batch size, replay counts by padded graph size,
  eager decode fallback counts by batch size, and replay input-copy bytes.
- Added opt-in greedy sampler capture for decode CUDA graph. This keeps
  `v1_moe_graph` unchanged and is exposed through the experimental
  `v1_moe_graph_sample` variant.
- Added explicit benchmark NVTX ranges for warmup/repeat profiling and made the
  Nsight sqlite summarizer report missing requested windows instead of silently
  reusing total-scope metrics.
- Added profiling-only `batch_prepare:*` NVTX ranges. Setting
  `MINISGL_BENCH_SYNC_PREPARE_NVTX=1` synchronizes inside those ranges so Nsight
  can attribute async scheduler/metadata kernels to prepare instead of leaving
  them in the gap before forward.
- Reused synchronized `copy_done` CUDA events in the scheduler path so each
  decode batch no longer creates a fresh event after graph replay.
- Fixed a DSV4 decode graph exactness risk for compressed/indexer KV stores.
  Capture metadata now uses fixed-size C4/C128 write-location tensors with
  `-1` masked rows, so graph capture includes the store path instead of
  skipping it when dummy capture positions are not on compression boundaries.
- Added a small Triton staging kernel for graph replay metadata that computes
  the fixed C4 and C128 masked write-location buffers in one launch.
- Added nested `batch_forward_enqueue:*` benchmark NVTX ranges and
  `forward_enqueue_s` report fields. The existing synchronized `forward_s`
  timing remains unchanged.
- Reduced replay metadata staging launch noise by only filling `_copy_2d`
  capture-buffer tails when the source metadata width does not cover the
  destination width.
- Bound DSV4 capture metadata `raw_out_loc` and `positions` to the CUDA graph
  input buffers, avoiding duplicate replay-time copies of those tensors.
- Added optional Nsight node-level graph tracing through
  `NSYS_CUDA_GRAPH_TRACE=node`.
- Extended the NVTX range summarizer with graph-node/non-graph-node kernel
  attribution and a `--scan-events` mode for large node-trace sqlite exports.
- Added opt-in DSV4 graph-capture NVTX ranges controlled by
  `MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1`. This is profiling-only and leaves the
  default model path unchanged.
- Extended the node-trace summarizer to map replayed `graphNodeId` kernel rows
  back to capture-time `dsv4.*` NVTX ranges, including a layer-collapsed module
  grouping such as `dsv4.layer*.mlp.routed`.
- Added opt-in sm80 Triton HC split/pre and post helpers under
  `MINISGL_DSV4_SM80_HC=1`, exposed as `v1_moe_graph_hc`. This reduces DSV4
  graph-body PyTorch small-kernel volume while keeping the default path
  unchanged.
- Added opt-in sm80 bf16 RMSNorm helper under
  `MINISGL_DSV4_SM80_RMSNORM=1`, exposed as `v1_moe_graph_hc_rmsnorm`. This
  removes more graph-body PyTorch norm nodes while keeping the default path
  unchanged.
- Added `v1_moe_graph_hc_rmsnorm_fp8gemm` as a guarded experiment for the
  existing `MINISGL_DSV4_SM80_FP8_GEMM=1` Triton path. This was measured as a
  negative graph-body result and is not a best path.
- Added `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm` as a guarded selective FP8 GEMM
  experiment for attention `wq_b` only. This keeps the global FP8 GEMM path off
  and gives a small positive graph-body result.
- Added `v1_moe_graph_hc_rmsnorm_wqb_woa` as a guarded experiment for the
  existing `MINISGL_DSV4_SM80_WO_A_BF16=1` path. A decode-like microbench was
  positive, but the target 4096/128 macro regressed, so this is not a best path.
- Added `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm` as a guarded selective FP8
  GEMM experiment for attention `wo_b` on top of the positive `wq_b` path. This
  is a positive exact graph-body path, but the gain is incremental.
- Added `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm` as a guarded
  selective FP8 GEMM experiment for the indexer `wq_b` projection on top of the
  positive attention `wq_b` + `wo_b` path. This is a positive exact graph-body
  path.
- Added `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm` as a guarded
  selective FP8 GEMM experiment for shared-expert gate/up and down projections.
  The isolated microbench was positive, but the 4096/128 macro regressed, so it
  is not a best path.
- Added `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache` as a guarded exact
  graph-body experiment for MoE gate fp32 weight caching. This removes repeated
  per-replay `gate.weight.float()` work while keeping the gate computation in
  fp32 and was a small positive exact graph-body path.
- Added `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache` as a
  guarded exact graph-body experiment for indexer-store norm fp32 weight
  caching on top of gatecache. This removes repeated per-replay
  `indexer.compressor.norm.weight.float()` work while keeping the norm multiply
  in fp32 and was the previous best exact graph-body path before the
  shared-activation `wq_a/wkv` experiment.
- Added
  `v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache` as a
  guarded selective FP8 GEMM experiment for attention `wq_a` on top of the then
  best graph path. It is correct and CUDA-graph replayable, but both the
  decode-like microbench and the 4096/128 macro regressed, so it is not a best
  path.
- Added
  `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache` as a
  guarded vLLM-aligned attention projection experiment. vLLM combines
  DeepSeek V4 `wq_a` and `wkv` behind a fused `fused_wqa_wkv` projection
  boundary; this mini path does not add a true fused GEMM/custom op, but it does
  reuse one exact FP8 activation quantization for `wq_a` and `wkv`, removing a
  duplicated graph-body projection preparation path while leaving the default
  exact eager path unchanged.
- Added
  `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`
  as a guarded vLLM-aligned q/KV graph-body fusion experiment. The current DSV4
  backend already enables the exact KV bf16 norm/rope/cache-store path, so this
  variant fuses q head RMSNorm+RoPE with KV RMSNorm+RoPE+cache-store into one
  opt-in Triton launch. This is closer to vLLM's fused qnorm/rope/KV insert
  boundary, keeps the default path unchanged, and leaves the logical
  communication pattern unchanged.
- Added
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`
  as a guarded vLLM-aligned merged `wq_a/wkv` projection-cache experiment. It
  caches a fused bf16 dequantized FP8 weight for `wq_a + wkv`, runs one
  `F.linear`, then splits the result into q-lora and KV branches. This better
  matches vLLM's `fused_wqa_wkv` boundary while keeping the default path
  unchanged. The first attempt exposed a graph-capture blocker: the KV split was
  a non-contiguous view, so the q/KV norm-rope-store Triton path refused it and
  capture fell into the old `k_norm_rope_cache_fallback`, where
  `bool(torch.any(valid))` is capture-unsafe. The initial guarded workaround
  made the KV branch contiguous and restored `[4,2,1]` capture and replay.
- Removed that forced KV materialization by teaching the fused q/KV
  norm-rope-store Triton wrapper to accept non-contiguous KV split views with
  `stride(-1) == 1` and pass `kv.stride(0)` into the kernel. This mirrors the
  vLLM `fused_q_kv_rmsnorm` boundary, which reads split views by stride before
  materializing its normalized outputs, while preserving mini's exact default
  path and logical communication pattern.
- Added
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`
  as a guarded current-best graph variant with greedy sampler replay included
  in the decode CUDA graph. This keeps default sampling behavior unchanged and
  removes the graph-outside greedy `argmax` launch path for the exact
  temperature-0 benchmark.

## Key Artifacts

| Artifact | Purpose |
| --- | --- |
| `summaries/text_smoke_v1_moe_pynccl_tp8.json` | TP8 DSV4 text smoke with PyNCCL. |
| `summaries/text_smoke_v1_moe_graph_tp8.json` | TP8 DSV4 graph probe with `[1,2,4]`. |
| `summaries/text_smoke_v1_moe_graph_tp8_replay_observability.json` | TP8 graph smoke validating replay coverage counters. |
| `summaries/text_smoke_v1_moe_graph_sample_tp8.json` | TP8 DSV4 graph smoke with greedy sampler captured in graph. |
| `summaries/text_smoke_v1_moe_graph_event_pool_tp8.json` | TP8 DSV4 graph smoke after `copy_done` event pooling. |
| `summaries/text_smoke_v1_moe_graph_masked_compress_tp8.json` | TP8 DSV4 graph smoke after masked compressed/indexer store capture fix. |
| `summaries/text_smoke_v1_moe_graph_fused_masked_locs_tp8.json` | TP8 DSV4 graph smoke after fused masked-loc replay staging. |
| `summaries/text_smoke_v1_moe_graph_enqueue_nvtx_tp8.json` | TP8 DSV4 graph smoke after adding nested enqueue NVTX ranges. |
| `summaries/text_smoke_v1_moe_graph_tail_fill_tp8.json` | TP8 DSV4 graph smoke after tail-only replay metadata fills. |
| `summaries/text_smoke_v1_moe_graph_bound_metadata_tp8.json` | TP8 DSV4 graph smoke after binding capture metadata to graph input buffers. |
| `summaries/text_smoke_v1_moe_graph_hc_tp8.json` | TP8 DSV4 graph smoke with experimental sm80 HC helpers. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_tp8.json` | TP8 DSV4 graph smoke with experimental sm80 HC and RMSNorm helpers. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fp8gemm_tp8.json` | TP8 DSV4 graph smoke with experimental sm80 HC/RMSNorm helpers and FP8 GEMM. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_tp8.json` | TP8 DSV4 graph smoke with selective attention `wq_b` FP8 GEMM. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_woa_tp8.json` | TP8 DSV4 graph smoke with selective `wq_b` FP8 GEMM plus `wo_a` helper. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_tp8.json` | TP8 DSV4 graph smoke with selective attention `wq_b` and `wo_b` FP8 GEMM. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_tp8.json` | TP8 DSV4 graph smoke with selective attention `wq_b`/`wo_b` and indexer `wq_b` FP8 GEMM. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm_tp8.json` | TP8 DSV4 graph smoke with selective shared-expert FP8 GEMM added to the then-best graph path. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_tp8.json` | TP8 DSV4 graph smoke with exact MoE gate fp32 weight cache added to the then-best graph path. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with exact indexer-store norm fp32 weight cache added to the gatecache graph path. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with selective attention `wq_a` FP8 GEMM added to the then-best graph path. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with vLLM-aligned shared-activation `wq_a/wkv` projection added to the previous best graph path. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with vLLM-aligned shared-activation `wq_a/wkv` plus fused q norm/rope and KV norm/rope/cache-store. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke after removing the forced KV contiguous copy from cached fused `wq_a/wkv`. |
| `summaries/text_smoke_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_tp8.json` | TP8 DSV4 graph smoke with current best graph-body path plus greedy sampler replay. |
| `summaries/perf_smoke_v1_moe_pynccl_tp8/` | Small TP8 perf smoke validating communication counters. |
| `summaries/dsv4_target07_2_fp8_linear_microbench_m4.json` | Microbench for the existing sm80 FP8 GEMM path at decode-like `m=4`. |
| `summaries/dsv4_target07_2_fp8_wq_a_microbench_m4_qwqa.json` | Decode-like microbench for selective attention `wq_a` FP8 GEMM. |
| `summaries/dsv4_target07_2_fp8_wo_b_microbench_m4.json` | Decode-like microbench for selective attention `wo_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_fp8_indexer_wq_b_microbench_m4.json` | Decode-like microbench for selective indexer `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_fp8_shared_experts_microbench_m4.json` | Decode-like microbench for selective shared-expert gate/up and down FP8 GEMM. |
| `summaries/dsv4_target07_2_wo_a_microbench_t4_g8_d512_r1024.json` | Decode-like microbench for the existing sm80 `wo_a` helper. |
| `summaries/dsv4_target07_2_mini_v1_moe_pynccl_4096x128_bs4_warmup1/` | 4096/128/bs4 PyNCCL macro report. |
| `summaries/dsv4_target07_2_mini_v1_moe_pynccl_4096x1024_bs4_warmup1/` | 4096/1024/bs4 PyNCCL macro report. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_4096x128_bs4_warmup1/` | 4096/128/bs4 guarded CUDA graph macro report. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_4096x1024_bs4_warmup1/` | 4096/1024/bs4 guarded CUDA graph macro report. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_sample_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with greedy sampler captured in graph. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_sample_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with greedy sampler captured in graph. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_event_pool_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after `copy_done` event pooling. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after masked compressed/indexer store capture fix. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_masked_compress_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro after masked compressed/indexer store capture fix. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_fused_masked_locs_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after fused masked-loc replay staging. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_tail_fill_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after tail-only replay metadata fills. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after binding capture metadata inputs. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with experimental sm80 HC helpers. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with experimental sm80 HC helpers. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with experimental sm80 HC and RMSNorm helpers. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with experimental sm80 HC and RMSNorm helpers. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fp8gemm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 negative graph macro for experimental sm80 FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with selective attention `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with selective attention `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_woa_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 negative graph macro for selective `wq_b` FP8 GEMM plus `wo_a` helper. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with selective attention `wq_b` and `wo_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with selective attention `wq_b` and `wo_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with selective attention `wq_b`/`wo_b` and indexer `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with selective attention `wq_b`/`wo_b` and indexer `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 negative graph macro for selective shared-expert FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with exact MoE gate fp32 weight cache. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with exact MoE gate fp32 weight cache. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with exact indexer-store norm fp32 weight cache on top of gatecache. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with exact indexer-store norm fp32 weight cache on top of gatecache. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 negative graph macro for selective attention `wq_a` FP8 GEMM. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with vLLM-aligned shared-activation `wq_a/wkv` projection. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with vLLM-aligned shared-activation `wq_a/wkv` projection. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with vLLM-aligned shared-activation `wq_a/wkv` plus fused q/KV norm-rope-store. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with vLLM-aligned shared-activation `wq_a/wkv` plus fused q/KV norm-rope-store. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro after the q/KV norm-rope-store wrapper accepted strided KV split views. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_4096x128_bs4_np72_warmup1/` | 4096/128/bs4 graph macro with current best graph-body path plus greedy sampler replay. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_4096x1024_bs4_np96_warmup1/` | 4096/1024/bs4 graph macro with current best graph-body path plus greedy sampler replay. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_4096x128_bs4_replay_schedule_warmup1/` | Full-KV 4096/128/bs4 graph macro with replay and schedule summaries. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_schedule_probe/` | Small-KV schedule probe proving 72 pages are enough for 4096/128/bs4 graph batch4. |
| `summaries/dsv4_target07_2_mini_v1_moe_graph_pynccl_4096x128_bs4_partial/` | Partial combined graph+PyNCCL run. Target decode scenario completed; later non-target scenarios were interrupted. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_warmup1_rank0.{json,md}` | Successful rank0 Nsight summary for guarded graph 4096/128/bs4. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_4096x128_bs4_warmup1/` | Workload report from the guarded graph Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_warmup1_nvtx_repeat_rank0.{json,md}` | Corrected rank0 Nsight summary with `repeat:decode_throughput_bs8:0` NVTX window found. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_4096x128_bs4_warmup1_nvtx_repeat/` | Workload report from the corrected NVTX-window graph Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Corrected rank0 Nsight summary with small-KV true batch4 replay. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the small-KV true batch4 Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_batchnvtx_warmup1_rank0.{json,md}` | Rank0 Nsight summary with small-KV true batch4 replay and batch-level NVTX ranges. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_batchnvtx_warmup1_rank0_batch_forward.{json,md}` | Prefill-vs-decode batch-forward Nsight split for the true-batch4 graph run. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_4096x128_bs4_np72_batch4_batchnvtx_warmup1/` | Workload report from the batch-level NVTX true-batch4 Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync2_warmup1_rank0.{json,md}` | Rank0 Nsight summary with profiling-only prepare NVTX synchronization. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync2_warmup1_rank0_prepare_forward.{json,md}` | Prepare-vs-forward Nsight split for true-batch4 graph run. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync2_warmup1/` | Workload report from the prepare-sync attribution run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_event_pool_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after `copy_done` event pooling. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_event_pool_4096x128_bs4_np72_batch4_warmup1_rank0_batch_forward.{json,md}` | Prefill-vs-decode batch-forward Nsight split after event pooling. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_event_pool_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the event-pool Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after masked compressed/indexer store capture fix. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_batch4_warmup1_rank0_batch_forward.{json,md}` | Prefill-vs-decode batch-forward Nsight split after masked compressed/indexer store capture fix. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the masked-store Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_fused_masked_locs_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after fused masked-loc replay staging. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_fused_masked_locs_4096x128_bs4_np72_batch4_warmup1_rank0_batch_forward.{json,md}` | Prefill-vs-decode batch-forward Nsight split after fused masked-loc replay staging. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_fused_masked_locs_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the fused masked-loc Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_enqueue_nvtx_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after adding nested enqueue NVTX ranges. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_enqueue_nvtx_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Prefill/decode forward-vs-enqueue Nsight split. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_enqueue_nvtx_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the enqueue-NVTX Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_tail_fill_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after tail-only replay metadata fills. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_tail_fill_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Prefill/decode forward-vs-enqueue Nsight split after tail-only fills. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_tail_fill_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the tail-fill Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Rank0 Nsight summary after binding DSV4 capture metadata inputs. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Prefill/decode forward-vs-enqueue Nsight split after bound metadata inputs. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the bound-metadata Nsight run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_node_trace_probe_4096x16_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Short node-level graph trace probe proving graph body kernels are visible through `graphNodeId`. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 rank0 node-level graph trace summary. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 graph-body kernel attribution for prefill/decode forward and enqueue ranges. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the full node-level graph trace run. |
| `summaries/text_smoke_v1_moe_graph_capture_nvtx_tp8.json` | TP8 DSV4 graph text smoke with opt-in capture NVTX enabled. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_probe_4096x16_bs4_np72_batch4_warmup1_rank0.{json,md}` | Short node-level graph trace with opt-in capture NVTX. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_probe_4096x16_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Short node trace mapping replay `graphNodeId` kernels back to capture-time DSV4 module ranges. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_capture_nvtx_node_trace_probe_4096x16_bs4_np72_batch4_warmup1/` | Workload report from the capture-NVTX node trace probe. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with opt-in capture NVTX. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` to DSV4 module attribution. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the full capture-NVTX node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with opt-in capture NVTX and sm80 HC helpers. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after sm80 HC helpers. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the sm80 HC capture-NVTX node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with opt-in capture NVTX, sm80 HC helpers, and sm80 RMSNorm. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after sm80 RMSNorm. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue_top80.{json,md}` | Expanded top-80 attribution showing the norm-node reductions. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the sm80 HC+RMSNorm capture-NVTX node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with selective attention `wq_b` FP8 GEMM. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after selective `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the selective `wq_b` FP8 GEMM node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with selective attention `wq_b` and `wo_b` FP8 GEMM. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after selective `wq_b` and `wo_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the selective `wq_b` and `wo_b` FP8 GEMM node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with selective attention `wq_b`/`wo_b` and indexer `wq_b` FP8 GEMM. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after selective indexer `wq_b` FP8 GEMM. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the selective attention `wq_b`/`wo_b` and indexer `wq_b` FP8 GEMM node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with exact MoE gate fp32 weight cache. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after exact MoE gate fp32 weight cache. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the exact MoE gate fp32 weight cache node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with exact indexer-store norm fp32 weight cache. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after exact indexer-store norm fp32 weight cache. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the exact indexer-store norm fp32 weight cache node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with vLLM-aligned shared-activation `wq_a/wkv` projection. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after vLLM-aligned shared-activation `wq_a/wkv` projection. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the vLLM-aligned shared-activation `wq_a/wkv` node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with vLLM-aligned fused q/KV norm-rope-store. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after vLLM-aligned fused q/KV norm-rope-store. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the fused q/KV norm-rope-store node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution after cached fused bf16 `wq_a/wkv` weight. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the cached fused `wq_a/wkv` node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace after removing the forced KV contiguous copy. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution for the strided-KV cleanup. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the strided-KV node trace run. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.{json,md}` | Full 4096/128/bs4 node-level graph trace with current best graph-body path plus greedy sampler replay. |
| `summaries/nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0_forward_enqueue.{json,md}` | Full 4096/128/bs4 replay `graphNodeId` attribution for current best plus greedy sampler replay. |
| `summaries/dsv4_target07_2_nsys_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1/` | Workload report from the current-best-plus-sampler node trace run. |
| `raw/dsv4_target07_2_nsys_mini_v1_moe_pynccl_4096x128_bs4_warmup1/run_config.json` | Failed PyNCCL nsys attempt config. |

## Results

| Variant | Shape | Output tok/s | Decode tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| `v1_moe_pynccl` | 4096/128/bs4 | 7.6321 | 11.3171 | Improves over fair mini 5.5071 output tok/s. |
| `v1_moe_pynccl` | 4096/1024/bs4 | 10.6461 | 11.3421 | Only slightly above fair mini 10.5768 output tok/s. |
| `v1_moe_graph` | 4096/128/bs4 | 11.7720 | 23.8733 | Captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph` | 4096/128/bs4 | 11.7715 | 23.8823 | Replay/schedule recheck: full-KV batch4, decode `bs4` x127, eager decode fallback 0. |
| `v1_moe_graph` | 4096/1024/bs4 | 20.9756 | 23.8743 | Captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_sample` | 4096/128/bs4 | 11.7677 | 23.8702 | Greedy sampler captured; true bs4; greedy sample replay count 254. |
| `v1_moe_graph_sample` | 4096/1024/bs4 | 21.0043 | 23.9124 | Greedy sampler captured; true bs4; greedy sample replay count 2046. |
| `v1_moe_graph` + event pool | 4096/128/bs4 | 11.7733 | 23.8741 | Removes per-decode `cudaEventCreateWithFlags`; throughput unchanged. |
| `v1_moe_graph` + masked compressed store | 4096/128/bs4 | 11.6939 | 23.5544 | Fixed graph exactness risk for compressed/indexer stores; replay count 254. |
| `v1_moe_graph` + masked compressed store | 4096/1024/bs4 | 20.7570 | 23.5922 | Fixed graph exactness risk for compressed/indexer stores; replay count 2046. |
| `v1_moe_graph` + fused masked-loc staging | 4096/128/bs4 | 11.6911 | 23.5639 | Reduces graph-adjacent loc-update kernels; throughput unchanged. |
| `v1_moe_graph` + tail-only metadata fills | 4096/128/bs4 | 11.7061 | 23.6057 | Reduces replay metadata fill launches; throughput remains within noise. |
| `v1_moe_graph` + bound metadata inputs | 4096/128/bs4 | 11.6963 | 23.5939 | Removes duplicate replay copies of `out_loc`/`positions`; throughput remains within noise. |
| `v1_moe_graph_hc` | 4096/128/bs4 | 12.6437 | 27.1310 | Opt-in sm80 HC helpers; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc` | 4096/1024/bs4 | 23.5562 | 27.1983 | Opt-in sm80 HC helpers; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm` | 4096/128/bs4 | 12.7926 | 27.6019 | Opt-in sm80 HC + bf16 RMSNorm helpers; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc_rmsnorm` | 4096/1024/bs4 | 23.9128 | 27.6512 | Opt-in sm80 HC + bf16 RMSNorm helpers; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm_fp8gemm` | 4096/128/bs4 | 11.8707 | 23.6431 | Negative result: existing sm80 FP8 GEMM path slows the full graph body despite a faster `wq_b` microbench. |
| `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm` | 4096/128/bs4 | 12.8048 | 27.6743 | Selective attention `wq_b` FP8 GEMM; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm` | 4096/1024/bs4 | 23.9902 | 27.7537 | Selective attention `wq_b` FP8 GEMM; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm_wqb_woa` | 4096/128/bs4 | 12.4909 | 26.2303 | Negative result: `wo_a` microbench improved 1.78x, but the full macro regressed; 4096/1024 and nsys were not run. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm` | 4096/128/bs4 | 12.8352 | 27.7860 | Selective attention `wq_b` + `wo_b` FP8 GEMM; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm` | 4096/1024/bs4 | 24.0576 | 27.8467 | Selective attention `wq_b` + `wo_b` FP8 GEMM; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm` | 4096/128/bs4 | 12.8670 | 27.9587 | Selective attention `wq_b`/`wo_b` plus indexer `wq_b` FP8 GEMM; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm` | 4096/1024/bs4 | 24.1831 | 28.0048 | Selective attention `wq_b`/`wo_b` plus indexer `wq_b` FP8 GEMM; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm` | 4096/128/bs4 | 12.5036 | 26.2507 | Negative result: selective shared-expert FP8 GEMM microbench was positive, but the full target-shape macro regressed; 4096/1024 and nsys were not run. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache` | 4096/128/bs4 | 12.8804 | 28.0294 | Exact MoE gate fp32 weight cache on top of the previous best graph path; captured `[4,2,1]`; replay count 254. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache` | 4096/1024/bs4 | 24.2265 | 28.0659 | Exact MoE gate fp32 weight cache on top of the previous best graph path; captured `[4,2,1]`; replay count 2046. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 12.8950 | 28.0521 | Exact indexer-store norm fp32 weight cache on top of gatecache; captured `[4,2,1]`; replay count 254. Tiny positive. |
| `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/1024/bs4 | 24.2286 | 28.0716 | Exact indexer-store norm fp32 weight cache on top of gatecache; captured `[4,2,1]`; replay count 2046. Tiny positive. |
| `v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 12.5642 | 26.5877 | Negative result: selective attention `wq_a` FP8 GEMM microbench regressed and the full target-shape macro regressed; captured `[4,2,1]`; replay count 254. 4096/1024 and nsys were not run. |
| `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 12.9822 | 28.3548 | vLLM-aligned shared-activation `wq_a/wkv` projection; captured `[4,2,1]`; replay count 254. Positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/1024/bs4 | 24.5185 | 28.4276 | vLLM-aligned shared-activation `wq_a/wkv` projection; captured `[4,2,1]`; replay count 2046. Positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 12.9881 | 28.3571 | vLLM-aligned shared-activation `wq_a/wkv` plus fused q/KV norm-rope-store; captured `[4,2,1]`; replay count 254. Tiny positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/1024/bs4 | 24.5264 | 28.4586 | vLLM-aligned shared-activation `wq_a/wkv` plus fused q/KV norm-rope-store; captured `[4,2,1]`; replay count 2046. Tiny positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 13.1928 | 29.4214 | Cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store; captured `[4,2,1]`; replay count 254. Positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/1024/bs4 | 25.3027 | 29.5032 | Cached fused bf16 `wq_a/wkv` weight plus fused q/KV norm-rope-store; captured `[4,2,1]`; replay count 2046. Positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 13.2070 | 29.4173 | Same guarded path after removing the forced KV contiguous copy; captured `[4,2,1]`; replay count 254. Macro-neutral/slightly positive on output tok/s. |
| `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/128/bs4 | 13.2160 | 29.4655 | Current best graph-body path with greedy sampler replay captured; replay count 254; greedy sample replay count 254. Small graph-surface positive. |
| `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache` | 4096/1024/bs4 | 25.3076 | 29.5035 | Current best graph-body path with greedy sampler replay captured; replay count 2046; greedy sample replay count 2046. Small graph-surface positive. |
| `v1_moe_graph_pynccl` | 4096/128/bs4 | 8.0431 | 13.0295 | Captured `[4,2,1]`; replay count 1524 in a partial matrix run. Slower than pure graph. |

The hard TARGET 07 win line remains 114.07 output tok/s. The best
guarded bf16-direct graph variant after the masked compressed/indexer store fix
is `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`
at 25.3076
output tok/s on 4096/1024/bs4, roughly
2.39x over the 4096/1024 fair mini baseline of 10.5768 output tok/s, but still
far from the old vLLM-based serving baseline.

## Communication Distribution

For 4096/1024/bs4 `v1_moe_pynccl`:

| Label | Count | Bytes | Share of bytes |
| --- | ---: | ---: | ---: |
| `dsv4.v1_moe_reduce_once_all_reduce` | 352,256 | 115,404,701,696 | 60.3% |
| `dsv4.row_parallel_projection_all_reduce` | 352,256 | 57,702,350,848 | 30.1% |
| `dsv4.lm_head_all_gather` | 8,192 | 16,944,988,160 | 8.9% |
| `dsv4.embedding_all_reduce` | 8,192 | 1,341,915,136 | 0.7% |

For 4096/128/bs4 `v1_moe_pynccl`:

| Label | Count | Bytes | Share of bytes |
| --- | ---: | ---: | ---: |
| `dsv4.v1_moe_reduce_once_all_reduce` | 44,032 | 95,204,933,632 | 65.2% |
| `dsv4.row_parallel_projection_all_reduce` | 44,032 | 47,602,466,816 | 32.6% |
| `dsv4.lm_head_all_gather` | 1,024 | 2,118,123,520 | 1.5% |
| `dsv4.embedding_all_reduce` | 1,024 | 1,107,034,112 | 0.8% |

## PyNCCL

Direct PyNCCL TP2 correctness passed for fp16, bf16, and fp32 all-reduce and
all-gather. TP8 DSV4 text smoke with `v1_moe_pynccl` also passed. The previous
fp32 lm_head all-gather dtype blocker is fixed.

PyNCCL should not be promoted as a DSV4 TP8 default yet. It improves the short
4096/128 macro but barely moves the main 4096/1024 target shape, and the
rank0 Nsight profile attempt with PyNCCL did not complete.

## CUDA Graph

DSV4 graph opt-in now captures and replays decode batch sizes `[1,2,4]` after
removing the graph-hostile host scalar extraction from the DSV4 bf16 indexer
selection path. Capture remains explicitly guarded by `allow_dsv4_cuda_graph`;
prefill stays eager.

Previous blocker from `text_smoke_v1_moe_graph_tp8.json`:

- capture size: first attempted `bs=4`
- exception: `torch.AcceleratorError`, `cudaErrorStreamCaptureInvalidated`
- call site:
  `python/minisgl/kernel/deepseek_v4.py:indexer_bf16_logits_fallback`
- offending operation:
  `int(seq_lens.clamp_min(0).max().item())`

Fix:

- `indexer_bf16_logits` accepts an optional static `max_seq_len`.
- During CUDA graph capture, the fallback wrapper uses
  `page_table.shape[1] * page_size` instead of `seq_lens.max().item()`.
- If capture is active and the Triton indexer path cannot run, the wrapper now
  raises a precise error instead of silently entering the torch fallback.

Current validation:

- TP8 text smoke `v1_moe_graph`: captured `[4,2,1]`, replay count 3.
- TP8 text smoke `v1_moe_graph_pynccl`: captured `[4,2,1]`, replay count 3.
- TP8 text smoke `v1_moe_graph` with replay observability:
  actual batch 3 replayed 3 times, padded graph size 4 replayed 3 times,
  eager decode fallback count 0, replay input-copy bytes 144.
- TP8 text smoke `v1_moe_graph_sample`: captured `[4,2,1]`; greedy sampler
  replay count 3; warning status only because `max_tokens=4` truncates the first
  answer before the expected substring appears.
- 4096/128/bs4 `v1_moe_graph`: captured `[4,2,1]`, replay count 254.
- 4096/128/bs4 full-KV replay/schedule recheck:
  schedule is one prefill `bs4` batch followed by 127 decode `bs4` batches;
  graph replay count by actual batch size is `{"4": 254}` across warmup +
  repeat; padded graph size is `{"4": 254}`; eager decode fallback count is 0.
- 4096/128/bs4 small-KV nsys recheck with `--num-pages 72` and
  `--max-extend-tokens 16384`: schedule is one prefill `bs4` batch followed by
  127 decode `bs4` batches; graph replay count by actual batch size is
  `{"4": 254}` across warmup + repeat; eager decode fallback count is 0.
- 4096/1024/bs4 `v1_moe_graph`: captured `[4,2,1]`, replay count 2046.
- 4096/128/bs4 `v1_moe_graph_sample`: captured `[4,2,1]`; schedule is one
  prefill `bs4` batch followed by 127 decode `bs4` batches; greedy sampler
  replay count 254; eager decode fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_sample`: captured `[4,2,1]`; schedule is one
  prefill `bs4` batch followed by 1023 decode `bs4` batches; greedy sampler
  replay count 2046; eager decode fallback count 0.
- 4096/128/bs4 `v1_moe_graph` with `copy_done` event pooling: TP8 smoke
  captured `[4,2,1]`; macro replay count is 254 with actual batch size 4,
  padded graph size 4, and eager decode fallback count 0.
- TP8 text smoke after the masked compressed/indexer store fix: captured
  `[4,2,1]`, replay count 3, and eager decode fallback count 0.
- 4096/128/bs4 after the masked compressed/indexer store fix: captured
  `[4,2,1]`, replay count 254, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- 4096/1024/bs4 after the masked compressed/indexer store fix: captured
  `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- TP8 text smoke after fused masked-loc staging: captured `[4,2,1]`, replay
  count 3, and eager decode fallback count 0.
- 4096/128/bs4 after fused masked-loc staging: captured `[4,2,1]`, replay
  count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- TP8 text smoke after nested enqueue NVTX ranges: captured `[4,2,1]`,
  replay count 3, and eager decode fallback count 0.
- TP8 text smoke after tail-only replay metadata fills: captured `[4,2,1]`,
  replay count 3, and eager decode fallback count 0.
- 4096/128/bs4 after tail-only replay metadata fills: captured `[4,2,1]`,
  replay count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- TP8 text smoke after binding capture metadata inputs: captured `[4,2,1]`,
  replay count 3, and eager decode fallback count 0.
- 4096/128/bs4 after binding capture metadata inputs: captured `[4,2,1]`,
  replay count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- TP8 text smoke `v1_moe_graph_hc`: captured `[4,2,1]`, replay count 3,
  eager decode fallback count 0. Warning status is only the usual
  `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc`: captured `[4,2,1]`, replay count 254,
  actual batch size 4, padded graph size 4, and eager decode fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_hc`: captured `[4,2,1]`, replay count 2046,
  actual batch size 4, padded graph size 4, and eager decode fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm`: captured `[4,2,1]`, replay count
  3, eager decode fallback count 0. Warning status is only the usual
  `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm`: captured `[4,2,1]`, replay count
  254, actual batch size 4, padded graph size 4, and eager decode fallback
  count 0.
- 4096/1024/bs4 `v1_moe_graph_hc_rmsnorm`: captured `[4,2,1]`, replay count
  2046, actual batch size 4, padded graph size 4, and eager decode fallback
  count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_fp8gemm`: captured `[4,2,1]`, replay
  count 3, eager decode fallback count 0. Warning status is only the usual
  `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_fp8gemm`: captured `[4,2,1]`, replay
  count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm`: captured `[4,2,1]`,
  replay count 3, eager decode fallback count 0. Warning status is only the
  usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm`: captured `[4,2,1]`,
  replay count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm`: captured `[4,2,1]`,
  replay count 2046, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_woa`: captured `[4,2,1]`,
  replay count 3, eager decode fallback count 0. Warning status is only the
  usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_woa`: captured `[4,2,1]`,
  replay count 254, actual batch size 4, padded graph size 4, and eager decode
  fallback count 0. Macro throughput regressed versus
  `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm`.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm`: captured
  `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning status is
  only the usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm`: captured
  `[4,2,1]`, replay count 254, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm`: captured
  `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm`: captured
  `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning status is
  only the usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm`: captured
  `[4,2,1]`, replay count 254, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm`: captured
  `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning
  status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, and eager decode fallback count 0. Macro throughput regressed versus
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm`.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache`: captured
  `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning status is
  only the usual `max_tokens=4` truncation.
- 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache`: captured
  `[4,2,1]`, replay count 254, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- 4096/1024/bs4 `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache`: captured
  `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- TP8 text smoke `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning
  status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache`: captured
  `[4,2,1]`, replay count 254, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- 4096/1024/bs4
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache`: captured
  `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size 4, and
  eager decode fallback count 0.
- TP8 text smoke
  `v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning
  status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, and eager decode fallback count 0. Macro throughput regressed versus
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache`.
- TP8 text smoke
  `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning
  status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- 4096/1024/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- TP8 text smoke
  `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. Warning
  status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- 4096/1024/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- TP8 text smoke
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, eager decode fallback count 0. A first
  attempt failed capture because the merged projection's KV split was a
  non-contiguous view, causing q/KV norm-rope-store to fall back into
  capture-unsafe `k_norm_rope_cache_fallback`; making the KV branch contiguous
  fixed capture. Warning status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- 4096/1024/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 2046, actual batch size 4, padded graph size
  4, and eager decode fallback count 0.
- TP8 text smoke after teaching q/KV norm-rope-store to accept strided KV split
  views: captured `[4,2,1]`, replay count 3, and eager decode fallback count 0.
  Warning status is only the usual `max_tokens=4` truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, actual batch size 4, padded graph size
  4, eager decode fallback count 0, and communication unchanged at 704
  collectives / 139,602,984,960 bytes.
- TP8 text smoke
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 3, greedy sample replay count 3, and eager
  decode fallback count 0. Warning status is only the usual `max_tokens=4`
  truncation.
- 4096/128/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 254, greedy sample replay count 254, actual
  batch size 4, padded graph size 4, eager decode fallback count 0, and
  communication unchanged at 704 collectives / 139,602,984,960 bytes.
- 4096/1024/bs4
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`:
  captured `[4,2,1]`, replay count 2046, greedy sample replay count 2046,
  actual batch size 4, padded graph size 4, eager decode fallback count 0, and
  communication unchanged at 704 collectives / 139,602,984,960 bytes.

Important exactness note:

- Before the masked-store fix, graph capture used dummy decode positions that
  did not necessarily hit C4/C128 compression boundaries. That meant Python
  branches around compressed/indexer KV stores could be skipped during capture,
  and replay could not add those stores later.
- Capture metadata now owns fixed C4/C128 loc tensors. Replay writes
  `raw_out_loc // ratio` for boundary rows and `-1` otherwise. The Triton store
  kernels already mask `loc < 0`, so the store path is present in the graph with
  stable shapes.
- The fix costs a small amount of throughput but gives the graph variant a much
  better exactness story.

Greedy sampler capture by itself is correctness-clean but not materially faster
on the early `v1_moe_graph` path. Layered onto the current best graph-body path,
it becomes a small graph-surface cleanup: ordinary macro throughput is slightly
positive and Nsight confirms the graph-outside greedy `argmax` launch is gone.
This is still not a major gap-closure lever compared with larger
communication/model kernels.

The Python communication counters undercount graph replay because replayed
collectives no longer call through the Python `DistributedCommunicator` wrapper
on each token. Use Nsight for replay kernel and CUDA graph event counts.

## Nsight Status

The 4096/128/bs4 pure `v1_moe_graph` rank0 nsys run completed. The first run
did not have a matching benchmark NVTX repeat range, so its `nvtx_window`
section was effectively total process scope. Benchmark NVTX markers and the
sqlite summarizer were updated so missing windows are explicit, then the profile
was rerun with `repeat:decode_throughput_bs8:0` found.

Workload throughput under Nsight was lower due to profiler overhead:

| Profile | Window | Kernels | Graph trace | Runtime calls | NCCL kernels | CUDA graph events | Graph launches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mini eager default prefill, 07.1 | total | 6,663,421 | 0 | 7,389,232 | 22,528 | 0 | 0 |
| mini `v1_moe_graph`, 07.2 small-KV true batch4 + batch NVTX | repeat:decode_throughput_bs8:0 | 73,485 | 127 / 21.1886s | 118,140 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 event pool true batch4 | repeat:decode_throughput_bs8:0 | 73,485 | 127 / 21.1805s | 117,884 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 masked compressed store true batch4 | repeat:decode_throughput_bs8:0 | 74,799 | 127 / 21.4271s | 119,335 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 fused masked-loc true batch4 | repeat:decode_throughput_bs8:0 | 73,402 | 127 / 21.4319s | 117,687 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 enqueue NVTX true batch4 | repeat:decode_throughput_bs8:0 | 73,402 | 127 / 21.4308s | 117,669 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 tail-only metadata fills true batch4 | repeat:decode_throughput_bs8:0 | 72,513 | 127 / 21.4321s | 116,804 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 bound metadata inputs true batch4 | repeat:decode_throughput_bs8:0 | 72,513 | 127 / 21.4362s | 116,553 | 88 | 12 total | 127 |
| mini `v1_moe_graph`, 07.2 node trace true batch4 | repeat:decode_throughput_bs8:0 | 3,405,501 | n/a, graph nodes in kernel rows | 116,538 | 11,264 | n/a | 127 |
| mini `v1_moe_graph`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 3,405,501 | n/a, graph nodes in kernel rows | 116,529 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,908,413 | n/a, graph nodes in kernel rows | 104,839 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,747,734 | n/a, graph nodes in kernel rows | 103,690 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,720,429 | n/a, graph nodes in kernel rows | 103,672 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,707,094 | n/a, graph nodes in kernel rows | 103,669 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,701,590 | n/a, graph nodes in kernel rows | 103,617 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,698,902 | n/a, graph nodes in kernel rows | 103,603 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,616,342 | n/a, graph nodes in kernel rows | 102,966 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,610,838 | n/a, graph nodes in kernel rows | 102,923 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,544,833 | n/a, graph nodes in kernel rows | 102,447 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,539,329 | n/a, graph nodes in kernel rows | 102,395 | 11,264 | n/a | 127 |
| mini `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`, 07.2 capture-NVTX node trace true batch4 | repeat:decode_throughput_bs8:0 | 1,539,329 | n/a, graph nodes in kernel rows | 102,150 | 11,264 | n/a | 127 |
| mini `v1_moe_graph`, 07.2 `num_pages=64` constrained | repeat:decode_throughput_bs8:0 | 166,092 | 254 / 39.2753s | 226,986 | 352 | 12 total | 254 |
| mini `v1_moe_graph`, 07.2 `num_pages=64` total | total | 450,913 | 508 / 78.5676s | 916,709 | 968 | 12 | 508 |
| fair vLLM, 07.1 | total | 124,480 | n/a | 1,908,662 | 16 | 7,200 | n/a |

Batch-level NVTX split for the corrected true-batch4 graph profile:

| Batch-forward range | Ranges | Kernels | Graph trace | NCCL kernels | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:prefill:bs4:padded4` | 1 | 26,982 / 21.2073s | 0 | 88 / 0.1502s | 0 |
| `batch_forward:decode:bs4:padded4` | 127 | 1,778 / 0.0090s | 127 / 21.1886s | 0 | 127 |

Batch-level NVTX split after `copy_done` event pooling:

| Batch-forward range | Ranges | Kernels | Graph trace | Runtime calls | NCCL kernels | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:prefill:bs4:padded4` | 1 | 26,982 / 21.2044s | 0 | 29,673 / 20.8495s | 88 / 0.1519s | 0 |
| `batch_forward:decode:bs4:padded4` | 127 | 1,778 / 0.0090s | 127 / 21.1805s | 5,215 / 21.1597s | 0 | 127 |

Batch-level NVTX split after masked compressed/indexer store capture fix:

| Batch-forward range | Ranges | Kernels | Graph trace | Runtime calls | NCCL kernels | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:prefill:bs4:padded4` | 1 | 26,772 / 21.2042s | 0 | 29,346 / 20.8580s | 88 / 0.1499s | 0 |
| `batch_forward:decode:bs4:padded4` | 127 | 3,302 / 0.0121s | 127 / 21.4271s | 6,993 / 21.4162s | 0 | 127 |

Batch-level NVTX split after fused masked-loc replay staging:

| Batch-forward range | Ranges | Kernels | Graph trace | Runtime calls | NCCL kernels | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:prefill:bs4:padded4` | 1 | 26,772 / 21.2086s | 0 | 29,349 / 20.8293s | 88 / 0.1549s | 0 |
| `batch_forward:decode:bs4:padded4` | 127 | 1,905 / 0.0092s | 127 / 21.4319s | 5,342 / 21.4075s | 0 | 127 |

Forward-vs-enqueue split after adding nested enqueue NVTX ranges:

| NVTX range | Ranges | Range duration | Kernels | Graph trace | Runtime calls | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:decode:bs4:padded4` | 127 | 21.5720s | 1,905 / 0.0092s | 127 / 21.4308s | 5,342 / 21.3982s | 127 |
| `batch_forward_enqueue:decode:bs4:padded4` | 127 | 1.1001s | 1,524 / 0.0031s | 0 | 5,215 / 0.9360s | 127 |

Forward-vs-enqueue split after tail-only replay metadata fills:

| NVTX range | Ranges | Range duration | Kernels | Graph trace | Runtime calls | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:decode:bs4:padded4` | 127 | 21.5526s | 1,016 / 0.0077s | 127 / 21.4321s | 4,453 / 21.4099s | 127 |
| `batch_forward_enqueue:decode:bs4:padded4` | 127 | 0.9630s | 635 / 0.0017s | 0 | 4,318 / 0.8283s | 127 |

Forward-vs-enqueue split after binding DSV4 capture metadata inputs:

| NVTX range | Ranges | Range duration | Kernels | Graph trace | Runtime calls | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:decode:bs4:padded4` | 127 | 21.5414s | 1,016 / 0.0077s | 127 / 21.4362s | 4,199 / 21.4054s | 127 |
| `batch_forward_enqueue:decode:bs4:padded4` | 127 | 0.9105s | 635 / 0.0017s | 0 | 4,064 / 0.7817s | 127 |

Node-level graph trace split after binding DSV4 capture metadata inputs
(`NSYS_CUDA_GRAPH_TRACE=node`; use for graph-body attribution, not strict
kernel-count comparison against normal graph trace):

| NVTX range | Ranges | Range duration | Kernels | graphNodeId kernels | NCCL kernels | cudaGraphLaunch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_forward:decode:bs4:padded4` | 127 | 22.0737s | 3,334,004 / 21.6497s | 3,332,988 / 21.6419s | 11,176 / 0.3200s | 127 |
| `batch_forward_enqueue:decode:bs4:padded4` | 127 | 6.3363s | 956,228 / 6.0687s | 955,593 / 6.0671s | 3,193 / 0.0934s | 127 |

Top graph-node kernels in `batch_forward:decode:bs4:padded4`:

| Kernel | Count | Duration s | Share of graph-node duration |
| --- | ---: | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5,461 | 5.2688 | 24.3% |
| `_grouped_fp4_linear_kernel` | 5,461 | 3.2719 | 15.1% |
| PyTorch direct-copy unrolled kernel | 270,383 | 2.5001 | 11.6% |
| `sparse_attention_kernel` | 5,207 | 1.9715 | 9.1% |
| PyTorch direct-copy elementwise kernel | 220,599 | 0.8509 | 3.9% |
| PyTorch fp32 multiply vectorized kernel | 48,895 | 0.8402 | 3.9% |
| PyTorch fp32 divide elementwise kernel | 472,313 | 0.8212 | 3.8% |
| PyTorch fp32 reduce kernel | 223,774 | 0.7301 | 3.4% |

Capture-NVTX node trace attribution
(`MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1`, `NSYS_CUDA_GRAPH_TRACE=node`,
profiling-only):

| Shape | Output tok/s | Decode tok/s | Replay by bs | Eager fallback | Capture map | Decode graph-node kernels |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 4096/16/bs4 | 2.5418 | 22.8321 | `{"4": 30}` | 0 | 26,244 | 393,660 / 2.5551s |
| 4096/128/bs4 | 11.3260 | 22.6781 | `{"4": 254}` | 0 | 26,244 | 3,332,988 / 21.6521s |
| 4096/128/bs4 + HC helpers | 12.2831 | 26.3491 | `{"4": 254}` | 0 | 14,548 | 1,847,596 / 18.7020s |
| 4096/128/bs4 + HC + RMSNorm | 12.4769 | 26.9576 | `{"4": 254}` | 0 | 13,508 | 1,715,516 / 18.3631s |
| 4096/128/bs4 + HC + RMSNorm + `wq_b` FP8 GEMM | 12.5118 | 27.0312 | `{"4": 254}` | 0 | 13,293 | 1,688,211 / 18.2992s |
| 4096/128/bs4 + HC + RMSNorm + `wq_b`/`wo_b` FP8 GEMM | 12.5307 | 27.1476 | `{"4": 254}` | 0 | 13,078 | 1,660,906 / 18.2420s |
| 4096/128/bs4 + HC + RMSNorm + `wq_b`/`wo_b`/indexer `wq_b` FP8 GEMM | 12.5289 | 27.2534 | `{"4": 254}` | 0 | 12,973 | 1,647,571 / 18.1173s |
| 4096/128/bs4 + HC + RMSNorm + `wq_b`/`wo_b`/indexer `wq_b` FP8 GEMM + gate cache | 12.5846 | 27.3722 | `{"4": 254}` | 0 | 12,930 | 1,642,110 / 18.0892s |
| 4096/128/bs4 + cached fused `wq_a/wkv` + q/KV norm-rope-store | 12.8579 | 28.6313 | `{"4": 254}` | 0 | 11,705 | 1,486,535 / 17.2133s |
| 4096/128/bs4 + cached fused `wq_a/wkv` + q/KV norm-rope-store + strided KV split | 12.8733 | 28.7597 | `{"4": 254}` | 0 | 11,662 | 1,481,074 / 17.1972s |
| 4096/128/bs4 + cached fused `wq_a/wkv` + q/KV norm-rope-store + greedy sample replay | 12.8813 | 28.7733 | `{"4": 254}` | 0 | 11,662 | 1,481,328 / 17.2060s |

Top layer-collapsed capture ranges for
`batch_forward:decode:bs4:padded4`:

| Capture range group | 4096/16 duration s | 4096/16 share | 4096/128 duration s | 4096/128 share |
| --- | ---: | ---: | ---: | ---: |
| `dsv4.layer*.mlp.routed` | 1.1099 | 43.4% | 9.4090 | 43.5% |
| `dsv4.layer*.attn.backend` | 0.2438 | 9.5% | 2.0634 | 9.5% |
| `dsv4.layer*.hc_attn_pre` | 0.1953 | 7.6% | 1.6544 | 7.6% |
| `dsv4.layer*.hc_ffn_pre` | 0.1947 | 7.6% | 1.6494 | 7.6% |
| `dsv4.layer*.attn.q_proj` | 0.1697 | 6.6% | 1.4370 | 6.6% |
| `dsv4.layer*.attn.indexer` | 0.1115 | 4.4% | 0.9442 | 4.4% |
| `dsv4.layer*.mlp.shared` | 0.1044 | 4.1% | 0.8837 | 4.1% |
| `dsv4.layer*.attn.wo_b` | 0.0951 | 3.7% | 0.8049 | 3.7% |

HC helper attribution delta in the full 4096/128 node trace:

| Capture range group | Before count | Before duration s | HC count | HC duration s |
| --- | ---: | ---: | ---: | ---: |
| `dsv4.layer*.hc_attn_pre` | 780,923 | 1.6544 | 54,610 | 0.2309 |
| `dsv4.layer*.hc_ffn_pre` | 780,923 | 1.6494 | 54,610 | 0.2320 |
| `dsv4.layer*.hc_attn_post` | n/a | n/a | 5,461 | 0.0198 |
| `dsv4.layer*.hc_ffn_post` | n/a | n/a | 5,461 | 0.0176 |

Prepare-sync attribution profile
(`MINISGL_BENCH_SYNC_PREPARE_NVTX=1`, profiling-only, true batch4):

| NVTX range | Ranges | Kernels | Runtime calls | Memcpy | Graph trace | NCCL kernels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch_prepare:prefill:bs4` | 1 | 32,394 / 0.0565s | 64,713 / 0.4316s | 32,267 / 26.8 MB | 0 | 0 |
| `batch_prepare:decode:bs4` | 127 | 12,093 / 0.0292s | 17,376 / 0.1228s | 2,576 / 1.15 MB | 0 | 0 |
| `batch_forward:prefill:bs4:padded4` | 1 | 26,982 / 21.2072s | 29,671 / 20.8355s | 191 / 178.4 MB | 0 | 88 |
| `batch_forward:decode:bs4:padded4` | 127 | 1,778 / 0.0090s | 5,342 / 21.1614s | 2,794 / 3.8 MB | 127 / 21.1876s | 0 |

Interpretation:

- In the measured true-batch4 repeat window, guarded graph cuts rank0 kernel
  count by about 90.7x and NCCL kernels by about 256.0x versus the previous mini
  eager total profile. Because the eager run lacks the corrected repeat NVTX
  window, this ratio should be treated as directional rather than strict
  apples-to-apples.
- The remaining top kernels are still DSV4 MoE and attention work:
  `_grouped_fp4_w13_kernel`, `_grouped_fp4_linear_kernel`,
  `sparse_attention_kernel`, and `_indexer_bf16_logits_kernel`.
- The 88 NCCL kernels in the corrected true-batch4 repeat window are all in the
  eager prefill batch. Decode replay is visible as 127 graph trace intervals and
  127 `cudaGraphLaunch_v10000` calls; this export does not expand captured graph
  work into per-op NCCL kernel rows.
- The profiling-only prepare-sync run shows the graph-outside decode metadata
  surface clearly: 127 decode prepare ranges launch 12,093 tiny kernels, but
  their aggregate GPU kernel duration is only 0.029s and their synchronized
  range duration is 0.421s. This is real launch noise, but not the dominant
  wall-time issue compared with the 21.19s decode graph trace.
- Reusing `copy_done` events removes the per-decode
  `cudaEventCreateWithFlags_v3020` calls from `batch_forward:decode:bs4:padded4`
  (127 -> 0). The remaining event work is recording the reused event once per
  batch. Macro throughput stayed within noise, so this is a safe graph-surface
  cleanup rather than a major performance lever.
- The masked compressed/indexer store fix increases visible decode-forward
  graph-adjacent kernels from 1,778 to 3,302 in the repeat window. The new work
  is mostly tiny `%`, `div`, `where`, and `fill` kernels used to update the
  fixed C4/C128 loc buffers before replay. Decode graph trace rises from about
  21.18s to 21.43s under Nsight, and macro decode throughput drops from about
  23.87 to 23.55 tok/s on 4096/128. This is an exactness tradeoff, not a speed
  optimization.
- The fused masked-loc staging kernel removes most of that graph-adjacent
  metadata launch noise: `batch_forward:decode` drops from 3,302 to 1,905
  kernels and runtime calls from 6,993 to 5,342. Macro throughput stays within
  noise at 11.6911 output tok/s and 23.5639 decode tok/s, so this is useful
  cleanup but not a material speed path.
- The nested enqueue NVTX run shows the synchronized `batch_forward:decode`
  duration is mostly benchmark attribution. The inner
  `batch_forward_enqueue:decode` ranges cover only 1.1001s total, with 0.8842s
  in `cudaGraphLaunch_v10000`; the remaining about 20.46s in the outer range is
  the benchmark's per-forward `cudaDeviceSynchronize`, which waits for graph
  body work.
- Tail-only replay metadata fills remove redundant fill launches when source
  metadata already covers the capture-buffer width: decode enqueue kernels drop
  from 1,524 to 635, `cudaLaunchKernel` calls from 1,778 to 889, and top
  `FillFunctor<int>` kernels from 1,016 to 127. Macro throughput remains within
  noise at 11.7061 output tok/s and 23.6057 decode tok/s.
- Binding DSV4 capture metadata `raw_out_loc` and `positions` to the graph input
  buffers removes two duplicate tiny copies per decode replay. Decode enqueue
  `cudaMemcpyAsync` runtime calls drop by 254 in the measured repeat, the
  CUPTI memcpy row count drops from 2,667 to 2,413, and total repeat runtime
  calls drop from 116,804 to 116,553. Macro throughput remains within noise at
  11.6963 output tok/s and 23.5939 decode tok/s.
- The opt-in sm80 HC helpers are the first graph-body cleanup in this phase with
  material macro impact. They fuse HC split/Sinkhorn/y mixing and HC post
  mixing into Triton kernels under `MINISGL_DSV4_SM80_HC=1`. On 4096/128/bs4,
  macro throughput improves from the bound-metadata exact graph result
  11.6963 / 23.5939 tok/s to 12.6437 / 27.1310 tok/s. On 4096/1024/bs4 it
  improves from the masked-store exact graph result 20.7570 / 23.5922 tok/s to
  23.5562 / 27.1983 tok/s. Communication count and bytes are unchanged.
- Normal Nsight graph trace exports graph replay as graph intervals:
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` has graph ids and timing, while
  `CUPTI_ACTIVITY_KIND_KERNEL.graphNodeId` is null. Node-level graph trace flips
  this view: `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` and `CUDA_GRAPH_EVENTS` are not
  present, but replayed graph-body kernels appear as kernel rows with non-null
  `graphNodeId`.
- The full 4096/128/bs4 node trace shows the decode graph body is still
  dominated by model kernels, not graph submission itself. In the synchronized
  `batch_forward:decode` range, 3,332,988 of 3,334,004 kernels are graph-node
  kernels and account for 21.6419s of 21.6497s GPU kernel time. The largest
  contributors are grouped FP4 MoE (`_grouped_fp4_w13_kernel` and
  `_grouped_fp4_linear_kernel`, 39.5% combined), PyTorch copy/elementwise/reduce
  kernels, and sparse attention.
- The capture-NVTX probe maps replayed graph nodes back to model regions. The
  layer-collapsed view confirms that the largest graph-body region is routed
  MoE (`dsv4.layer*.mlp.routed`). It also makes the PyTorch small-kernel surface
  visible at the module level: HC pre-processing, q projection, indexer,
  shared experts, and `wo_b` all contribute many graph nodes even when their
  individual kernels are small.
- With the HC helpers enabled, full 4096/128 node trace `batch_forward:decode`
  graph-node kernels drop from 3,332,988 / 21.6521s to 1,847,596 / 18.7020s.
  The two HC pre groups drop from 1,561,846 kernels / 3.3038s combined to
  109,220 kernels / 0.4629s combined. Routed MoE remains effectively unchanged
  at about 9.405s and is again the dominant graph-body region.
- With the opt-in RMSNorm helper layered on top of HC, full 4096/128 node trace
  `batch_forward:decode` graph-node kernels drop again to 1,715,516 / 18.3631s.
  The repeat-window kernel count drops from 1,908,413 to 1,775,039. The biggest
  attributed reduction is `dsv4.layer*.attn.q_proj`, 294,894 -> 251,206
  graph-node kernels and 1.4348s -> 1.3298s. Lower-ranked norm ranges show the
  intended shape: `dsv4.layer*.attn_input_norm` and
  `dsv4.layer*.mlp_input_norm` each drop from 49,149 -> 5,461 graph-node
  kernels, and `dsv4.model.final_norm` drops from 1,143 -> 127. NCCL kernel
  count is unchanged at 11,176 in `batch_forward:decode`.
- The existing sm80 FP8 GEMM path is correctness-clean but not a good global
  graph-body lever in this configuration. A decode-like `m=4` microbench shows
  mixed results: `fp8_wq_b` improves 0.831ms -> 0.487ms, but `fp8_wq_a`
  regresses 0.362ms -> 0.414ms and `fp8_wkv` regresses 0.348ms -> 0.414ms.
  The full 4096/128/bs4 macro confirms the negative aggregate result:
  `v1_moe_graph_hc_rmsnorm_fp8gemm` reaches only 11.8707 output tok/s and
  23.6431 decode tok/s versus 12.7926 and 27.6019 for
  `v1_moe_graph_hc_rmsnorm`. The 4096/1024 macro was intentionally not run
  after this short-shape regression.
- Selectively applying the existing FP8 GEMM path only to attention `wq_b` avoids
  that global regression and gives a small positive result. On ordinary macro
  runs, `v1_moe_graph_hc_rmsnorm_wqb_fp8gemm` improves 4096/128 from 12.7926 /
  27.6019 to 12.8048 / 27.6743 tok/s, and 4096/1024 from 23.9128 / 27.6512 to
  23.9902 / 27.7537 tok/s. The node trace attributes the change to
  `dsv4.layer*.attn.q_proj`: graph-node kernels drop 251,206 -> 223,901 and
  attributed duration drops 1.3298s -> 1.2716s. Decode `batch_forward` graph-node
  kernels drop 1,715,516 -> 1,688,211, while NCCL remains unchanged at 11,176
  kernels in the decode forward range.
- Selectively adding the same FP8 GEMM path to attention `wo_b` is also
  positive, but only incrementally. A decode-like `fp8_wo_b` microbench improves
  0.338ms -> 0.297ms. On ordinary macro runs,
  `v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm` improves 4096/128 from 12.8048 /
  27.6743 to 12.8352 / 27.7860 tok/s, and 4096/1024 from 23.9902 / 27.7537 to
  24.0576 / 27.8467 tok/s. The node trace attributes the change to
  `dsv4.layer*.attn.wo_b`: graph-node kernels drop 125,603 -> 98,298 and
  attributed duration drops 0.8160s -> 0.7542s. Decode `batch_forward`
  graph-node kernels drop 1,688,211 -> 1,660,906, while NCCL remains unchanged
  at 11,176 kernels in the decode forward range.
- Selectively applying the same FP8 GEMM path to the indexer `wq_b` projection
  is another small positive graph-body cleanup. A decode-like
  `fp8_indexer_wq_b` microbench improves 0.349ms -> 0.309ms. On ordinary macro
  runs, `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm` improves 4096/128
  from 12.8352 / 27.7860 to 12.8670 / 27.9587 tok/s, and 4096/1024 from
  24.0576 / 27.8467 to 24.1831 / 28.0048 tok/s. The node trace attributes the
  change to `dsv4.layer*.attn.indexer`: graph-node kernels drop 221,361 ->
  208,026 and attributed duration drops 0.9444s -> 0.8306s. Decode
  `batch_forward` graph-node kernels drop 1,660,906 -> 1,647,571, while NCCL
  remains unchanged at 11,176 kernels in the decode forward range.
- Exact MoE gate fp32 weight caching is a small positive graph-body cleanup. It
  does not use the bf16 tensor-core precision-lane experiment; it caches
  `gate.weight.float().contiguous()` behind
  `MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE=1`, invalidated by weight
  data/version/shape/stride, so the runtime gate remains
  `hidden.float() @ cached_weight_fp32`. On ordinary macro runs,
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache` improves 4096/128 from
  12.8670 / 27.9587 to 12.8804 / 28.0294 tok/s, and 4096/1024 from 24.1831 /
  28.0048 to 24.2265 / 28.0659 tok/s. The node trace attributes the change to
  `dsv4.layer*.mlp.gate`: graph-node kernels drop 76,073 -> 70,612 and
  attributed duration drops 0.3210s -> 0.2892s. Decode `batch_forward`
  graph-node kernels drop 1,647,571 -> 1,642,110, while NCCL remains unchanged
  at 11,176 kernels in the decode forward range. The cache costs about 180 MB
  peak allocated memory per rank in the measured TP8 run.
- Exact indexer-store norm fp32 weight caching is another small graph-body
  cleanup. It caches `indexer.compressor.norm.weight.float().contiguous()`
  behind `MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE=1`, with the
  same data/version/shape/stride invalidation as gatecache. The runtime
  indexer-store fallback still computes `flat.float()` RMSNorm and multiplies
  by fp32 values from the original bf16 weight. On ordinary macro runs,
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache` improves
  4096/128 from 12.8804 / 28.0294 to 12.8950 / 28.0521 tok/s, and 4096/1024
  from 24.2265 / 28.0659 to 24.2286 / 28.0716 tok/s. The node trace attributes
  the change to `dsv4.layer*.attn.indexer_store`: graph-node kernels drop
  144,018 -> 141,351 and attributed duration drops 0.2362s -> 0.2317s. Decode
  `batch_forward` graph-node kernels drop 1,642,110 -> 1,639,443, while NCCL
  remains unchanged at 11,176 kernels in the decode forward range.
- Selective attention `wq_a` FP8 GEMM is not a useful target-shape graph-body
  path. The decode-like `fp8_wq_a` microbench regresses 0.336ms -> 0.396ms
  despite exact output, and the full 4096/128/bs4 macro with
  `v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache`
  regresses to 12.5642 / 26.5877 tok/s versus 12.8950 / 28.0521 for the
  previous best path. The path is correctness- and graph-replay-safe, but
  4096/1024 and nsys runs were intentionally skipped after this short-shape
  regression.
- The vLLM-aligned shared-activation `wq_a/wkv` path was a positive exact
  graph-body variant. In `/workspace/vllm-dsv4-docker`, DeepSeek V4 attention
  uses a `fused_wqa_wkv` projection boundary before splitting the low-rank q
  branch from the compressed KV branch. The mini variant does not yet implement
  that as one fused GEMM or a vLLM-style custom op boundary, but it does share
  the exact FP8 activation quantization result for `wq_a` and `wkv`. On ordinary
  macro runs,
  `v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache`
  improves 4096/128 from 12.8950 / 28.0521 to 12.9822 / 28.3548 tok/s, and
  4096/1024 from 24.2286 / 28.0716 to 24.5185 / 28.4276 tok/s. The node trace
  shows the expected attribution shape: `dsv4.layer*.attn.kv_proj` disappears
  as a separate capture group (125,603 graph-node kernels / 0.4526s removed),
  `dsv4.layer*.attn.q_proj` grows by 43,688 graph-node kernels / 0.2828s, and
  decode `batch_forward` graph-node kernels drop 1,639,443 -> 1,557,528. NCCL
  remains unchanged at 11,176 kernels in the decode forward range, and the
  macro communication aggregate remains 704 collectives / 139,602,984,960 bytes.
- Fusing q norm/rope with KV norm/rope/cache-store on top of `fwqakv` is the
  previous best exact graph-body variant, though the macro gain is tiny. This
  aligns with vLLM's fused qnorm/rope/KV insert boundary while preserving the
  existing exact KV bf16 path. On ordinary macro runs,
  `v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`
  improves 4096/128 from 12.9822 / 28.3548 to 12.9881 / 28.3571 tok/s, and
  4096/1024 from 24.5185 / 28.4276 to 24.5264 / 28.4586 tok/s. The node trace
  shows the intended shape: `dsv4.layer*.attn.q_norm_rope` and
  `dsv4.layer*.attn.kv_norm_rope_store` each had 5,461 graph-node kernels and
  are replaced by `dsv4.layer*.attn.q_kv_norm_rope_store` with 5,461 kernels.
  Decode `batch_forward` graph-node kernels drop 1,557,528 -> 1,552,067 and
  repeat-window kernels drop 1,616,342 -> 1,610,838. NCCL remains unchanged at
  11,176 kernels in the decode forward range, and the macro communication
  aggregate remains 704 collectives / 139,602,984,960 bytes.
- Caching a fused bf16 dequantized `wq_a/wkv` weight on top of `qkvrope` is a
  positive vLLM-aligned graph-body path. This is closer to vLLM's
  `fused_wqa_wkv` merged projection boundary than sharing activation quant
  alone. The first TP8 smoke attempt found a precise blocker: `torch.split` on
  the merged output produced a non-contiguous KV view, so the q/KV
  norm-rope-store Triton wrapper returned `False` and graph capture entered the
  old Python fallback with capture-unsafe `bool(torch.any(valid))`. The initial
  workaround made the KV branch contiguous and restored capture. On ordinary
  macro runs,
  `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache`
  improves 4096/128 from 12.9881 / 28.3571 to 13.1928 / 29.4214 tok/s, and
  4096/1024 from 24.5264 / 28.4586 to 25.3027 / 29.5032 tok/s. The node trace
  shows the intended target: `dsv4.layer*.attn.q_proj` drops from 174,752
  graph-node kernels / 0.9552s to 109,220 / 0.3142s. Decode `batch_forward`
  graph-node kernels drop 1,552,067 -> 1,486,535 and repeat-window kernels drop
  1,610,838 -> 1,544,833. NCCL remains unchanged at 11,176 kernels in the
  decode forward range, and the macro communication aggregate remains 704
  collectives / 139,602,984,960 bytes.
- Removing the forced KV contiguous copy from `fwqakvcache_qkvrope` is a
  smaller vLLM-aligned graph cleanup. The q/KV norm-rope-store wrapper now
  accepts the merged projection's KV split view as long as `kv.stride(-1) == 1`
  and passes `kv.stride(0)` into the Triton kernel. This follows the same
  stride-aware input contract as vLLM's `fused_q_kv_rmsnorm`, while mini keeps
  the q norm/rope plus KV norm/rope/cache-store work in one guarded Triton
  launch. On ordinary 4096/128 macro, output throughput is macro-neutral/slightly
  positive at 13.2070 / 29.4173 tok/s versus 13.1928 / 29.4214. In the full
  4096/128 node trace, repeat-window kernels drop 1,544,833 -> 1,539,329,
  runtime calls drop 102,447 -> 102,395, decode `batch_forward` graph-node
  kernels drop 1,486,535 -> 1,481,074, and decode enqueue graph-node kernels
  drop 326,301 -> 234,909. NCCL remains unchanged at 11,176 kernels in
  `batch_forward:decode` and 11,264 kernels in the repeat window.
- Capturing greedy sampling on the current best graph-body path is a small
  graph-surface cleanup, not a model-body speedup. It keeps the exact
  temperature-0 path and mirrors vLLM's V1 sampler shape, where greedy sampling
  is the `argmax(dim=-1)` path after logits. On ordinary macro runs it moves
  4096/128 from 13.2070 / 29.4173 to 13.2160 / 29.4655 tok/s and 4096/1024
  from 25.3027 / 29.5032 to 25.3076 / 29.5035 tok/s. Nsight confirms the
  intended effect: the graph-outside greedy `argmax` `reduce_kernel` count of
  127 disappears from outer decode, `batch_forward:decode` `cudaLaunchKernel`
  calls drop 635 -> 381, and runtime calls drop 4,072 -> 3,810. The graph body
  gains 254 replayed sampler nodes, so decode `batch_forward` graph-node
  kernels move 1,481,074 -> 1,481,328. Repeat-window runtime calls drop
  102,395 -> 102,150, while total kernels stay 1,539,329 and NCCL stays 11,264.
- Selective shared-expert FP8 GEMM is not a useful target-shape graph-body path
  despite positive isolated linear microbenches. `fp8_shared_gate_up` improves
  0.471ms -> 0.416ms and `fp8_shared_down` improves 0.329ms -> 0.290ms, both
  with exact output. However the full 4096/128/bs4 macro with
  `v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm` regresses to 12.5036
  / 26.2507 tok/s versus 12.8950 / 28.0521 for the previous best path. The
  4096/1024 macro and nsys runs were intentionally skipped after this
  short-shape regression.
- The existing `wo_a` helper is a negative target-shape graph experiment despite
  a positive isolated microbench. At `tokens=4, groups=8, d_per_group=512,
  rank=1024`, the helper improves 0.187ms -> 0.105ms with exact output, but the
  full 4096/128/bs4 `v1_moe_graph_hc_rmsnorm_wqb_woa` macro reaches only
  12.4909 / 26.2303 tok/s versus 12.8950 / 28.0521 for the then-best
  selective attention `wq_b`/`wo_b`, indexer `wq_b`, gate-cache, and
  indexer-store-cache path. The
  4096/1024 macro and nsys runs were intentionally skipped after this
  short-shape regression.
- The full 4096/128 capture-NVTX node trace matches the short 4096/16 probe
  almost exactly by percentage. This means the short probe is reliable for
  rapid graph-body attribution, while the full run confirms the target-shape
  prioritization: routed MoE first, attention backend second, then HC/q-proj/
  indexer/shared/wo_b PyTorch-heavy regions.
- The graph-node capture map is order-based: Nsight emits graph node creation
  records after `cudaStreamEndCapture`, so containment inside `dsv4.*` NVTX
  ranges does not work. The summarizer uses the capture window's kernel-launch
  runtime order, including `cudaLaunchKernel_v7000`, `cuLaunchKernel`, and
  `cuLaunchKernelEx`, then maps those labels to replayed kernel `graphNodeId`
  rows. This has been sanity-checked by verifying that top `_grouped_fp4_*`
  nodes map to `dsv4.layer*.mlp.routed`.
- Node trace also exposes replayed NCCL kernels inside the graph body:
  11,176 NCCL kernel rows in `batch_forward:decode`, totaling 0.3200s. This is
  a large count but a small share of graph-body duration, so further
  communication-count reduction would need fewer logical collectives or fused
  communication sites; swapping communicator plumbing alone is unlikely to close
  the remaining vLLM gap.
- The first full node-trace workload reached 11.3560 output tok/s and 22.7947
  decode tok/s under profiler. The `fwqakvcache_qkvrope` node-trace workload
  reached 12.8579 output tok/s and 28.6313 decode tok/s. The strided-KV
  follow-up reached 12.8733 output tok/s and 28.7597 decode tok/s, with
  `replay_count_by_batch_size={"4": 254}` and eager decode fallback count 0.
  The current-best-plus-sampler follow-up reached 12.8813 output tok/s and
  28.7733 decode tok/s, with greedy sample replay count 254 and eager decode
  fallback count 0. Treat node trace throughput as profiler-influenced; use the
  ordinary macro and normal graph-trace runs for performance tracking.
- The vLLM implementation in `/workspace/vllm-dsv4-docker` is still more
  structurally fused than mini in the attention boundary. It defines
  `attn.fused_wqa_wkv` as a merged `wq_a + wkv` projection, then splits the
  result and calls `fused_q_kv_rmsnorm`, followed by the
  `torch.ops.vllm.deepseek_v4_attention` custom op. It also has a fused
  qnorm/rope/KV-quant/cache-insert op and preallocated compressed-slot/C128A
  metadata buffers for graph address stability. The mini `fwqakv` path aligns
  the merged-projection idea by sharing the exact activation quantization across
  `wq_a` and `wkv`; `fwqakvcache` goes one step closer by caching a fused
  dequantized `wq_a + wkv` weight and using one projection before splitting.
  The `qkvrope` path then partially aligns the qnorm/rope/KV insert boundary by
  fusing q norm/rope with KV norm/rope/cache-store. The strided-KV follow-up
  removes mini's extra materialization after that split, matching the
  stride-aware handling of split q/KV inputs at the fused norm boundary. On the
  graph dispatcher side, vLLM's `CUDAGraphWrapper` assumes persistent runtime
  inputs are prepared outside the wrapper and dispatches by padded decode batch
  descriptor; mini's guarded graph runner now follows the same practical rule
  for this target by copying decode metadata/input ids/output locations into
  fixed capture buffers before replay. The greedy-sample variant additionally
  captures the exact temperature-0 `argmax` surface, matching vLLM's greedy
  sampler semantics at the boundary. These are measured incremental cleanups,
  not full vLLM parity.
- Compared with `/workspace/vllm-dsv4-docker`, the earlier fair vLLM reference
  remains much faster at macro level: 80.9050 output tok/s on 4096/128/bs4 and
  201.874 output tok/s on 4096/1024/bs4. The normal mini graph trace now has
  fewer visible repeat-window kernels than the 07.1 vLLM total profile, but
  that is not an apples-to-apples graph-body comparison because normal graph
  trace hides replayed node kernels. The node trace makes clear that mini still
  spends most decode time inside MoE/attention/PyTorch graph-body kernels.
- The large prefill prepare count, especially `arange_cuda_out`, belongs to the
  eager prefill path. Prefill remains intentionally outside decode graph in
  TARGET 07.2.
- NCCL fragmentation outside graph replay is much better, but total logical
  communication is not eliminated. Further reduction would need fewer logical
  collectives or fused communication sites, not just graph replay.
- Nsight reports `CUDA_GRAPH_EVENTS` for graph creation/instantiation only
  (`Graph Creation=9`, `GraphExec Creation=3`); replay is visible through
  `cudaGraphLaunch_v10000` runtime calls and
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`.
- The first corrected Nsight run intentionally used `--num-pages 64` to keep
  profiler memory low. That constrained KV pool cannot hold four full 4096+128
  requests, so its graph replay distribution was actual batch size 1: 254 and
  3: 254 across warmup + repeat.
- The follow-up `--num-pages 72 --max-extend-tokens 16384` Nsight run is the
  corrected small-KV true batch4 profile: actual batch size 4 for all decode
  replays, 127 graph launches in the measured repeat, and 88 NCCL kernels on
  profiled rank0. The Python communication aggregate for the same run is 704
  collectives, matching 8 TP ranks x 88 per-rank NCCL kernels.
- The batch-level NVTX rerun keeps the same true-batch4 schedule:
  4096/128/bs4 under Nsight reached 11.5406 output tok/s and 23.5691 decode
  tok/s, with `replay_count_by_batch_size={"4": 254}`,
  `replay_count_by_padded_size={"4": 254}`, and eager decode fallback count 0.
- The event-pool Nsight rerun also keeps the same true-batch4 schedule:
  4096/128/bs4 under Nsight reached 11.5597 output tok/s and 23.6292 decode
  tok/s, with `replay_count_by_batch_size={"4": 254}`,
  `replay_count_by_padded_size={"4": 254}`, and eager decode fallback count 0.
- The masked-store Nsight rerun keeps the same true-batch4 schedule:
  4096/128/bs4 under Nsight reached 11.4897 output tok/s and 23.3184 decode
  tok/s, with `replay_count_by_batch_size={"4": 254}`,
  `replay_count_by_padded_size={"4": 254}`, and eager decode fallback count 0.
- The fused masked-loc Nsight rerun keeps the same true-batch4 schedule:
  4096/128/bs4 under Nsight reached 11.4707 output tok/s and 23.3249 decode
  tok/s, with `replay_count_by_batch_size={"4": 254}`,
  `replay_count_by_padded_size={"4": 254}`, and eager decode fallback count 0.

The 4096/128/bs4 PyNCCL rank0 nsys attempt was interrupted after about
12 minutes. Observed state before interrupt:

- rank0, which was wrapped by `nsys`, showed 0% GPU utilization;
- ranks 1-7 showed 100% GPU utilization;
- only `run_config.json` was written;
- no `.nsys-rep` or sqlite summary was produced.

Treat this as a PyNCCL + rank0 Nsight blocker for now, not as a completed nsys
measurement.

## Next Recommendation

Continue TARGET 07.2 graph-readiness before entering MoE V2. The next focused
step is to inspect the remaining graph outside-surface around metadata/index
kernels and decide whether further communication fusion/reduction is plausible
before moving to TARGET 07.3 MoE V2. Greedy sampler capture is now tested on
the current best guarded variant and should not be treated as a remaining major
speed path. If communication cannot be reduced meaningfully beyond the current
graph replay behavior, the next major speedup should come from MoE V2.
