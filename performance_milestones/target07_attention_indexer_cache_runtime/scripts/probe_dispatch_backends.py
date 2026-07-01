#!/usr/bin/env python3
"""DeepSeek V4 sm80 attention/indexer/cache/runtime dispatch report.

This is intentionally source-driven.  The point of TARGET 07.393 is to avoid
guessing from one kernel name and instead spell out the backend boundary that
vLLM and mini-sglang actually dispatch on A100/sm80.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VLLM_ROOT = Path("/workspace/vllm-dsv4-docker")


def line_hit(root: Path, rel: str, needle: str) -> dict[str, Any]:
    path = root / rel
    out: dict[str, Any] = {
        "path": str(path),
        "relpath": rel,
        "needle": needle,
        "exists": path.exists(),
    }
    if not path.exists():
        return out
    try:
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if needle in line:
                out.update({"line": lineno, "text": line.strip()})
                return out
    except Exception as exc:  # pragma: no cover - diagnostic artifact.
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def source(root: Path, rel: str, needle: str) -> str:
    hit = line_hit(root, rel, needle)
    if hit.get("line") is None:
        return f"{hit['relpath']}:?: {needle}"
    return f"{hit['relpath']}:{hit['line']}: {hit.get('text', needle)}"


def mini_report() -> dict[str, Any]:
    root = REPO_ROOT
    return {
        "name": "mini-sglang current canonical exact path",
        "canonical_variant": (
            "v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        "precision_contract": {
            "activation_cache_default": "bf16 exact activation/cache path",
            "moe_weights": "MXFP4 weights with mini-owned Marlin WNA16 expert backend",
            "not_default": "Do not silently switch exact default to vLLM deepseek_v4_fp8.",
            "sources": [
                source(root, "python/minisgl/kvcache/deepseek_v4_pool.py", "storage_dtype: torch.dtype = torch.bfloat16"),
                source(root, "python/minisgl/kvcache/deepseek_v4_pool.py", "indexer_layout: Literal"),
                source(root, "python/minisgl/kernel/deepseek_v4.py", "DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16"),
            ],
        },
        "attention": {
            "frontend_qkv_cache": {
                "dispatch": "q_kv_norm_rope_cache_fallback -> Triton q_kv_norm_rope_cache_bf16 when MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE and sm80/triton are enabled; otherwise torch fallback pieces.",
                "cache_write_dtype": "bf16 SWA cache",
                "sources": [
                    source(root, "python/minisgl/models/deepseek_v4.py", "q_kv_norm_rope_cache_fallback("),
                    source(root, "python/minisgl/kernel/deepseek_v4.py", "def q_kv_norm_rope_cache_fallback"),
                    source(root, "python/minisgl/kernel/triton/deepseek_v4.py", "def q_kv_norm_rope_cache_bf16"),
                ],
            },
            "sparse_decode": {
                "dispatch": "DSV4AttentionBackend._sparse_attention_two_source calls mini JIT/CUDA dsv4_sparse_attention_two_source_bf16 when MINISGL_DSV4_SM80_SPARSE_ATTN_BF16 is enabled on sm80.",
                "input_boundary": "q bf16 [T, local_heads=8, 512], SWA bf16 cache, optional C4/C128 bf16 compressed cache, int32 flat indices/lens.",
                "not_same_as_vllm": "This kernel reads bf16 cache directly; vLLM sm80 first gathers/dequantizes packed fp8_ds_mla cache into a bf16 gathered buffer.",
                "sources": [
                    source(root, "python/minisgl/attention/deepseek_v4.py", "dsv4_sparse_attention_two_source_bf16("),
                    source(root, "python/minisgl/kernel/deepseek_v4.py", "def dsv4_sparse_attention_two_source_bf16"),
                    source(root, "python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu", "DeepSeek V4 sm80 two-source sparse attention"),
                ],
            },
        },
        "indexer": {
            "dispatch": "DSV4AttentionBackend.select_indexer calls indexer_select_bf16_fallback, which dispatches Triton indexer_bf16_logits + topk_transform_512 when enabled; otherwise torch logits/topk.",
            "cache_layout": "bf16 flat indexer cache, one 128-wide row per C4-compressed slot.",
            "query_boundary": "bf16 indexer q [T, 64, 128] + fp32/folded weights; topk width 512, ratio 4.",
            "sources": [
                source(root, "python/minisgl/attention/deepseek_v4.py", "select_indexer"),
                source(root, "python/minisgl/kernel/deepseek_v4.py", "def indexer_select_bf16_fallback"),
                source(root, "python/minisgl/kernel/triton/deepseek_v4.py", "def indexer_bf16_logits"),
                source(root, "python/minisgl/kernel/triton/deepseek_v4.py", "def topk_transform_512"),
            ],
        },
        "cache": {
            "layout": "DSV4CacheLayoutPolicy bf16_flat for SWA/C4/C128/indexer plus bf16 compressor state rings.",
            "swa": "_swa_buffer [layers, pages, page_size, 512] bf16; q/kv norm+RoPE store can run in graph.",
            "compressed": "C4/C128 component caches [layers, slots, 512] bf16; store path compress_norm_rope_store_fallback -> Triton compress_norm_rope_store_bf16/store_cache when enabled.",
            "indexer": "C4 indexer cache [layers, slots, 128] bf16; store path applies hadamard then bf16 store.",
            "sources": [
                source(root, "python/minisgl/kvcache/deepseek_v4_pool.py", "class DSV4CacheLayoutPolicy"),
                source(root, "python/minisgl/kvcache/deepseek_v4_pool.py", "self._swa_buffer"),
                source(root, "python/minisgl/kvcache/deepseek_v4_pool.py", "self._c4_indexer_buffer"),
                source(root, "python/minisgl/kernel/deepseek_v4.py", "def compress_norm_rope_store_fallback"),
            ],
        },
        "graph_runtime": {
            "graph_capture": "mini GraphRunner captures/replays selected batch sizes and binds input_ids/out_loc/positions to graph buffers.",
            "attention_metadata": "DSV4AttentionBackend prepares decode metadata, then copies page tables, sparse indices/lens, and compressed loc buffers before graph replay.",
            "known_profile_signal": "Post-Marlin rank0 4096/128 profile showed large runtime/copy/metadata event count beside GPU sparse attention/indexer kernels.",
            "sources": [
                source(root, "python/minisgl/engine/graph.py", "class GraphRunner"),
                source(root, "python/minisgl/engine/graph.py", "attn_backend.prepare_for_replay(batch_size)"),
                source(root, "python/minisgl/attention/deepseek_v4.py", "def _copy_metadata_for_replay"),
                source(root, "python/minisgl/attention/deepseek_v4.py", "def stage_capture_metadata_for_graph"),
            ],
        },
    }


def vllm_report(vllm_root: Path) -> dict[str, Any]:
    root = vllm_root
    return {
        "name": "vLLM DeepSeek V4 sm80 dispatch",
        "model_quantization": {
            "reported_name": "deepseek_v4_fp8",
            "sm80_moe": "Mxfp4MoEMethod/Marlin family for MoE, but non-MoE activations/cache use DeepSeek V4 FP8 policy.",
            "sources": [
                source(root, "vllm/model_executor/models/deepseek_v4.py", "class DeepseekV4FP8Config"),
                source(root, "vllm/model_executor/models/deepseek_v4.py", "return \"deepseek_v4_fp8\""),
                source(root, "vllm/model_executor/models/deepseek_v4.py", "Mxfp4MoEMethod"),
            ],
        },
        "attention": {
            "custom_op_boundary": {
                "dispatch": "DeepseekV4MultiHeadLatentAttentionWrapper.forward calls torch.ops.vllm.deepseek_v4_attention; the registered Python impl performs qnorm/RoPE/KV insert, indexer/compressor scheduling, and sparse MLA attention.",
                "sources": [
                    source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "torch.ops.vllm.deepseek_v4_attention"),
                    source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "direct_register_custom_op("),
                    source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "def attention_impl"),
                ],
            },
            "backend_class": {
                "name": "DeepseekV4FlashMLASparseBackend / V4_FLASHMLA_SPARSE",
                "sm80_reality": "Backend class is FlashMLA sparse, but FlashMLA sparse kernels only support compute capability 9/10; on sm80 use_dsv4_reference_kernels routes decode to reference gather/dequant + Triton split-K attention.",
                "sources": [
                    source(root, "vllm/v1/attention/backends/mla/flashmla_sparse.py", "class DeepseekV4FlashMLASparseBackend"),
                    source(root, "vllm/v1/attention/backends/mla/flashmla_sparse.py", "return \"V4_FLASHMLA_SPARSE\""),
                    source(root, "vllm/v1/attention/backends/mla/flashmla_sparse.py", "return [9, 10]"),
                    source(root, "vllm/utils/deep_gemm.py", "def use_dsv4_reference_kernels"),
                ],
            },
            "sm80_decode_pipeline": {
                "dispatch": "gather_dequant_two_scopes_with_mask one/two Triton launches over packed fp8_ds_mla cache -> _dsv4_sm80_sparse_attn_decode_triton split-K sparse attention core.",
                "sources": [
                    source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "def _dsv4_sm80_sparse_attn_decode_triton"),
                    source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "gather_dequant_two_scopes_with_mask("),
                    source(root, "vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py", "def gather_dequant_two_scopes_with_mask"),
                ],
            },
        },
        "indexer": {
            "backend": "DeepseekV4IndexerBackend / DEEPSEEK_V4_INDEXER",
            "cache": "FP8 indexer cache on sm80, uint8 rows with 128 fp8 values plus 4 bytes of scales (head dim 132); FP4 indexer cache is SM100-only.",
            "op_pipeline": "DeepseekV4Indexer.forward -> fused_indexer_q_rope_quant -> SparseAttnIndexer custom op; on sm80 SparseAttnIndexer uses fp8_paged_mqa_logits_triton and persistent_topk/top_k_per_row_decode.",
            "sources": [
                source(root, "vllm/v1/attention/backends/mla/indexer.py", "class DeepseekV4IndexerBackend"),
                source(root, "vllm/v1/attention/backends/mla/indexer.py", "return \"DEEPSEEK_V4_INDEXER\""),
                source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "Using FP8 indexer cache"),
                source(root, "vllm/model_executor/layers/sparse_attn_indexer.py", "fp8_paged_mqa_logits_triton"),
                source(root, "vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py", "def fused_indexer_q_rope_quant"),
            ],
        },
        "cache": {
            "kv_cache_dtype": "fp8_ds_mla uint8 packed cache; attention code canonicalizes auto/fp8 to fp8_ds_mla and asserts fp8.",
            "swa_cache": "DeepseekV4SWACache dtype uint8, internal block size 64, preferred kernel block size 256; each token is 584 bytes in DeepSeek V4 layout.",
            "compressed_cache": "MLAAttentionSpec dtype uint8, head_size 512, alignment 576, model_version deepseek_v4.",
            "cache_insert": "SWA insert uses fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert; compressor/cache paths use deepseek_v4_compressor_sparse_sm80/indexer_sm80 custom ops.",
            "sources": [
                source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "Using DeepSeek's fp8_ds_mla KV cache format"),
                source(root, "vllm/v1/attention/backends/mla/sparse_swa.py", "self.cache_dtype = dtype"),
                source(root, "vllm/v1/attention/backends/mla/sparse_swa.py", "return (num_blocks, self.block_size, 584)"),
                source(root, "vllm/model_executor/layers/deepseek_v4_attention.py", "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"),
                source(root, "vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py", "deepseek_v4_compressor_sparse_sm80"),
            ],
        },
        "graph_runtime": {
            "graph_capture": "vLLM V1 GPUModelRunner uses CudagraphDispatcher and parallel_state.graph_capture on a separate stream; graph sizes in 07.392 logs were [1,2,4].",
            "persistent_buffers": "Runner preallocates graph input/metadata buffers, sampling pinned CPU outputs, and backend-specific decode buffers.",
            "streams": "DeepseekV4Model creates AuxStreamType.Attention; maybe_execute_in_parallel overlaps C4 indexer work on current stream with KV insert/compressor work on aux stream.",
            "custom_all_reduce": "Custom all-reduce registers graph buffer addresses and switches behavior when graph capturing.",
            "sources": [
                source(root, "vllm/v1/worker/gpu_model_runner.py", "CudagraphDispatcher"),
                source(root, "vllm/v1/worker/gpu_model_runner.py", "with graph_capture(device=self.device)"),
                source(root, "vllm/distributed/parallel_state.py", "def graph_capture"),
                source(root, "vllm/model_executor/models/deepseek_v4.py", "AuxStreamType.Attention"),
                source(root, "vllm/utils/multi_stream_utils.py", "def maybe_execute_in_parallel"),
                source(root, "vllm/distributed/device_communicators/custom_all_reduce.py", "register_graph_buffers"),
            ],
        },
    }


def paired_alignment() -> list[dict[str, Any]]:
    return [
        {
            "boundary": "attention_front_qkv_cache_insert",
            "mini": "bf16 fused q/KV norm+RoPE+SWA cache write where enabled; compressed/indexer stores remain bf16.",
            "vllm": "custom op boundary quant-inserts packed fp8_ds_mla SWA cache; compressor/indexer cache insert is also quantized.",
            "alignment": "semantic boundary matches, dtype/layout does not.",
            "portability": "direct port would change cache precision/layout; exact mini needs bf16 adaptation or separate opt-in precision target.",
        },
        {
            "boundary": "sparse_decode_attention",
            "mini": "single two-source bf16 cache sparse attention kernel reads SWA plus C4/C128 cache by int32 indices.",
            "vllm": "compute global topk, gather/dequant fp8 cache to merged bf16 [T,total_topk,512] + invalid mask, then split-K sparse attention core.",
            "alignment": "not same microbench boundary; vLLM splits gather/layout from attention math.",
            "portability": "adapt gather/mask/split-K design for bf16 exact path, or direct-port only under fp8_ds_mla cache target.",
        },
        {
            "boundary": "indexer_logits_select_topk",
            "mini": "bf16 q/cache logits + topk transform, width 512, ratio 4.",
            "vllm": "fused indexer Q RoPE+FP8 quant/weight fold, FP8 paged MQA logits over uint8 indexer cache, persistent_topk.",
            "alignment": "same algorithmic role, different q/cache precision and op partitioning.",
            "portability": "direct port is precision-changing; exact path may borrow global-index/topk buffer discipline.",
        },
        {
            "boundary": "cache_layout",
            "mini": "bf16 flat SWA/C4/C128/indexer caches plus bf16 compressor state rings.",
            "vllm": "byte-packed fp8_ds_mla KV cache and FP8 indexer cache; block/page metadata owned by vLLM attention backends.",
            "alignment": "layout mismatch is central, not incidental.",
            "portability": "new precision/cache target if this is chosen as the main route.",
        },
        {
            "boundary": "graph_runtime_metadata",
            "mini": "GraphRunner replays captured graphs but attention metadata copies remain visible around replay.",
            "vllm": "CudagraphDispatcher, persistent backend buffers, graph capture stream, graph-aware custom all-reduce and pinned output copies.",
            "alignment": "graph sizes overlap [1,2,4], buffer ownership differs.",
            "portability": "optimize mini exact metadata/runtime after attention/indexer boundary is matched.",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-root", default=os.environ.get("VLLM_ROOT", str(DEFAULT_VLLM_ROOT)))
    args = parser.parse_args()

    out = {
        "suite": "target07_393_dispatch_backend_report",
        "generated_at_unix": time.time(),
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "repo_root": str(REPO_ROOT),
            "vllm_root": args.vllm_root,
        },
        "mini": mini_report(),
        "vllm": vllm_report(Path(args.vllm_root)),
        "paired_alignment": paired_alignment(),
        "decision_implications": [
            "vLLM's sm80 attention backend is not simply FlashMLA; it is FlashMLA sparse metadata/cache format plus sm80 reference gather/dequant and Triton split-K decode.",
            "The largest backend mismatch is cache precision/layout: mini exact bf16 flat vs vLLM packed fp8_ds_mla + FP8 indexer cache.",
            "A fair microbench must pair mini bf16 direct sparse attention with vLLM gather/dequant+split-K, not only vLLM's split-K core.",
            "Any direct vLLM cache/indexer port would be a precision/cache experiment, not an exact default optimization.",
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
