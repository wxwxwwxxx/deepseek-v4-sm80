# TARGET 07.55 Remaining Graph/Layout Candidate Summary

- Source classified JSON: `/workspace/mini-sglang/performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.json`
- Source SQLite: `/workspace/mini-sglang/performance_milestones/target07_graph_layout_replay_deforestation/raw/nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0.sqlite`
- Decode envelope wall: `5.474079 s`
- 07.54 graph/layout cluster: `1.827097 s`
- 10% graph/layout gate: `0.182710 s`

Candidate groups below are evidence slices, not additive totals; some kernel-name groups overlap with bucket-level classifications.

| Candidate group | Duration s | Cluster share | Count | Graph nodes | Top kernel evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| remaining_direct_copy_kernels | `0.945592` | `51.75%` | `191622` | `1487` | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tens...` |
| bf16_and_float8_copy_kernels | `0.131814` | `7.21%` | `40894` | `322` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_c...` |
| cat_index_gather_topk_assembly | `0.183043` | `10.02%` | `65789` | `488` | `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detai...` |
| pow_mean_mul_elementwise_nodes | `0.514829` | `28.18%` | `143373` | `1119` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, f...` |
| projection_gemm_intrinsic | `1.796796` | `98.34%` | `100965` | `795` | `_quantized_linear_fp8_kernel` |
| fp8_activation_quant_poc_kernel | `0.075900` | `4.15%` | `35433` | `279` | `_fp8_activation_quantize_kernel` |

## Bucket Baseline

| Bucket | Kernel s | Count | Graph nodes |
| --- | ---: | ---: | ---: |
| `elementwise_graph_nodes` | `0.639607` | `207131` | `1594` |
| `fp8_activation_quant_poc` | `0.075900` | `35433` | `279` |
| `fp8_indexer` | `0.131115` | `20828` | `164` |
| `graph_runtime_copy_cat_index` | `1.187490` | `288963` | `2217` |
| `kv_compressor_cache_store` | `0.028107` | `8128` | `64` |
| `moe_marlin` | `0.316973` | `43688` | `344` |
| `nccl_communication` | `0.342830` | `11176` | `88` |
| `projection_gemm` | `1.796796` | `100965` | `795` |
| `sampling_logits` | `0.183811` | `43815` | `345` |
| `sparse_attention_decode` | `0.117934` | `21590` | `170` |
| `unknown` | `0.023531` | `11301` | `87` |

Decision encoded by TARGET 07.55 README: `pivot to projection/GEMM backend parity`
