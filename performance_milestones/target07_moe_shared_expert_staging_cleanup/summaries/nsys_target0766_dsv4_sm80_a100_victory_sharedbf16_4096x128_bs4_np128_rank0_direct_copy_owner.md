# Direct-Copy Owner Attribution: sqlite

- total direct_copy: `0.449052s` / `137012` kernels
- named owner coverage: `99.94%`
- residual: `0.000252s` (`0.06%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `graph node source: dsv4.layer*.mlp.runner.experts` | `0.054546` | 16326 | `12.15%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.lm_head` | `0.044360` | 381 | `9.88%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward` | graphNodeId original creation under lm_head NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_ffn_pre` | `0.041932` | 10874 | `9.34%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_attn_pre` | `0.038834` | 10768 | `8.65%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.shared` | `0.031721` | 10846 | `7.06%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.kv_quant` | `0.029445` | 10737 | `6.56%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner finalize to fp32.layer*` | `0.023179` | 5429 | `5.16%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `moe shared expert staging.runner shared to fp32.layer*` | `0.022688` | 5414 | `5.05%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.mlp.runner.route` | `0.021900` | 5823 | `4.88%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.indexer_store` | `0.020904` | 10346 | `4.66%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.q_proj` | `0.016153` | 5368 | `3.60%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress` | `0.015069` | 5224 | `3.36%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.indexer.compressor` | `0.014007` | 5166 | `3.12%` | `python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward` | graphNodeId original creation under coarse indexer NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `attention boundary.positions to i64.layer*` | `0.012185` | 5372 | `2.71%` | `python/minisgl/models/deepseek_v4.py and python/minisgl/attention/deepseek_v4.py` | direct NVTX around attention positions/cache/index dtype or layout staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `moe shared expert staging.runner output to flat dtype.layer*` | `0.011906` | 5388 | `2.65%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.backend` | `0.011366` | 5074 | `2.53%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress_store` | `0.009892` | 5241 | `2.20%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.shared hidden to up dtype` | `0.007866` | 5417 | `1.75%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.indexer` | `0.007253` | 2578 | `1.62%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare metadata.decode.bs*` | `0.006479` | 2520 | `1.44%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `batch forward bridge.engine forward batch.decode.bs*` | `0.002311` | 816 | `0.51%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `sampler logits staging.next tokens to cpu.bs*` | `0.001313` | 480 | `0.29%` | `python/minisgl/engine/engine.py:Engine.forward_batch` | direct NVTX around sampler/logits token staging; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_head` | `0.001109` | 381 | `0.25%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.token pool write.decode.bs*` | `0.001058` | 409 | `0.24%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_expand` | `0.000403` | 127 | `0.09%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare input tuple.decode.bs*` | `0.000351` | 126 | `0.08%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `replay metadata copy.fallback scalar vectors.bs*` | `0.000289` | 127 | `0.06%` | `python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend._copy_metadata_for_replay` | direct NVTX around replay metadata helper/fallback copies; innermost direct-copy NVTX |
| `graph node source: dsv4.model.embed` | `0.000282` | 127 | `0.06%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000252` | 127 | `0.06%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | coarse benchmark NVTX; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000252` | `0.06%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Classifier first uses innermost direct-copy NVTX in the decode envelope.
- If replay kernels are only under coarse graph replay ranges, graphNodeId is mapped through originalGraphNodeId to capture-time direct-copy or dsv4 source NVTX when Nsight exposes it.
- Residual static_graph_replay or batch_forward owners mean more graph-node/source NVTX is needed before an implementation target is safe.
