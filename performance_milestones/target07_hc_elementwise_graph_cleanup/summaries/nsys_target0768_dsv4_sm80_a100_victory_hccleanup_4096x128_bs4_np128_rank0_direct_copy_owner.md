# Direct-Copy Owner Attribution: sqlite

- total direct_copy: `0.412174s` / `126090` kernels
- named owner coverage: `99.95%`
- residual: `0.000217s` (`0.05%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `graph node source: dsv4.layer*.mlp.runner.experts` | `0.054274` | 16254 | `13.17%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.lm_head` | `0.044338` | 380 | `10.76%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward` | graphNodeId original creation under lm_head NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.shared` | `0.030971` | 10815 | `7.51%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.kv_quant` | `0.029564` | 10753 | `7.17%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_ffn_pre` | `0.023566` | 5386 | `5.72%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner finalize to fp32.layer*` | `0.023153` | 5410 | `5.62%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `moe shared expert staging.runner shared to fp32.layer*` | `0.022567` | 5397 | `5.48%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.mlp.runner.route` | `0.021849` | 5784 | `5.30%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.indexer_store` | `0.021634` | 10647 | `5.25%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_attn_pre` | `0.020875` | 5387 | `5.06%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.q_proj` | `0.016180` | 5377 | `3.93%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress` | `0.015402` | 5322 | `3.74%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.indexer.compressor` | `0.014291` | 5320 | `3.47%` | `python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward` | graphNodeId original creation under coarse indexer NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner output to flat dtype.layer*` | `0.012214` | 5388 | `2.96%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `attention boundary.positions to i64.layer*` | `0.012122` | 5380 | `2.94%` | `python/minisgl/models/deepseek_v4.py and python/minisgl/attention/deepseek_v4.py` | direct NVTX around attention positions/cache/index dtype or layout staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.backend` | `0.010948` | 4923 | `2.66%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress_store` | `0.010037` | 5322 | `2.44%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.shared hidden to up dtype` | `0.007920` | 5403 | `1.92%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.indexer` | `0.007473` | 2658 | `1.81%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare metadata.decode.bs*` | `0.006425` | 2520 | `1.56%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `batch forward bridge.engine forward batch.decode.bs*` | `0.001791` | 604 | `0.43%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_head` | `0.001120` | 378 | `0.27%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `sampler logits staging.next tokens to cpu.bs*` | `0.001031` | 357 | `0.25%` | `python/minisgl/engine/engine.py:Engine.forward_batch` | direct NVTX around sampler/logits token staging; innermost direct-copy NVTX |
| `batch forward bridge.token pool write.decode.bs*` | `0.000891` | 291 | `0.22%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_expand` | `0.000402` | 127 | `0.10%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare input tuple.decode.bs*` | `0.000353` | 126 | `0.09%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `replay metadata copy.fallback scalar vectors.bs*` | `0.000286` | 127 | `0.07%` | `python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend._copy_metadata_for_replay` | direct NVTX around replay metadata helper/fallback copies; innermost direct-copy NVTX |
| `graph node source: dsv4.model.embed` | `0.000282` | 127 | `0.07%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000217` | 127 | `0.05%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | coarse benchmark NVTX; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000217` | `0.05%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Classifier first uses innermost direct-copy NVTX in the decode envelope.
- If replay kernels are only under coarse graph replay ranges, graphNodeId is mapped through originalGraphNodeId to capture-time direct-copy or dsv4 source NVTX when Nsight exposes it.
- Residual static_graph_replay or batch_forward owners mean more graph-node/source NVTX is needed before an implementation target is safe.
