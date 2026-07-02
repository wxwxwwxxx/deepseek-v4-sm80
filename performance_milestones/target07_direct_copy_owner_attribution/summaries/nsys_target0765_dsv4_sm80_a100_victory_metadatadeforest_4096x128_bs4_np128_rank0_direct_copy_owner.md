# Direct-Copy Owner Attribution: sqlite

- total direct_copy: `0.732078s` / `189732` kernels
- named owner coverage: `99.97%`
- residual: `0.000246s` (`0.03%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `graph node source: dsv4.shared_experts.gate_up_proj` | `0.166850` | 26982 | `22.79%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | graphNodeId original creation under coarse shared_experts NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.shared_experts.down_proj` | `0.119651` | 26813 | `16.34%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | graphNodeId original creation under coarse shared_experts NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.experts` | `0.054522` | 16319 | `7.45%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.lm_head` | `0.044759` | 381 | `6.11%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward` | graphNodeId original creation under lm_head NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_ffn_pre` | `0.041920` | 10896 | `5.73%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_attn_pre` | `0.038202` | 10714 | `5.22%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.shared` | `0.031365` | 10747 | `4.28%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.kv_quant` | `0.029393` | 10722 | `4.02%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner finalize to fp32.layer*` | `0.023138` | 5417 | `3.16%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.mlp.runner.route` | `0.021914` | 5834 | `2.99%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.indexer_store` | `0.020929` | 10360 | `2.86%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner shared to fp32.layer*` | `0.019963` | 5358 | `2.73%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.q_proj` | `0.016186` | 5350 | `2.21%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress` | `0.015240` | 5275 | `2.08%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.indexer.compressor` | `0.013992` | 5158 | `1.91%` | `python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward` | graphNodeId original creation under coarse indexer NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `attention boundary.positions to i64.layer*` | `0.012029` | 5359 | `1.64%` | `python/minisgl/models/deepseek_v4.py and python/minisgl/attention/deepseek_v4.py` | direct NVTX around attention positions/cache/index dtype or layout staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `moe shared expert staging.runner output to flat dtype.layer*` | `0.011850` | 5354 | `1.62%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.backend` | `0.011582` | 5076 | `1.58%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress_store` | `0.009980` | 5289 | `1.36%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.shared hidden to up dtype` | `0.007751` | 5371 | `1.06%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.indexer` | `0.007194` | 2570 | `0.98%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.engine forward batch.decode.bs*` | `0.005987` | 1515 | `0.82%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `sampler logits staging.next tokens to cpu.bs*` | `0.002001` | 644 | `0.27%` | `python/minisgl/engine/engine.py:Engine.forward_batch` | direct NVTX around sampler/logits token staging; innermost direct-copy NVTX |
| `batch forward bridge.token pool write.decode.bs*` | `0.001493` | 583 | `0.20%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `batch forward bridge.prepare metadata.decode.bs*` | `0.001488` | 630 | `0.20%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_head` | `0.001096` | 381 | `0.15%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.model.hc_expand` | `0.000439` | 127 | `0.06%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare input tuple.decode.bs*` | `0.000350` | 126 | `0.05%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `replay metadata copy.fallback scalar vectors.bs*` | `0.000284` | 127 | `0.04%` | `python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend._copy_metadata_for_replay` | direct NVTX around replay metadata helper/fallback copies; innermost direct-copy NVTX |
| `graph node source: dsv4.model.embed` | `0.000283` | 127 | `0.04%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000246` | 127 | `0.03%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | coarse benchmark NVTX; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000246` | `0.03%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Classifier first uses innermost direct-copy NVTX in the decode envelope.
- If replay kernels are only under coarse graph replay ranges, graphNodeId is mapped through originalGraphNodeId to capture-time direct-copy or dsv4 source NVTX when Nsight exposes it.
- Residual static_graph_replay or batch_forward owners mean more graph-node/source NVTX is needed before an implementation target is safe.
