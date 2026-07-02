# Direct-Copy Owner Attribution: sqlite

- total direct_copy: `0.737039s` / `191622` kernels
- named owner coverage: `99.97%`
- residual: `0.000245s` (`0.03%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `graph node source: dsv4.shared_experts.gate_up_proj` | `0.165751` | 26802 | `22.49%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | graphNodeId original creation under coarse shared_experts NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.shared_experts.down_proj` | `0.119724` | 26835 | `16.24%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | graphNodeId original creation under coarse shared_experts NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.experts` | `0.053714` | 16072 | `7.29%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.lm_head` | `0.044720` | 381 | `6.07%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward` | graphNodeId original creation under lm_head NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_ffn_pre` | `0.041722` | 10826 | `5.66%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.hc_attn_pre` | `0.038527` | 10794 | `5.23%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | graphNodeId original creation under hidden-carrier staging NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.mlp.runner.shared` | `0.031286` | 10722 | `4.24%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.kv_quant` | `0.029743` | 10842 | `4.04%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner finalize to fp32.layer*` | `0.022872` | 5354 | `3.10%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.mlp.runner.route` | `0.021675` | 5773 | `2.94%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | graphNodeId original creation under coarse layer MLP/MoE NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.indexer_store` | `0.021488` | 10628 | `2.92%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.runner shared to fp32.layer*` | `0.020026` | 5374 | `2.72%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.q_proj` | `0.016408` | 5420 | `2.23%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress` | `0.015381` | 5319 | `2.09%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.indexer.compressor` | `0.014392` | 5306 | `1.95%` | `python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward` | graphNodeId original creation under coarse indexer NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `attention boundary.positions to i64.layer*` | `0.012147` | 5414 | `1.65%` | `python/minisgl/models/deepseek_v4.py and python/minisgl/attention/deepseek_v4.py` | direct NVTX around attention positions/cache/index dtype or layout staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `moe shared expert staging.runner output to flat dtype.layer*` | `0.011815` | 5394 | `1.60%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.backend` | `0.011540` | 5063 | `1.57%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.layer*.attn.compress_store` | `0.010039` | 5320 | `1.36%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `moe shared expert staging.shared hidden to up dtype` | `0.007730` | 5357 | `1.05%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts` | direct NVTX around MoE/shared expert dtype staging; graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX |
| `graph node source: dsv4.layer*.attn.indexer` | `0.007402` | 2643 | `1.00%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | graphNodeId original creation under coarse layer attention NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare metadata.decode.bs*` | `0.006477` | 2520 | `0.88%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `batch forward bridge.engine forward batch.decode.bs*` | `0.004228` | 992 | `0.57%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `sampler logits staging.next tokens to cpu.bs*` | `0.002883` | 715 | `0.39%` | `python/minisgl/engine/engine.py:Engine.forward_batch` | direct NVTX around sampler/logits token staging; innermost direct-copy NVTX |
| `batch forward bridge.token pool write.decode.bs*` | `0.002681` | 741 | `0.36%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `graph node source: dsv4.model.hc_head` | `0.001096` | 381 | `0.15%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `graph node source: dsv4.model.hc_expand` | `0.000407` | 127 | `0.06%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `batch forward bridge.prepare input tuple.decode.bs*` | `0.000349` | 126 | `0.05%` | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward` | direct NVTX around scheduler to engine bridge; innermost direct-copy NVTX |
| `replay metadata copy.fallback scalar vectors.bs*` | `0.000290` | 127 | `0.04%` | `python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend._copy_metadata_for_replay` | direct NVTX around replay metadata helper/fallback copies; innermost direct-copy NVTX |
| `graph node source: dsv4.model.embed` | `0.000282` | 127 | `0.04%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward` | graphNodeId original creation under coarse model NVTX; graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000245` | 127 | `0.03%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | coarse benchmark NVTX; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs*:padded*` | `0.000245` | `0.03%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Classifier first uses innermost direct-copy NVTX in the decode envelope.
- If replay kernels are only under coarse graph replay ranges, graphNodeId is mapped through originalGraphNodeId to capture-time direct-copy or dsv4 source NVTX when Nsight exposes it.
- Residual static_graph_replay or batch_forward owners mean more graph-node/source NVTX is needed before an implementation target is safe.
