# TARGET 07.65: Direct-Copy Owner Attribution

Date: 2026-07-02

## Scope

Measurement-only. This target did not implement a performance optimization, did
not promote `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST`, and did not change the
`dsv4_sm80_a100_victory` bundle.

## Artifacts

- `raw/`: symlinks to the captured Nsight reports, sqlite exports, and perf
  matrix output directories.
- `summaries/`: pre-instrumentation controls and post-instrumentation owner
  classifications.
- `scripts/classify_direct_copy_owners.py`: sqlite/sub-boundary classifier.
- `scripts/nsys_direct_copy_owner_4096x128_bs4.sh`: TP8 short profile runner.

## Commands

Pre-instrumentation controls:

```bash
performance_milestones/target07_direct_copy_owner_attribution/scripts/classify_direct_copy_owners.py \
  --subboundary-summary performance_milestones/target07_decode_metadata_deforestation/summaries/baseline_decode_metadata_subboundary.json \
  --json-out performance_milestones/target07_direct_copy_owner_attribution/summaries/preinstrumentation_0763_from_summary_direct_copy_owner.json \
  --md-out performance_milestones/target07_direct_copy_owner_attribution/summaries/preinstrumentation_0763_from_summary_direct_copy_owner.md

performance_milestones/target07_direct_copy_owner_attribution/scripts/classify_direct_copy_owners.py \
  --subboundary-summary performance_milestones/target07_decode_metadata_deforestation/summaries/nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0_decode_metadata_subboundary.json \
  --json-out performance_milestones/target07_direct_copy_owner_attribution/summaries/preinstrumentation_0764_metadatadeforest_from_summary_direct_copy_owner.json \
  --md-out performance_milestones/target07_direct_copy_owner_attribution/summaries/preinstrumentation_0764_metadatadeforest_from_summary_direct_copy_owner.md
```

Required promoted profile:

```bash
performance_milestones/target07_direct_copy_owner_attribution/scripts/nsys_direct_copy_owner_4096x128_bs4.sh
```

Optional 07.64 opt-in stability profile:

```bash
VARIANT=dsv4_sm80_a100_victory_metadatadeforest \
RUN_TAG=target0765_dsv4_sm80_a100_victory_metadatadeforest_4096x128_bs4_np128 \
performance_milestones/target07_direct_copy_owner_attribution/scripts/nsys_direct_copy_owner_4096x128_bs4.sh
```

Validation:

```bash
python -m py_compile \
  python/minisgl/utils/torch_utils.py \
  python/minisgl/engine/graph.py \
  python/minisgl/engine/engine.py \
  python/minisgl/scheduler/scheduler.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  performance_milestones/target07_direct_copy_owner_attribution/scripts/classify_direct_copy_owners.py
```

## Profiles Used

The requested 07.63/07.64 raw sqlite files were not present in this workspace's
milestone `raw/` directories. Pre-instrumentation control therefore reused the
existing checked-in sub-boundary JSON summaries:

| Profile | Source | direct_copy s | Attribution |
| --- | --- | ---: | --- |
| 07.63 promoted control | `baseline_decode_metadata_subboundary.json` | `0.736769` | coarse `batch_forward` / `batch_forward_enqueue` only |
| 07.64 opt-in control | `nsys_target0764_metadatadeforest_..._decode_metadata_subboundary.json` | `0.731834` | coarse `batch_forward` / `batch_forward_enqueue` only |
| 07.65 promoted required | `raw/nsys_target0765_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0.sqlite` | `0.737039` | direct NVTX + `originalGraphNodeId` source mapping |
| 07.65 07.64 opt-in stability | `raw/nsys_target0765_dsv4_sm80_a100_victory_metadatadeforest_4096x128_bs4_np128_rank0.sqlite` | `0.732078` | direct NVTX + `originalGraphNodeId` source mapping |

## Instrumentation Summary

New env flag: `MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1`.

The flag is default-off. The helper returns a null context when disabled, and
the default runtime path is unchanged. The instrumentation adds NVTX only; it
does not add synchronization, does not allocate CUDA storage, and does not
change any CUDA graph or tensor-copy semantics.

Instrumented owner boundaries:

- graph input staging in `python/minisgl/engine/graph.py`;
- replay metadata copy/static graph input updates in
  `python/minisgl/attention/deepseek_v4.py`;
- attention dtype/layout staging in `python/minisgl/models/deepseek_v4.py` and
  `python/minisgl/attention/deepseek_v4.py`;
- MoE/shared expert dtype staging in `python/minisgl/models/deepseek_v4.py`;
- sampler/logits staging in `python/minisgl/engine/engine.py`;
- scheduler batch-forward bridge in `python/minisgl/scheduler/scheduler.py`.

The classifier also maps replay kernels through `CUDA_GRAPH_NODE_EVENTS`:
`graphNodeId -> originalGraphNodeId -> capture-time direct-copy/dsv4 NVTX`.
That was necessary because graph replay kernels execute asynchronously after
Python enqueue ranges; adding synchronization to force range overlap would have
violated this target.

## Graph Replay / Eager Decode

| Variant | Status | Graph replay | Greedy replay | Eager decode | Decode tok/s | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | pass | `127` | `127` | `0` | `135.4181` | `48.3008` |
| `dsv4_sm80_a100_victory_metadatadeforest` | pass | `127` | `127` | `0` | `135.6446` | `48.0162` |

Text smoke was not rerun because the only runtime edits are default-off NVTX
contexts. The profiled runs validate that graph replay remains active and eager
decode remains `0` with instrumentation enabled.

## Direct-Copy Owner Table

Promoted `dsv4_sm80_a100_victory`, 4096/128/batch4, rank0:

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `dsv4.shared_experts.gate_up_proj` | `0.165751` | `26802` | `22.49%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | `originalGraphNodeId` to capture-time `dsv4.shared_experts.gate_up_proj` |
| `dsv4.shared_experts.down_proj` | `0.119724` | `26835` | `16.24%` | `python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward` | `originalGraphNodeId` to capture-time `dsv4.shared_experts.down_proj` |
| `dsv4.layer*.mlp.runner.experts` | `0.053714` | `16072` | `7.29%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | `originalGraphNodeId` to capture-time MLP runner NVTX |
| `dsv4.lm_head` | `0.044720` | `381` | `6.07%` | `python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward` | `originalGraphNodeId` to capture-time `lm_head` |
| `dsv4.layer*.hc_ffn_pre` | `0.041722` | `10826` | `5.66%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | hidden-carrier staging source NVTX |
| `dsv4.layer*.hc_attn_pre` | `0.038527` | `10794` | `5.23%` | `python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward` | hidden-carrier staging source NVTX |
| `dsv4.layer*.mlp.runner.shared` | `0.031286` | `10722` | `4.24%` | `python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward` | MLP shared path source NVTX |
| `dsv4.layer*.attn.kv_quant` | `0.029743` | `10842` | `4.04%` | `python/minisgl/models/deepseek_v4.py:DSV4Attention.forward` | attention source NVTX |
| `moe_shared_expert_staging.runner_finalize_to_fp32.layer*` | `0.022872` | `5354` | `3.10%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner` | direct-copy NVTX |
| `dsv4.layer*.mlp.runner.route` | `0.021675` | `5773` | `2.94%` | `python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner.forward` | MLP route source NVTX |

Full table: `summaries/nsys_target0765_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0_direct_copy_owner.md`.

Coverage:

| Metric | Value |
| --- | ---: |
| total direct_copy | `0.737039s` |
| named owner direct_copy | `0.736794s` |
| named owner coverage | `99.97%` |
| residual | `0.000245s` |

## Owner Groups

| Clean owner group | Kernel s | Share |
| --- | ---: | ---: |
| MoE/shared expert staging | `0.379204` | `51.45%` |
| attention/indexer boundary | `0.138539` | `18.80%` |
| hidden-carrier staging | `0.080656` | `10.94%` |
| MoE routed runner | `0.075389` | `10.23%` |
| sampler/logits/head | `0.048699` | `6.61%` |
| batch forward bridge | `0.013735` | `1.86%` |
| graph/replay metadata | `0.000290` | `0.04%` |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `batch_forward:decode:bs*:padded*` | `0.000245` | `0.03%` | not worth another NVTX pass |

## 07.64 Opt-In Stability

The optional `dsv4_sm80_a100_victory_metadatadeforest` profile does not change
the direct-copy owner shape. It remains an opt-in ablation, not the promoted
baseline.

| Owner group | Promoted s | 07.64 opt-in s | Stable? |
| --- | ---: | ---: | --- |
| MoE/shared expert staging | `0.379204` | `0.380568` | yes |
| attention/indexer boundary | `0.138539` | `0.136524` | yes |
| hidden-carrier staging | `0.080656` | `0.080561` | yes |
| MoE routed runner | `0.075389` | `0.076437` | yes |
| sampler/logits/head | `0.048699` | `0.047856` | yes |
| batch forward bridge | `0.013735` | `0.009319` | small metadata-copy-side movement only |

## Next Target Recommendation

Exactly one recommendation: **TARGET 07.66: MoE/shared-expert direct-copy staging cleanup**.

Rationale: the clean owner group is `0.379204s` in 4096/128/batch4 direct_copy
(`51.45%`), and the single largest owner,
`dsv4.shared_experts.gate_up_proj`, is `0.165751s`, above the `0.15s` gate.
The target should focus on the shared-expert gate/up/down and runner shared
dtype/materialization boundaries, then prove whether direct-copy nodes can be
removed or moved without changing MoE numerics or graph replay. A plausible
4096/1024 output-token upside is roughly `0.5%` to `2%` if only a fraction of
the group is eliminated; full removal is an upper-bound scenario and should not
be assumed.

Because the dominant owner is MoE/shared-expert related, this thread stops at
measurement per the TARGET 07.65 stop condition. No MoE implementation work is
included here.

## Do-Not-Continue Conditions

- Do not implement MoE, projection, communication, precision, or cache-manager
  changes inside TARGET 07.65.
- Do not promote `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST`.
- Do not continue if graph replay drops or eager decode becomes nonzero.
- Do not pursue an implementation target unless a finer split keeps a clean
  owner group at or above `0.15s`.
- Do not use synchronous profiling to make Python NVTX cover asynchronous graph
  replay kernels.

## Git State

Dirty files are limited to profiling-only instrumentation and this milestone:

- `python/minisgl/utils/torch_utils.py`
- `python/minisgl/utils/__init__.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `performance_milestones/target07_direct_copy_owner_attribution/`
