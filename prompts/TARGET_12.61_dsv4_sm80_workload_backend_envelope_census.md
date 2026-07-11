# TARGET 12.61: DSV4 SM80 Workload Backend Envelope Census

## Status

Planned after the 1M and large-decode-graph routes converge. Review and split
into implementation targets after the census identifies measured owners.

## Purpose

Scan the release engine across context length, prefill token count, decode
batch size, and backend `M` regimes. Find kernels or fallback paths whose
dispatch, temporary memory, arithmetic intensity, or scaling is unsuitable for
their workload, then rank focused backend adaptations.

## Initial Matrix

Include representative regimes rather than a Cartesian explosion:

```text
short/medium/long prefill with chunk M up to the release maximum
decode bs 1 through the promoted cuda_graph_max_bs
short and long cached context
prefix-hit and no-prefix cases
C4A, C128A, indexer, HC, MoE, projection, sampler, and communication owners
```

Start with no-weight/one-layer microbench and actual backend census. Use full
macro only after one owner is implicated. Compare mini against SGLang/vLLM
dispatch and mature kernels before implementing a new backend.

The first known long-context performance candidate is bounded FP8 paged
indexer select: TARGET 12.58 measured it at about 48% of 512k TTFT. Evaluate a
streaming/fused logits+top-k backend against the current bounded Triton+CUDA
oracle, but do not assume it remains the top owner after C128 and graph changes.

For each owner record compute/memory intensity, temporary bytes, launch count,
backend guard, A100 roofline context, and expected macro upside. Open focused
implementation targets only for material owners; stop polishing when the
remaining delta is small or another subsystem dominates.

## Output

```text
performance_milestones/target12_workload_backend_envelope_census/README.md
```

