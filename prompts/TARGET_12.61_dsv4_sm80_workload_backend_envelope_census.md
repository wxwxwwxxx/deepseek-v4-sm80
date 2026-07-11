# TARGET 12.61: DSV4 SM80 Workload And Large-M Backend Envelope Census

## Status

Planned after TARGET 12.606 promotes the selected graph recipes. Review and
split implementation targets only after measured owners are ranked.

## Purpose

Measure how the release engine and major backends scale across context length,
prefill chunk M, and decode M. Determine whether a kernel, communication path,
fallback, temporary, or dispatch choice becomes materially unsuitable at
larger practical batches.

This target produces evidence for kernel work; it does not assume larger M
requires rewritten kernels.

## Primary Range

```text
decode/backend M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
```

Use exact-M and selected-recipe padded-M rows where applicable. Treat the
TARGET 12.605/12.606 balanced and optional high-concurrency recipes as the
release envelopes. Values above the promoted recipe and through 512 justify
kernel work only when a supported named high-throughput recipe has credible
serving demand. Keep 1024/2048 as isolated capability smoke only.

## Required Work

1. Build no-weight, one-layer, or subgraph microbenches before full macros. Use
   production kernels and shapes rather than synthetic GEMMs that omit routing,
   metadata, graph padding, or communication.
2. For each representative M, record per-step latency, aggregate token
   throughput, padded-work efficiency, launch count, temporary bytes, selected
   backend, graph/eager mode, resolved bucket, graph reserve, and effective KV
   capacity.
3. Attribute GPU time to:

```text
C4A / C128A / indexer
HC and dense projections
MoE routing, Marlin expert GEMMs, shared expert, and reductions
PyNCCL/NCCL collectives
lm_head, logits, sampler, and output gathering
metadata and cache writes
```

4. Compare actual mini dispatch and representative subgraph performance with
   SGLang and vLLM on DSV4 sm80. Adapt mature implementations before designing
   a new backend.
5. Report arithmetic intensity, effective bandwidth or tensor-core use where
   measurable, A100 roofline context, scaling slope, and expected macro upside
   for each material owner.
6. Include short and long cached contexts, prefix hit/no-hit, and chunked
   prefill without creating a Cartesian workload explosion.
7. Re-rank the bounded FP8 indexer, previously about 48% of 512k TTFT, after
   graph/C128 changes. Evaluate streaming/fused logits+top-k only if it remains
   material.
8. Use TARGET 12.604 resolved-policy telemetry so exact/padded/eager rows cannot
   be mislabeled. Separate true kernel scaling from expected padded work, eager
   launch overhead, and graph-memory capacity tradeoffs.
9. Keep balanced, high-concurrency, and long-context profiles separate. A
   kernel material only in an unpromoted graph512 research shape must not outrank
   a smaller release-recipe owner.

## Kernel-Target Rule

Open a focused optimization target only when all are true:

- the owner is material in a release-relevant workload;
- scaling or backend parity shows credible headroom;
- expected E2E gain exceeds measurement noise and integration cost;
- the work is not merely for M>512 or an unsupported research-only policy;
- SGLang/vLLM does not already provide an adaptable solution.

If throughput rises normally with M and no owner is anomalous, conclude that no
large-M kernel rewrite is currently justified.

## Output

```text
performance_milestones/target12_workload_backend_envelope_census/README.md
```
