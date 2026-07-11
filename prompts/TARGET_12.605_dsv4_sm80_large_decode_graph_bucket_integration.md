# TARGET 12.605: DSV4 SM80 Large Decode Graph Bucket Integration

## Status

Planned after TARGET 12.60. Expand from its measured bucket/memory policy.

## Purpose

Implement generated decode CUDA graph buckets and promote the largest useful
release `cuda_graph_max_bs` supported by correctness, graph-pool memory, KV
capacity, and serving throughput.

## Required Gates

- generated buckets match the TARGET 12.60 policy;
- graph inputs and outputs have stable addresses and bounded padding;
- batches above the captured maximum fail over predictably to eager execution;
- graph private-pool memory is included in automatic KV planning;
- batch ladders cover realistic serving waves and representative 1/2/4/8/16/
  32/64/128/256+ decode shapes;
- sampler, C4/C128/SWA metadata, Marlin MoE, PyNCCL, and output gathering remain
  correct at the largest captured bucket;
- promotion preserves small-batch latency and avoids incoherent memory loss.

Do not mix prefill CUDA graph capture into this target.

## Output

```text
performance_milestones/target12_large_decode_graph_bucket_integration/README.md
```

