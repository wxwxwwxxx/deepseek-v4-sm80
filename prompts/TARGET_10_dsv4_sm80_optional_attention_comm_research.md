# TARGET 10: DSV4 SM80 Optional Attention And Communication Research

## Status

Recommended next bounded target after TARGET 08.30.

TARGET 08.30 closed the prefix-cache milestone and showed that prefix
metadata/runtime is no longer the first bottleneck.  The next evidence-based
surface is decode forward, especially communication/all-reduce owners.  Start
with communication attribution before changing attention kernels or graph
runtime layout.

## Goal

Keep a bounded research plan for surfaces that are real.  The first pass should
focus on communication/all-reduce because it is the current post-prefix profile
leader:

- C4A/C128A sparse attention kernels;
- FlashAttention-style alternatives for selected-token attention;
- PyNCCL and symmetric-memory collectives;
- communication overlap and stream scheduling;
- CUDA graph node/layout introspection;
- small graph/runtime compile opportunities.

## Current State

### Attention

Current promoted decode path uses BF16 sparse split-K for the comparable sparse
decode boundary.  Earlier TARGET 07 evidence showed mini's exact BF16 sparse
decode is essentially at parity with the comparable vLLM gather+split-K probe.

Do not assume C4A/C128A attention is the main gap without a fresh profile.

Possible future probes:

- C4A sparse decode kernel audit if it reappears as a top-two bucket;
- C128A selected-token attention microbench;
- pre-gathered selected-token FlashAttention comparison;
- prefill sparse attention comparison after TARGET 08 changes prefix reuse.

Plain FlashAttention over the full context plus a mask is unlikely to be the
first choice for C128A because it computes many tokens that sparse attention is
trying to skip.  A fair experiment should compare against a gathered selected
token path, not a full-context masked path alone.

### Communication

Current visible collectives include:

- `attn.wo_b` row-parallel all-reduce;
- MoE reduce-once all-reduce;
- embedding all-reduce;
- lm-head all-gather.

TARGET 07.79 counted `2816` collectives and `558.4 GB` counter bytes on the
long macro, but did not isolate clean NCCL wall time.  A communication target
must first collect a dedicated timeline before modifying collectives.

PyNCCL symmetric-memory support exists in mini, but it is not automatically a
win.  Some paths copy tensors into an internal symmetric buffer, run NCCL, then
copy back.  The benefit must beat those extra D2D copies.

Possible future probes:

- PyTorch/NCCL vs PyNCCL repeat-stable macro on the current default path;
- all-reduce wall time and achieved bandwidth with NVTX owner ranges;
- overlap feasibility for `wo_b` and MoE reduce-once;
- NCCL group or stream scheduling changes only after timeline evidence.

### Graph / Runtime

mini already has decode CUDA graph replay in the promoted path.  vLLM eager
ablation proved graph execution is mandatory, but mini's remaining graph/layout
work became fragmented after TARGET 07.

Possible future probes:

- inspect graph node count and captured subgraph boundaries against vLLM;
- identify repeated H2D/D2D copies that remain after prefix cache;
- test narrow `torch.compile` or fused runtime helpers only if a single owner
  exceeds the gate below.

## Gate To Start This Target

TARGET 08.30 already satisfies the gate for a communication-first pass.  For
attention kernels, graph/runtime rewrites, or broader PyNCCL changes, start
implementation only if a fresh post-TARGET08 or post-TARGET09 profile shows one
coherent surface with:

- at least `5%` of decode envelope or `3%` of E2E elapsed;
- plausible `>=2%` E2E improvement;
- clear comparison to vLLM or a hardware roofline reason;
- correctness risk lower than the expected gain.

## Done Criteria

If run, this target should produce:

- one parity table against vLLM for the selected surface;
- one microbench or timeline that isolates the surface;
- one TP8 macro A/B;
- a promote/keep-opt-in/reject decision;
- no broad tuning after the selected bottleneck is disproven.

## Non-Goals

- Continuing generic TARGET 07 polishing.
- Running attention, communication, and graph experiments in one thread.
- Introducing low-precision changes; those belong in TARGET 09.
- Changing prefix-cache ownership; that belongs in TARGET 08.
