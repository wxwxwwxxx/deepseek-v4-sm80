# TARGET 08.21.1: DSV4 Component Loc Table Preflight

## Status

Active next TARGET 08 subtarget.

This is Route B / B0.  Run this before implementing independent component
allocators or freeing old full/SWA pages.

## Goal

Prove that mini can represent DSV4 component locations independently from
full-token locations at the metadata boundary.

This target does **not** implement runtime component retention.  Full pages stay
alive.  The goal is to build component loc table abstractions and probes that
produce the same effective metadata as today's `full_loc // ratio` derivation.

The target should answer:

1. Can C4, C128, indexer, and optional SWA loc tables be represented separately
   from the full page table?
2. Can DSV4 attention/indexer metadata consume those loc tables directly?
3. Which current paths still assume `full_loc // ratio`:
   - Python metadata construction;
   - Triton/CUDA metadata deforest;
   - indexer logits wrappers;
   - graph metadata copy;
   - compression store locations.
4. Can existing sparse attention kernels consume direct component locs without
   kernel rewrites?
5. What must TARGET 08.21.2 change to allocate and own those locs independently?

## Required Reading

- `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `performance_milestones/target08_swa_tail_retention_v1/README.md`
- `performance_milestones/target08_swa_tail_retention_v1/DESIGN.md`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- SGLang references listed in the 08.21 overview.

## Scope

Allowed:

- add small metadata dataclasses/helpers for direct component loc tables;
- add probes comparing direct component loc metadata against phase-1 derived
  metadata while full pages are live;
- add test-only Route A materialization oracle if useful for equality checks;
- add documentation and stop conditions for B1.

Not allowed:

- freeing old full/SWA pages;
- independent component allocators/refcounts;
- runtime retained-store materialization as serving path;
- graph/deforest integration beyond mapping dependencies;
- default promotion.

## Required Design Note

Create:

```text
performance_milestones/target08_component_loc_table_preflight/DESIGN.md
```

It must include:

- current mini coupling map:
  - full page table;
  - SWA indices;
  - C4/C128/indexer loc derivation;
  - compression-state loc derivation;
  - graph metadata copy and deforest dependencies.
- proposed direct loc table shapes and dtypes:
  - C4 loc table;
  - C128 loc table;
  - indexer loc table;
  - optional SWA loc table or full-to-SWA map;
  - state loc placeholder.
- exact list of call sites that must change in B1/B2/B3.
- Route A oracle scope if any.

## Required Probes

Build a narrow probe suite under the milestone directory.

At minimum:

- generate phase-1 metadata for page size `256`;
- build equivalent direct component loc tables from the same live full page
  table;
- compare:
  - `swa_page_indices`;
  - `c4_sparse_page_indices`;
  - `c128_page_indices`;
  - `c4_out_loc`;
  - `c128_out_loc`;
  - `c4_indexer_out_loc`;
  - indexer page/loc table inputs;
  - state loc derivation, or document why state remains placeholder.
- cover:
  - full hit, partial hit, miss-style metadata;
  - page boundary `255/256/257/258`;
  - C4 boundary;
  - C128 boundary;
  - SWA boundary `127/128/129`;
  - batched same-layout rows.

The probe should report exact equality while full pages are live.  Any mismatch
must be traced before proceeding to B1.

## Deliverables

Create:

```text
performance_milestones/target08_component_loc_table_preflight/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- SGLang parity map relevant to loc translation;
- direct component loc table schema;
- equality table against phase-1 derived metadata;
- call-site dependency table for B1/B2/B3;
- Route A oracle usage if any;
- decision: proceed to TARGET 08.21.2, revise B0, or block Route B.

## Decision Rules

Proceed to TARGET 08.21.2 only if:

- direct C4/C128/indexer loc tables can exactly reproduce phase-1 metadata while
  full pages remain live;
- existing sparse attention kernels can consume direct `compressed_indices` or
  the required change is localized to metadata construction;
- indexer logits path dependencies are explicitly mapped;
- state dependencies are either mapped or deliberately deferred to TARGET
  08.21.3.

Stop and report blocked if:

- direct component loc metadata requires broad attention-kernel rewrites before
  any allocator work;
- equality probes cannot reproduce phase-1 metadata;
- graph/deforest assumptions make even eager B1 impossible without a broader
  runtime rewrite.
