# TARGET 11.8: DSV4 SM80 MTP Target-Verify Runtime Contract Unification

## Status

Next after TARGET 11.7.

TARGET 11.7 should be treated as a stop sign for per-batch patching.  It did
not leave the rejected experiments in the final code; final code remains at the
TARGET 11.6 behavior:

```text
bs=1/2/4/5 exact
bs=6 fails for req0 and req3
accepted commit enabled
```

The important TARGET 11.7 result is the contract split:

```text
current 11.6:
  parent>2 active 2/3 -> force_torch attention + separate exact KV store
  passes bs=4/5, fails bs=6 req0/req3

KV-unify experiment:
  parent>2 active 2/3 -> force_torch attention + fused KV store
  fixes bs=6 req3, fails bs=4/5 and leaves req0

attention-unify experiment:
  parent>2 active 2/3 -> splitk target_verify attention + fused KV store
  fixes bs=6 req0, fails bs=6 req3/req4
```

This is no longer a single batch-size bug.  Mini currently has multiple
target-verify numerical contracts selected by parent batch size, active verify
length, attention backend, and KV-store path.  A new local `if bs == ...` style
fix is likely to make another green cell while regressing an older one.

## Goal

Define and implement, or prove the need for, a single MTP target-verify runtime
contract for DSV4 on A100/sm80.

The target passes when one of these is true:

1. A correctness-first unified target-verify runtime passes:

```text
bs=1/2/4/5/6 exact
accepted commit enabled
bs=7/8/16 light exposure either passes or fails with a new concrete owner
```

2. Or, if implementation is too broad for one target, a precise contract and
port plan is written with enough evidence that the next target can implement it
without returning to per-batch patching.

Do not start CUDA graph or throughput work in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_bs6_path_census_contract_closure/README.md
performance_milestones/target11_mtp_bs5_exposure_state_parity/README.md
performance_milestones/target11_mtp_bs4_accepted_commit_state_parity/README.md
```

Current exactness state:

```text
bs=1 exact
bs=2 exact
bs=4 exact
bs=5 exact
bs=6 not exact: req0 and req3
```

Known owners:

```text
bs=4 class:
  target-verify attention consumer for parent batch sizes > 2

bs=5 class:
  single-row target verify used a separate KV norm/rope/store path instead of
  matching normal fused q_kv_norm_rope_store semantics

bs=6 class:
  force_torch/separate-KV, force_torch/fused-KV, and splitk/fused-KV each fix
  one subset and regress another subset
```

Interpretation:

```text
parent batch size and verify shape are currently choosing numerical semantics.
They should only choose scheduling/performance strategy.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/csrc/jit/dsv4_online_c128_mtp.cu
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/spec_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
```

Prefer SGLang's contract when mini and SGLang differ, unless mini has a
documented reason to intentionally diverge.

## Contract To Write

Create a contract section in the report, and optionally a separate markdown file
under the milestone directory, covering at least:

### 1. Row and Token Semantics

Define for top-k 1 / `draft_len=2`:

- verify row order;
- input token for each depth;
- position for each depth;
- active vs padded rows;
- correction row;
- bonus row if present;
- rejected/padded row invisibility.

Do not rely on comments from prior targets alone.  Reconfirm against SGLang.

### 2. Attention Semantics

For each target-verify row, define exactly what keys/values it may attend to:

- committed prefix;
- earlier verify rows for the same request, if the SGLang contract requires
  them;
- no rows from other requests;
- no rejected or padded future rows;
- same causal result as the corresponding normal target decode prefix.

State whether the chosen mini implementation uses torch attention, split-k
decode-row attention, base target-verify attention, or a new wrapper.  The
contract should be independent of the backend name.

### 3. KV Producer And Store Semantics

Accepted target-verify rows must produce long-lived state equivalent to normal
target decode for the same visible prefix:

- full/SWA KV;
- C4 compressed cache;
- C4 indexer cache/state;
- C128 compressed cache;
- online C128 MTP pending/write/commit state;
- page/component mapping;
- request `cached_len` and `device_len`.

Specify one numerical contract for fused vs separate KV norm/rope/store.  Parent
batch size must not choose a different numerical contract.

### 4. Accepted Commit Semantics

Define:

- `accepted_prefix`;
- `copy_rows`;
- correction row commit;
- bonus row commit;
- `req.complete_one()` order;
- rollback of unaccepted rows;
- behavior at page, C4, and C128 boundaries.

### 5. Shape Independence

The same visible target prefix should not change target-verify semantics because
of:

- parent batch size;
- verify group size;
- active verify length 1/2/3;
- request slot/table index;
- request order in the batch.

If normal target decode itself is not globally batch-invariant, document the
scope carefully.  MTP must at minimum be exact against the same-batch baseline
for a fixed scheduler configuration.

## Work Plan

### 1. Reconfirm SGLang's runtime contract

Inspect the SGLang files listed above and produce a source-parity table:

```text
concept
SGLang behavior
current mini behavior
same/different/missing
correctness risk
implementation action
```

Focus on DSV4-specific target verify, online C128 MTP, and attention metadata.
Do not spend time on unrelated speculative decoding modes.

### 2. Convert TARGET 11.7 census into a contract matrix

Start from the TARGET 11.7 path census and reframe it as:

```text
which branches are scheduling-only
which branches currently change numerical semantics
which branches must be eliminated or unified
```

Examples of branches that should become scheduling-only:

- parent batch size;
- verify group size;
- active/padded verify length;
- force-torch vs split-k fast-path choice, once both implement the same
  contract;
- fused vs separate KV-store choice, if both can be made numerically equivalent.

### 3. Build a correctness-first unified oracle mode

Before optimizing, build or expose a single correctness-first target-verify
mode that follows the written contract.

Acceptable as an oracle:

- slower than the final path;
- uses extra debug synchronization;
- uses torch/reference kernels;
- emits detailed per-layer parity.

Not acceptable as the final candidate:

- disables accepted commit;
- sequentially recomputes accepted rows after verification and calls that MTP;
- forces all serving decode to single request;
- switches behavior based on observed token ids or specific batch sizes.

The oracle may be used to prove the contract, but the final candidate should be
a real target-verify runtime path.

### 4. Implement the smallest runtime unification

Possible implementation directions:

1. A dedicated `mtp_target_verify` execution mode that calls the same attention
   and KV-store semantics for all active verify lengths.
2. A SGLang-aligned port of the DSV4 target-verify attention/metadata path.
3. A conservative mini-owned reference backend for target verify, kept opt-in,
   that is exact for all current exposure cases.
4. A cleanup that removes or centralizes current special flags:

```text
dsv4_force_exact_kv_store
dsv4_force_torch_attention
dsv4_target_verify_decode_rows
sticky parent batch size
verify group size fallback
```

Do not add a fourth numerical contract.  If a new helper is introduced, it must
replace or centralize the old branch decisions.

### 5. Validate exactness before performance

Run:

```text
bs=1/2/4/5/6 exact
bs=7/8/16 light exposure
draft_len=2
decode_len=8
page_size=256
TP8
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
accepted commit enabled
```

If possible, add one page/C4/C128 boundary-focused short test.  If it is too
expensive, document it as a required follow-up.

### 6. Decide next step

If exactness passes through `bs=16`, TARGET 11.3 can start after a small
acceptance/throughput sanity check.

If exactness still fails, report:

- first failing batch size;
- first failing request/token;
- whether it is within the unified contract or an uncovered boundary;
- whether the next target should port more SGLang code or repair a mini-owned
  implementation bug.

If the written contract reveals that the current mini runtime is too fragmented,
stop and propose a broader refactor target rather than adding another local
patch.

## Success Criteria

Minimum:

```text
MTP target-verify runtime contract written
SGLang parity table written
current branch conflicts mapped to contract violations
one correctness-first unified mode implemented or a precise implementation plan
written
bs=1/2/4/5 remains exact
accepted commit remains enabled
```

Full:

```text
bs=1/2/4/5/6 exact
bs=7/8/16 exposure passes or fails only on a newly documented boundary
no new per-batch special case is introduced
TARGET 11.3 go/no-go updated with evidence
```

## Stop Lines

Stop and report if:

- the target cannot write a coherent contract from mini plus SGLang evidence;
- exactness requires disabling accepted commit;
- exactness requires sequentially recomputing accepted rows as the final path;
- a proposed fix regresses `bs=1/2/4/5`;
- another local branch would be required without reducing the branch matrix;
- the work clearly requires a larger SGLang runtime port.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_runtime_contract_unification/README.md
```

Include:

- MTP target-verify runtime contract;
- SGLang parity table;
- branch conflict matrix from TARGET 11.7;
- oracle/unified mode design;
- implementation summary or no-go explanation;
- validation matrix for `bs=1/2/4/5/6`;
- exposure matrix for `bs=7/8/16`;
- recommendation for TARGET 11.3 or the next correctness target.
