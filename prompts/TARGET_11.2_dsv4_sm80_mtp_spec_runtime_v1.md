# TARGET 11.2: DSV4 SM80 MTP Speculative Runtime V1

## Goal

Implement a conservative top-k 1 greedy MTP speculative runtime that preserves
exact greedy output through target verification.

Start from TARGET 11.1.  Do not begin this target until MTP weights and a
one-step MTP oracle are healthy.

## Runtime Contract

For greedy decoding:

```text
baseline greedy output == MTP speculative greedy output
```

The MTP draft model may propose tokens, but the target model decides which
tokens are accepted.  Only accepted tokens update request state, token pools,
KV/cache ownership, prefix handles, and final output.

## Reference Design

Use SGLang's frozen-KV MTP as the primary reference:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/spec_info.py
```

Important behaviors:

- draft reads target KV read-only;
- draft owns no independent KV pool;
- draft uses target request-to-token/page metadata;
- draft positions are based on current target sequence state;
- verification uses target forward to accept a prefix of draft tokens;
- top-k 1 first; do not implement tree/top-k > 1 initially.

## Mini Surfaces To Inspect

- `python/minisgl/core.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/scheduler/decode.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`

Current mini decode returns one token per request.  MTP runtime will likely need
a new output path that can return variable accepted token counts per request.

## V1 Scope

Allowed:

- Add an opt-in speculative decoding mode.
- Greedy-only MTP with top-k 1.
- Fixed small draft length, for example 2 or 4, behind a flag.
- Target verification in eager mode first.
- Disable CUDA graph in MTP mode until correctness is proven.
- Update `Req`/`Batch`/`ForwardOutput` or add MTP-specific output structs as
  needed.
- Add acceptance counters and exactness tests.

Not allowed:

- Do not support sampling first.
- Do not support top-k > 1 tree speculative decoding first.
- Do not promote by default.
- Do not mix this with low-precision changes.
- Do not change prefix/SWA ownership contracts without a focused test proving
  the change.

## Suggested Algorithm

For each decode step:

1. Build the normal current target state.
2. Run MTP draft for up to `draft_len` tokens using frozen target KV and the
   previous accepted target hidden state.
3. Allocate target verify slots for the proposed draft tokens.
4. Run the target model on the draft token sequence in a verify/extend-style
   pass.
5. Compare target greedy predictions to draft tokens.
6. Accept the longest matching prefix.
7. Append accepted tokens to each request and update KV/token/page state only
   for accepted tokens.
8. Emit the target bonus token when appropriate, following the SGLang/EAGLE
   verify contract if possible.
9. Fall back to normal one-token decode when no token is accepted or when a
   request is incompatible with MTP.

If mini's current attention/cache path cannot verify multiple tokens cleanly,
write down the blocker and implement the smallest correct verify path before
doing performance work.

## Correctness Tests

Minimum:

- single request, short prompts, greedy exactness;
- batch size 1/2/4, same prompts as baseline, greedy exactness;
- `page_size=256`;
- prefix-cache disabled and enabled if the code path touches prefix state;
- no乱码 text smoke;
- no NaN/Inf logits;
- deterministic acceptance counters.

Add a test that runs baseline greedy and MTP greedy for the same prompts and
compares token ids, not only decoded text.

## Metrics

Record:

- average draft tokens proposed;
- average accepted tokens;
- acceptance histogram;
- target verify calls;
- MTP draft latency;
- target verify latency;
- scheduler/metadata overhead;
- output tok/s for small macro smoke;
- memory overhead.

## Stop Lines

Stop if:

- greedy token ids do not exactly match baseline;
- rejected draft tokens leak into KV/cache/token pools;
- prefix/SWA/component ownership invariants fail;
- average accepted tokens are too low to plausibly amortize draft overhead;
- the implementation requires broad graph or attention rewrites before any
  correctness proof.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_spec_runtime_v1/README.md
```

Include:

- exactness matrix;
- source-parity notes against SGLang;
- acceptance metrics;
- code paths changed;
- recommendation: proceed to TARGET 11.3, keep opt-in, or stop.

