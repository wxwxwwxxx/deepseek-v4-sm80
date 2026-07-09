# TARGET 11.1: DSV4 SM80 MTP Weight Loader And One-Step Oracle

## Goal

Prove that mini-sglang can load the DeepSeek V4 MTP weights and run a minimal
one-step MTP forward path without changing the scheduler or serving runtime.

This target is intentionally narrow.  Do not implement speculative decoding
yet.

## Background

DeepSeek-V4-Flash contains MTP weights:

```text
mtp.0.attn.*
mtp.0.ffn.*
mtp.0.e_proj.*
mtp.0.h_proj.*
mtp.0.enorm.weight
mtp.0.hnorm.weight
mtp.0.head.*
mtp.0.norm.weight
mtp.0.emb.tok_emb
```

Mini currently ignores them in the weight loader:

```text
python/minisgl/models/weight.py
```

SGLang's DeepSeek V4 model has the reference mapping:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
```

Relevant SGLang concepts:

- `num_nextn_predict_layers == 1`;
- `is_nextn=True` loading path;
- `remap_weight_name_to_dpsk_hf_format(..., is_nextn=True, ...)`;
- nextn layer id usually equals `config.num_hidden_layers`;
- MTP-specific out-of-layer weights include shared MTP embedding/head/norm and
  `e_proj` / `h_proj` pieces.

## Required Investigation

1. Re-scan `/models/DeepSeek-V4-Flash` and record the exact MTP tensor names,
   shapes, dtypes, and total bytes.
2. Compare SGLang's MTP remap/load behavior against mini's DSV4 layer/module
   names.
3. Identify which mini modules can be reused and which MTP-specific wrappers are
   needed.
4. Determine the minimal hidden-state interface needed from the target model.
   The likely input to MTP is the previous target hidden plus the last accepted
   token, but verify this against SGLang source.
5. Decide whether MTP should share target `embed_tokens` / `lm_head` or use the
   checkpoint's MTP shared head/embedding path.  Prefer SGLang parity.

## Implementation Scope

Allowed:

- Add an opt-in MTP weight-loading path.
- Add small MTP module classes or wrappers if mini's existing DSV4 modules
  cannot be reused directly.
- Expose target hidden states for oracle/debug use behind an opt-in or test-only
  path.
- Add a standalone script or test that runs one target forward, captures the
  needed hidden state, and runs one MTP step.
- Add focused unit tests for mapping, shapes, dtype, and no-NaN output.

Not allowed:

- Do not modify decode scheduler semantics.
- Do not implement token acceptance or target verification.
- Do not add CUDA graph capture for MTP.
- Do not promote any default flag.

## Suggested Opt-In Surface

Use a clearly named experimental flag or env var, for example:

```text
--enable-dsv4-mtp
MINISGL_DSV4_EXPERIMENTAL_MTP=1
```

If an existing config/feature flag pattern is more local, use that instead.

## Oracle Requirements

The first oracle can be source-derived if SGLang cannot be run in the current
environment, but it must be explicit about what is proven.

Minimum pass criteria:

- MTP weights load without missing/unexpected critical tensors.
- The one-step MTP path produces finite logits for several short prompts.
- Output shapes match vocab/hidden expectations.
- MTP memory overhead is reported per rank.
- Text/logit smoke has no obvious corruption.

Preferred pass criteria:

- Compare mini's one-step MTP output against a SGLang-derived or direct SGLang
  run for the same prompt/hidden/token if practical.
- Record top-k overlap or max error for logits when a reference exists.

## Benchmarks

This target is not a throughput target.  Only run lightweight checks:

- single-rank shape/load tests if possible;
- TP8 load smoke only if needed to prove sharded mapping;
- a few prompts with `page_size=256`;
- memory before/after MTP load.

## Stop Lines

Stop and report if any of these happen:

- MTP tensor mapping cannot be made unambiguous from SGLang source.
- The MTP path requires broad scheduler/runtime changes just to run one step.
- MTP logits are NaN/Inf or obviously corrupt.
- Extra persistent MTP memory is too high to be plausible for the current GPU
  budget.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_weight_oracle/README.md
```

Include:

- exact MTP tensor census;
- mini/SGLang mapping table;
- memory ledger;
- tests added or commands run;
- one-step oracle result;
- recommendation: proceed to TARGET 11.2 or stop.

