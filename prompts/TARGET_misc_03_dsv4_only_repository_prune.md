# TARGET misc 03: DSV4-Only Repository Prune

## Status

Planned after the two-path runtime cleanup passes.

## Goal

Remove model families and implementation code not required to run DeepSeek V4
Flash through the retained optimized and fallback modes.

This is a dependency-directed prune, not a filename-directed prune.

## Required Inputs

```text
prompts/TARGET_misc_dsv4_release_cleanup.md
prompts/TARGET_misc_01_dsv4_two_path_census.md
prompts/TARGET_misc_02_dsv4_two_path_runtime_cleanup.md
performance_milestones/misc_release_two_path_census/model_dependency_manifest.json
performance_milestones/misc_release_two_path_runtime_cleanup/README.md
```

Begin from a checkpoint commit after TARGET misc 02.

## Required Work

### 1. Reduce The Model Registry

The final model registry supports only the architecture(s) actually emitted by
the DeepSeek V4 Flash checkpoint, centered on:

```text
DeepseekV4ForCausalLM
```

Delete unsupported model implementations and registrations after confirming
they are not imported by DSV4:

```text
python/minisgl/models/llama.py
python/minisgl/models/mistral.py
python/minisgl/models/qwen2.py
python/minisgl/models/qwen3.py
python/minisgl/models/qwen3_moe.py
```

Remove Llama/Qwen/Mistral fallback defaults from config parsing.  Unsupported
architectures must fail early with a concise message stating that this release
supports DeepSeek V4 Flash only.

### 2. Prune Unreachable Model Infrastructure

Use the census manifest plus import/call-site checks to review:

- generic RoPE attention and gated MLP helpers;
- generic MHA attention backend code;
- FlashAttention/FlashInfer/TensorRT-LLM attention backend adapters;
- generic MHA/naive cache pools;
- generic fused MoE backends;
- non-DSV4 rotary/config branches;
- unsupported model weight remapping logic.

Delete only code unreachable from DSV4 optimized/fallback, sampling, scheduler,
server, tokenizer, or benchmarks.  FlashInfer sampling is currently used by
the retained sampler and is not removable merely because generic FlashInfer
attention is removed.

Keep reusable serving primitives that DSV4 actually imports, even if their
names are generic:

- `BaseOP`, model/weight loading primitives, embeddings, and output head;
- scheduler, request, message, tokenizer, server, and OpenAI API machinery;
- tensor parallel communication and PyNCCL;
- radix/component/SWA cache infrastructure;
- sampling implementation and its dependencies;
- CUDA graph and graph-memory planning;
- DSV4-required JIT/CUDA/vendor sources.

### 3. Prune Packaging Dependencies

For each `pyproject.toml` dependency, provide a retained import owner.  Remove a
dependency only after a clean-environment import/test proves it is unnecessary.

Do not guess based on backend names.  Examples:

- `flashinfer-python` may remain required for sampling;
- `sgl_kernel`, TVM FFI, quack kernels, and compiler dependencies must be tied
  to actual retained DSV4 code;
- `modelscope` may remain only if the documented `--model-source modelscope`
  entry is retained and tested;
- dev-only benchmark dependencies belong in optional dependencies.

Update package metadata to describe DSV4-only scope accurately.

### 4. Prune Tests And Documentation References

Delete tests that exclusively instantiate removed model classes.  Rewrite
generic scheduler/config/registry tests with local fake DSV4 config/tokenizer
fixtures where their serving behavior remains important.

Remove unsupported model claims from:

```text
README.md
docs/features.md
docs/structures.md
comments/help text/package metadata
```

TARGET misc 04 owns the final README examples, but this target must prevent
imports/tests/docs from referring to files already deleted.

### 5. Remove Generated Repository Debris

Remove repository-local generated artifacts from the working tree where safe:

```text
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
htmlcov/
```

Do not use a destructive global `git clean`.  Preserve user files and ignored
performance reports.

## Validation

Minimum static gates:

```bash
python -m compileall -q python/minisgl
python -c "import minisgl; from minisgl.llm import LLM"
python -m minisgl --help
python -m minisgl.shell --help
```

Required source checks:

- no removed model registry entries;
- no import of removed model/backend modules;
- no README/docs claim of Qwen/Llama/Mistral support;
- every packaged native source has a retained owner;
- every required dependency has an import/runtime owner.

Required tests:

```bash
python -m pytest -q
```

Required GPU gates:

- default optimized DSV4 text smoke;
- explicit fallback text smoke;
- one short prefix/cache smoke;
- one short balanced CUDA graph macro.

## Deliverables

```text
performance_milestones/misc_release_dsv4_only_prune/README.md
performance_milestones/misc_release_dsv4_only_prune/deleted_files.txt
performance_milestones/misc_release_dsv4_only_prune/retained_dependency_owners.md
performance_milestones/misc_release_dsv4_only_prune/package_size_before_after.json
```

Record Python source LOC and wheel/package size before and after pruning.  These
are clarity metrics, not targets that justify unsafe deletion.

## Stop Conditions

Stop and repair before continuing if:

- DSV4 imports a deleted general-purpose module;
- server/scheduler/sampling tests regress;
- optimized or fallback DSV4 smoke fails;
- a dependency is removed without a clean-environment proof;
- an unsupported architecture falls through to a misleading default instead
  of failing clearly.

Do not retain a removed model just to satisfy an outdated README example.

## Completion Criteria

- Only DeepSeek V4 architecture is registered.
- Unsupported model implementations are deleted.
- Unreachable generic backends/caches/layers/native sources are deleted.
- Retained generic infrastructure has a documented DSV4/public-entry owner.
- Package dependencies are justified.
- Full tests and short optimized/fallback GPU gates pass.
- A checkpoint commit is recommended before TARGET misc 04.
