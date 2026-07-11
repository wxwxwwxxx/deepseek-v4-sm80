# TARGET misc 04: DSV4 Benchmark And README Surface

## Status

Planned after the DSV4-only repository prune.

## Goal

Make the public repository truthful and immediately usable for DeepSeek V4
Flash while moving development benchmark clutter out of the public benchmark
surface.

The release promises entry-point usability for DSV4.  It does not promise that
old README model names remain supported.

## Public Benchmark Files

Keep and make runnable:

```text
benchmark/offline/bench.py
benchmark/offline/bench_wildchat.py
benchmark/online/bench_qwen.py
benchmark/online/bench_simple.py
```

The filename `bench_qwen.py` may remain for compatibility.  Its Qwen trace is a
request workload, not a claim that the engine serves a Qwen model.

Each script must have argparse-based help and defaults centered on:

```text
model: /models/DeepSeek-V4-Flash
page size: 256 where applicable
TP: 8 for the validated offline/full-model path
server: http://127.0.0.1:1919/v1 for online clients
```

Allow CLI overrides.  Avoid hard-coded user-specific output directories.

## Required Work

### 1. Offline Bench Entry

Update `benchmark/offline/bench.py` to:

- accept model path, TP size, request count, input/output ranges, seed, recipe,
  and output/report arguments;
- derive rank/world size from `torchrun` and construct `DistributedInfo`;
- fail clearly when requested TP and `WORLD_SIZE` disagree;
- default to DSV4 optimized/balanced behavior without env vars;
- print results only on the primary rank;
- use valid token ids for the loaded tokenizer/model;
- keep a short warmup and machine-readable summary.

The documented default invocation should use `torchrun --nproc_per_node=8`.

### 2. WildChat Offline Entry

Update `benchmark/offline/bench_wildchat.py` with the same DSV4/TP-aware launch
contract.  Add arguments for request count, languages, output length, dataset
cache path, and model path.

Do not download dataset shards into the tracked source directory by default.
Use an explicit cache directory or a standard user cache.

### 3. Online Entries

Update `benchmark/online/bench_qwen.py` and `bench_simple.py` to:

- accept base URL, expected model, request count, concurrency/batch settings,
  seed, and output path;
- default the expected model to `/models/DeepSeek-V4-Flash` while accepting the
  server's `/v1/models` identity;
- report a clear mismatch if the connected server exposes another model;
- keep workload generation tokenizer-compatible with DSV4;
- avoid implying Qwen model support in logs/help/README;
- produce nonzero exit status for connection or request failures.

### 4. Move Development DSV4 Scripts

Use `git mv` to move development and microbenchmark scripts from
`benchmark/offline/` to a clear debug hierarchy, for example:

```text
debug/dsv4/benchmark/offline/
```

Move all `deepseek_v4_*` scripts and `dsv4_graph_reserve_lifecycle.py` unless
the public benchmark manifest explicitly retained one.  Preserve imports and
relative paths needed to run them from the repository root.

Add `debug/dsv4/README.md` stating:

- these scripts are developer tools, not supported public benchmark APIs;
- the primary public benchmarks are the four retained files;
- many scripts assume DGX A100 TP8 and `/models/DeepSeek-V4-Flash`;
- archived TARGET reports provide historical context.

Remove generated `__pycache__` content.  Remove the current source-less
`debug/mtp` directory from the release branch; MTP history remains on
`dsv4-mtp-paused-reference` and in archived prompts.

### 5. Rewrite README Public Contract

Rewrite README around the actual release:

```text
distribution: minisgl==0.1.0+dsv4.sm80
release name: Mini-SGLang 0.1.0, DSV4 on SM80
DeepSeek V4 Flash only
A100/sm80 TP8 validated
optimized/balanced default
fallback/oracle opt-in
page size 256 resolved automatically
radix prefix cache, SWA independent lifecycle, chunked prefill, CUDA graph,
Marlin WNA16 MoE, and PyNCCL as tested release features
MTP not included
```

Explain that `v0.0.0` is the historical performance baseline and that the
qualified cleaned release uses the proposed `v0.1.0-dsv4-sm80` git tag.  Do
not describe `0.1.0+dsv4.sm80` as a PyPI publication version; it is a PEP 440
local version for source/direct-wheel/private-index distribution.

Replace all Qwen/Llama/Mistral serving examples with tested DSV4 commands.

README must include tested examples for:

1. installation/import;
2. `python -m minisgl` OpenAI-compatible server;
3. one `curl` or OpenAI client request;
4. `python -m minisgl.shell` interactive entry;
5. Python `LLM` entry under the actual TP launch contract;
6. offline benchmark;
7. online simple/trace benchmark;
8. explicit fallback/oracle launch;
9. optional public recipes and their capacity/performance intent.

Do not claim entry-point success based only on `--help`.  TARGET misc 05 must
exercise every documented command shape.

Update `docs/features.md` and `docs/structures.md` so they do not contradict
README.  Keep architecture explanations that remain true for DSV4.

### 6. Add Lightweight Benchmark Tests

Add tests that do not load full weights:

- every benchmark `--help` exits zero;
- argument parsing uses DSV4 defaults;
- offline TP/world-size mismatch fails clearly;
- online model mismatch is reported;
- moved debug scripts remain importable or syntactically valid where expected;
- README-referenced paths exist.

## Validation

Cheap gates:

```bash
python benchmark/offline/bench.py --help
python benchmark/offline/bench_wildchat.py --help
python benchmark/online/bench_qwen.py --help
python benchmark/online/bench_simple.py --help
python -m pytest -q tests/benchmark tests/server
python -m compileall -q benchmark debug/dsv4 python/minisgl
```

GPU/integration gates:

- short TP8 offline `bench.py` run;
- small WildChat run using an already downloaded shard or explicit skip reason;
- launch DSV4 server, query `/v1/models`, send a completion/chat request;
- run `bench_simple.py` against that server;
- run a small `bench_qwen.py` trace against that server;
- exercise shell startup and clean exit noninteractively if possible.

## Deliverables

```text
performance_milestones/misc_release_benchmark_readme_surface/README.md
performance_milestones/misc_release_benchmark_readme_surface/command_matrix.md
performance_milestones/misc_release_benchmark_readme_surface/moved_files.txt
```

The command matrix must map every README command to its exact validation result.

## Stop Conditions

Stop before final soak if:

- any README command is untested or known broken;
- benchmark defaults still name an unsupported model;
- an online script silently benchmarks a different server model;
- DSV4 development scripts remain mixed into the public offline benchmark
  directory without a documented reason;
- moved scripts are imported by production code.

## Completion Criteria

- Four requested public benchmark files run with DSV4 defaults.
- DSV4 development/microbench scripts live under `debug/`.
- README/docs only promise DSV4 model support.
- README's Python, CLI, server, shell, offline, and online entry shapes are
  represented in the command matrix.
- Cheap and short integration gates pass.
