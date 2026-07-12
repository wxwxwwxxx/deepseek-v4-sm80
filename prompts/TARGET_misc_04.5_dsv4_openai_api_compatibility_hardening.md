# TARGET misc 04.5: DSV4 OpenAI API Compatibility Hardening

## Status

Planned release-interface hardening after TARGET misc 04 and before the final
TARGET misc 05 soak.

## Goal

Make the text-only DeepSeek V4 serving API work predictably with common
OpenAI-compatible clients without changing the model execution path.

The central contract is:

```text
Be liberal about harmless transport/schema variants.
Never silently accept a generation feature that minisgl does not implement.
Return standard response shapes and actionable errors.
Keep all work in the frontend/API boundary unless a tiny CPU-only metadata
propagation is required for truthful token usage.
```

This is not a feature-expansion or performance target.  Do not modify DSV4
kernels, attention/MoE backends, CUDA graphs, cache ownership, communication,
precision, scheduling policy, or batching behavior.

## Starting Evidence

Read first:

```text
python/minisgl/server/api_server.py
python/minisgl/server/args.py
python/minisgl/tokenizer/tokenize.py
python/minisgl/tokenizer/detokenize.py
python/minisgl/message/{frontend,tokenizer,backend}.py
python/minisgl/core.py
tests/server/test_openai_compat.py
README.md
prompts/TARGET_misc_04_dsv4_benchmark_readme_surface.md
```

The first compatibility defect has already been identified and fixed in the
working tree:

- vLLM 0.21 `openai-chat` sends text as
  `content: [{"type": "text", "text": "..."}]`;
- it uses `max_completion_tokens` rather than only `max_tokens`;
- the old minisgl schema rejected the content-parts payload with HTTP 422 and
  ignored `max_completion_tokens`, falling back to 16 output tokens.

Preserve support for both legacy string content and OpenAI text content parts.
Preserve both output-length parameter spellings with a documented precedence.

Current remaining risks include:

- request fields `stop`, `n`, `presence_penalty`, and `frequency_penalty` are
  declared but do not affect generation;
- unsupported semantic fields may be silently ignored by Pydantic;
- streaming chunks report `object: text_completion.chunk` instead of
  `chat.completion.chunk` and omit stable `created`/`model` fields;
- `stream_options.include_usage` is ignored;
- non-streaming usage is fabricated as all zeros;
- `/v1/models` exposes the filesystem path rather than a clear served-model
  identity;
- invalid request combinations and backend errors do not consistently use an
  OpenAI-style error envelope;
- only `system`, `user`, and `assistant` roles are accepted even though the
  bundled DSV4 formatter also understands `developer`.

## Compatibility Scope

### Required In This Target

```text
POST /v1/chat/completions
GET  /v1/models
text-only messages
streaming and non-streaming responses
OpenAI Python SDK
vLLM bench serve --backend openai-chat
common HTTP clients using OpenAI-compatible JSON/SSE
```

### Explicitly Unsupported In This Release

```text
Responses API
embeddings/audio/image APIs
multimodal message parts
tool/function execution and tool-result conversations
structured output / JSON schema guarantees
logprobs
multiple choices (n > 1)
model-side speculative decoding metrics
```

Unsupported features must fail clearly if requested.  They must not be
accepted and then ignored.

`/metrics` is not part of the OpenAI API and is not required here.  vLLM bench
probes it before and after a run only for speculative-decoding Prometheus
metrics; a 404 is non-fatal.  Do not add a fake metrics endpoint.  A real
low-overhead Prometheus surface may be a separate observability target.

## Required Work

### 1. Define One Request Contract

Refactor the API models only as much as needed to express a truthful contract.

Required behavior:

- require `model` and a non-empty `messages` list for the standard chat route;
- retain string content and arrays of text content parts;
- accept the `developer` role and pass it through to the existing DSV4
  formatter; retain `system`, `user`, and `assistant`;
- reject image/audio/file content parts with a structured unsupported-feature
  error rather than an opaque Pydantic traceback;
- accept `max_tokens` and `max_completion_tokens`; document and test precedence
  when both are present;
- accept `stop` in OpenAI-compatible forms (`string`, `list[string]`, or null),
  but reject non-empty stop requests until stop semantics are genuinely
  implemented;
- accept `n=1`; reject `n != 1` clearly;
- accept zero/default presence and frequency penalties; reject nonzero values
  until the sampler implements them;
- continue supporting minisgl extensions already used by benchmarks, including
  `top_k` and `ignore_eos`;
- accept harmless transport metadata such as `stream_options` without changing
  generation;
- explicitly recognize and reject known unsupported semantic fields such as
  `tools`, `tool_choice`, `response_format`, `logprobs`, `top_logprobs`, and
  multimodal options when they request behavior minisgl cannot provide.

Choose a deliberate policy for unknown extra fields.  Prefer a small
allow-list of harmless client metadata plus structured rejection of unknown
generation-affecting options.  Do not accidentally turn a broad
`extra="ignore"` default into a claim of support.

Validation failures should use HTTP 400 where practical and an OpenAI-style
body:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "...",
    "code": "..."
  }
}
```

Do not use `assert` for user-controlled request validation.

### 2. Standardize Chat Responses

For non-streaming responses, ensure:

```text
id: stable chatcmpl-* identifier
object: chat.completion
created: Unix timestamp
model: served/request model identity
choices[0].index: 0
choices[0].message.role: assistant
choices[0].message.content: generated text
choices[0].finish_reason: standard mapped value
```

For streaming responses, ensure every normal chunk has:

```text
same id for the request
same created timestamp
same model identity
object: chat.completion.chunk
choices[0].delta
choices[0].index: 0
choices[0].finish_reason
```

Keep correct SSE framing (`data: <json>\n\n` and final
`data: [DONE]\n\n`).  Do not emit backend errors inside `delta` as though they
were assistant content.  Map internal finish reasons to the closest standard
values such as `stop` or `length`, while preserving actionable error details in
an error response/log.

### 3. Make Usage Truthful

Never return fabricated zero usage for a successful non-empty request.

Preferred implementation:

- obtain prompt token count from the existing tokenizer result;
- obtain completion token count from the existing detokenizer decode state or
  final scheduler request state;
- propagate only small CPU integers through existing messages;
- never re-tokenize generated text on the API event loop;
- never call CUDA synchronization or inspect GPU tensors for usage;
- when `stream_options.include_usage=true`, emit a final standard usage chunk;
- return the same exact counts in non-streaming responses.

If exact counts require a disproportionately broad scheduler/message rewrite,
the acceptable first-release fallback is to omit `usage` consistently and
document that limitation.  Omission is preferable to false zeros.  Record the
reason and a bounded follow-up.  Do not block vLLM bench: it can retokenize the
generated text when usage is absent.

### 4. Establish A Served Model Identity

Add a low-cost public served-model-name contract, preferably:

```bash
--served-model-name deepseek-v4-flash
```

Requirements:

- default deterministically from the model path basename when not specified;
- `/v1/models` returns an ID that can be passed back to chat completions;
- completion and streaming responses report that same identity;
- preserve compatibility with callers that currently send the configured
  model path or an explicit alias unless strict validation is intentionally
  enabled and documented;
- do not couple model identity to weight loading or kernel dispatch.

Avoid adding a general multi-model registry.  This release serves one DSV4
model instance.

### 5. Add Lightweight Operational Endpoints Only If Cheap

`/health` may report that the HTTP process is alive.  `/ready` may report that
the frontend/backend initialization has completed.  Add them only if readiness
can be derived from existing state without polling or synchronization.

Do not add Prometheus dependencies or GPU queries in this target.

### 6. Tests Before Full-Model Work

Build no-weight tests around the exact wire contract.  Prefer FastAPI
`TestClient`/ASGI tests with a small fake frontend manager over tests that only
instantiate a Pydantic model.

Required cases:

```text
legacy string content
OpenAI text content parts
vLLM 0.21 openai-chat payload
max_tokens and max_completion_tokens precedence
developer role
stop null/empty and non-empty rejection
n=1 and n>1 rejection
zero and nonzero penalties
known unsupported semantic fields
unknown-field policy
non-stream response schema
stream chunk schema and SSE termination
stream_options.include_usage behavior
served model listing/identity
structured invalid-request errors
client disconnect/abort remains functional
```

Exercise both sync and async OpenAI Python clients if this can be done against
the fake/no-weight server.  Add a fixture payload copied structurally from the
vLLM benchmark rather than importing vLLM as a test dependency.

## Integration Validation

After no-weight tests pass, start the actual DSV4 TP8 server once and run a
small matrix.  Reuse the already loaded engine rather than repeatedly paying
model startup cost.

Required clients:

1. OpenAI Python SDK, streaming and non-streaming;
2. `benchmark/online/bench_simple.py` with a very small request count;
3. vLLM 0.21 `bench serve --backend openai-chat` with approximately:

```text
random input:   128-512 tokens
random output:  16-32 tokens
num prompts:    2-4
request rate:   low
ignore EOS:     enabled when validating a fixed output length
```

For the vLLM benchmark, confirm:

- no 422 responses;
- requested output limit reaches `SamplingParams`;
- all requests succeed;
- generated token accounting is nonzero and plausible;
- `/metrics` 404 probes are harmless and are not treated as failures.

Run one multilingual text sanity request and verify no乱码/NaN/Inf.

## Performance Safety

This target should have no measurable model throughput effect.

Required code-review evidence:

- no kernel, model-forward, attention, MoE, graph, cache, communication, or
  scheduler-policy edits;
- no CUDA calls in request parsing, response formatting, or usage collection;
- no generated-text re-tokenization on the API event loop;
- no per-token logging added;
- only bounded CPU dictionary construction and small integer propagation in
  the streaming path.

A short TP8 smoke is sufficient.  Do not rerun the full historical performance
matrix unless the implementation touches a runtime hot path contrary to the
target boundary.

## Documentation

Update README's API section with a concise supported surface:

```text
text-only /v1/chat/completions
streaming/non-streaming
supported sampling fields
minisgl extension fields
explicitly unsupported features
served model name behavior
dummy API key behavior, if authentication remains disabled
```

Do not claim full OpenAI API parity.  Use wording such as
`OpenAI-compatible text chat endpoint` and link unsupported requests to clear
runtime errors.

## Deliverables

```text
performance_milestones/misc_openai_api_compatibility_hardening/README.md
performance_milestones/misc_openai_api_compatibility_hardening/compatibility_matrix.md
performance_milestones/misc_openai_api_compatibility_hardening/request_fixtures/
```

The compatibility matrix must distinguish:

```text
SUPPORTED
SUPPORTED AS NO-OP TRANSPORT METADATA
REJECTED WITH ACTIONABLE ERROR
NOT IMPLEMENTED
```

Record exact versions of the OpenAI Python SDK and vLLM benchmark client used.

## Stop Conditions

Stop and report a blocker rather than widening the target if:

- truthful usage requires GPU synchronization or invasive scheduler changes;
- tool calling, structured output, or multimodal support becomes necessary to
  pass a basic text-client test;
- a proposed compatibility shim changes tokenization or sampling semantics;
- model throughput or CUDA graph behavior changes;
- broad exception handling hides backend failures;
- unsupported generation parameters remain silently ignored.

Do not spend time implementing `/metrics`, Responses API, tool execution, or
multimodal serving in this target.

## Completion Criteria

- OpenAI SDK and vLLM chat benchmark payloads complete successfully.
- Text content strings and text parts produce the same tokenizer input.
- Streaming chunks use the standard chat-completion schema and SSE framing.
- Non-streaming responses use the standard chat-completion schema.
- Model listing and response identity are coherent.
- No declared semantic parameter is silently ignored.
- Usage is exact or honestly omitted, never fabricated.
- Unsupported capabilities return actionable errors.
- No model execution or GPU performance path is changed.
- README describes the supported compatibility surface truthfully.
