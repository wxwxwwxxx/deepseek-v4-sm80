# TARGET 08.20 Go/No-Go

**Decision: GO with guardrails.**

The sustained 08.10 retained state uses 56/128 pages, and eviction pressure reaches 112/128 pages.  That is above the 20%-30% investigation threshold and removes multiple 4096+1024 request equivalents from the fixed page pool.

Guardrails: resolve the generated-token/logit correctness follow-up before any default promotion, and keep 08.20 focused on independent SWA/component retention rather than new low-precision, graph, or global allocator work.
