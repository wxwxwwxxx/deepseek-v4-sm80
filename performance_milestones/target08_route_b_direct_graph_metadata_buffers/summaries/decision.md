# Decision

Decision: **keep_experimental**.

Reasons:

- Correctness, text smoke, graph replay, prefix-hit sanity, and eviction-pressure
  sanity pass.
- C4-first direct generation succeeds and removes the largest C4 eager
  materialization/copy bytes.
- Expanded SWA+C4+C128 direct generation removes the intended source/copy bytes,
  but only reduces large-wave decode prepare by 13.43%.
- Expanded direct output throughput is 0.757x phase1 prefix-on and 0.955x Route B
  graph baseline on the large-wave scenario.
- Owner timing shows the targeted index source/copy bytes are no longer the main
  explanation for large-wave performance.  Further work likely needs stable-row
  lifetime tracking or compute-side changes, which is outside this milestone.

Do not promote this path by default.
