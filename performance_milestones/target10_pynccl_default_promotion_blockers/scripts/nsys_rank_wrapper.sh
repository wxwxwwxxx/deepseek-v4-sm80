#!/usr/bin/env bash
set -euo pipefail

NSYS_BASE="${NSYS_BASE:?NSYS_BASE must be set by the caller}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
NSYS_BACKTRACE="${NSYS_BACKTRACE:-none}"
NSYS_CUDABACKTRACE="${NSYS_CUDABACKTRACE:-none}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-graph}"
NSYS_TRACE_FORK_BEFORE_EXEC="${NSYS_TRACE_FORK_BEFORE_EXEC:-true}"

rank="${LOCAL_RANK:-${RANK:-0}}"

should_profile=0
if [[ "${NSYS_PROFILE_RANKS}" == "all" ]]; then
  should_profile=1
else
  IFS=',' read -ra ranks <<< "${NSYS_PROFILE_RANKS}"
  for item in "${ranks[@]}"; do
    if [[ "${item}" == "${rank}" ]]; then
      should_profile=1
      break
    fi
  done
fi

if [[ "${should_profile}" != "1" ]]; then
  exec "$@"
fi

exec nsys profile \
  -t "${NSYS_TRACE}" \
  --sample="${NSYS_SAMPLE}" \
  --cpuctxsw="${NSYS_CPUCTXSW}" \
  --backtrace="${NSYS_BACKTRACE}" \
  --cudabacktrace="${NSYS_CUDABACKTRACE}" \
  --cuda-graph-trace="${NSYS_CUDA_GRAPH_TRACE}" \
  --trace-fork-before-exec="${NSYS_TRACE_FORK_BEFORE_EXEC}" \
  --force-overwrite=true \
  -o "${NSYS_BASE}_rank${rank}" \
  "$@"
