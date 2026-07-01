#!/usr/bin/env bash
set -euo pipefail

NSYS_BASE="${NSYS_BASE:?NSYS_BASE must be set by the caller}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
NSYS_BACKTRACE="${NSYS_BACKTRACE:-none}"
NSYS_CUDABACKTRACE="${NSYS_CUDABACKTRACE:-none}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-}"

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

profile_args=(
  profile
  -t "${NSYS_TRACE}"
  --sample="${NSYS_SAMPLE}" \
  --cpuctxsw="${NSYS_CPUCTXSW}" \
  --backtrace="${NSYS_BACKTRACE}" \
  --cudabacktrace="${NSYS_CUDABACKTRACE}" \
  --force-overwrite=true \
  -o "${NSYS_BASE}_rank${rank}"
)
if [[ -n "${NSYS_CUDA_GRAPH_TRACE}" ]]; then
  profile_args+=(--cuda-graph-trace="${NSYS_CUDA_GRAPH_TRACE}")
fi

exec nsys "${profile_args[@]}" "$@"
