#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target10_pynccl_default_promotion_blockers}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC="${NPROC:-8}"
SCENARIOS="${SCENARIOS:-serving_mixed_112req_wave16}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
SEED="${SEED:-20260711}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target1027_nsys_pynccl_full_model_${SCENARIOS}}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_target1027_pynccl_full_model_${SCENARIOS}}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-0}"
ALLOW_PROFILE_FAILURE="${ALLOW_PROFILE_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MINISGL_DSV4_SM80_MOE_REDUCE_BF16="${MINISGL_DSV4_SM80_MOE_REDUCE_BF16:-1}"
export MINISGL_PYNCCL_MAX_BUFFER_SIZE="${MINISGL_PYNCCL_MAX_BUFFER_SIZE:-32M}"
export NSYS_BASE
export NSYS_PROFILE_RANKS

cmd=(
  "${TORCHRUN_BIN}" --standalone "--nproc_per_node=${NPROC}" --no-python
  "${MILESTONE_DIR}/scripts/nsys_rank_wrapper.sh"
  "${PYTHON_BIN}"
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
  --scenarios ${SCENARIOS}
  --page-size 256
  --num-pages 128
  --enable-dsv4-radix-prefix-cache
  --enable-dsv4-component-loc-ownership
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --repeats "${REPEATS}"
  --warmup-repeats "${WARMUP_REPEATS}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
  --use-pynccl
  --keep-going
)

cd "${MINISGL_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

run_status=0
"${cmd[@]}" || run_status=$?

if [[ "${run_status}" != "0" ]]; then
  echo "warning: full-model nsys workload exited with status ${run_status}" >&2
fi

for rep in "${NSYS_BASE}"_rank*.nsys-rep; do
  if [[ -f "${rep}" ]]; then
    sqlite="${rep%.nsys-rep}.sqlite"
    nsys export --type sqlite --force-overwrite=true \
      --output "${sqlite}" \
      "${rep}" || true
  fi
done

mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"
for artifact in "${OUTPUT_DIR}" "${NSYS_BASE}"_rank*.nsys-rep "${NSYS_BASE}"_rank*.sqlite; do
  if [[ -e "${artifact}" ]]; then
    rm -rf "${MILESTONE_DIR}/raw/$(basename "${artifact}")"
    ln -s "${artifact}" "${MILESTONE_DIR}/raw/$(basename "${artifact}")"
  fi
done

for sqlite in "${NSYS_BASE}"_rank*.sqlite; do
  [[ -f "${sqlite}" ]] || continue
  python "${MINISGL_ROOT}/performance_milestones/target07_vllm_gap/scripts/summarize_nsys_sqlite.py" \
    "${sqlite}" \
    --output-json "${MILESTONE_DIR}/summaries/$(basename "${sqlite%.sqlite}").json" \
    --output-md "${MILESTONE_DIR}/summaries/$(basename "${sqlite%.sqlite}").md" \
    --nvtx-window "repeat:${SCENARIOS}:0" \
    --top 30 || true
done

echo "full-model nsys output: ${OUTPUT_DIR}"
echo "full-model nsys base: ${NSYS_BASE}"
echo "full-model nsys profiled ranks: ${NSYS_PROFILE_RANKS}"

if [[ "${run_status}" != "0" && "${ALLOW_PROFILE_FAILURE}" != "1" ]]; then
  exit "${run_status}"
fi
