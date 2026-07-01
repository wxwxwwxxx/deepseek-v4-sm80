#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_vllm_gap}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC="${NPROC:-8}"
MAX_EXTEND_TOKENS="${MAX_EXTEND_TOKENS:-}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-0}"
NSYS_MEMORY_RATIO="${NSYS_MEMORY_RATIO:-0.8}"
ALLOW_PROFILE_FAILURE="${ALLOW_PROFILE_FAILURE:-1}"
DRY_RUN="${DRY_RUN:-0}"

mode_suffix="${MODE_SUFFIX:-default_prefill}"
if [[ -n "${MAX_EXTEND_TOKENS}" ]]; then
  mode_suffix="max_extend_${MAX_EXTEND_TOKENS}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07_nsys_mini_v1_4096x128_bs4_${mode_suffix}_warmup1}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_target07_mini_v1_4096x128_bs4_${mode_suffix}_warmup1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NSYS_BASE
export NSYS_PROFILE_RANKS

cmd=(
  "${TORCHRUN_BIN}" --standalone "--nproc_per_node=${NPROC}" --no-python
  "${MILESTONE_DIR}/scripts/nsys_rank_wrapper.sh"
  "${PYTHON_BIN}"
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --variants v1_moe
  --scenarios decode_throughput_bs8
  --prompt-len 4096
  --decode-len 128
  --batch-size 4
  --repeats 1
  --warmup-repeats 1
  --page-size 256
  --memory-ratio "${NSYS_MEMORY_RATIO}"
  --output-dir "${OUTPUT_DIR}"
  --keep-going
)

if [[ -n "${MAX_EXTEND_TOKENS}" ]]; then
  cmd+=(--max-extend-tokens "${MAX_EXTEND_TOKENS}")
fi
cmd+=("$@")

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
  echo "warning: mini nsys workload exited with status ${run_status}" >&2
  echo "warning: continuing to export any generated nsys reports" >&2
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
  python "${MILESTONE_DIR}/scripts/summarize_nsys_sqlite.py" \
    "${sqlite}" \
    --output-json "${MILESTONE_DIR}/summaries/$(basename "${sqlite%.sqlite}").json" \
    --output-md "${MILESTONE_DIR}/summaries/$(basename "${sqlite%.sqlite}").md" \
    --nvtx-window "repeat:decode_throughput_bs8:0" || true
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  summary_dir="${MILESTONE_DIR}/summaries/$(basename "${OUTPUT_DIR}")"
  rm -rf "${summary_dir}"
  mkdir -p "${summary_dir}"
  for name in run_config.json summary.json matrix.jsonl; do
    if [[ -f "${OUTPUT_DIR}/${name}" ]]; then
      cp "${OUTPUT_DIR}/${name}" "${summary_dir}/${name}"
    fi
  done
fi

echo "mini fair nsys output: ${OUTPUT_DIR}"
echo "mini fair nsys base: ${NSYS_BASE}"
echo "mini fair nsys profiled ranks: ${NSYS_PROFILE_RANKS}"
echo "mini fair nsys memory ratio: ${NSYS_MEMORY_RATIO}"

if [[ "${run_status}" != "0" && "${ALLOW_PROFILE_FAILURE}" != "1" ]]; then
  exit "${run_status}"
fi
