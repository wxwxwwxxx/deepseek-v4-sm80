#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target07_post_0778_roofline_bottleneck_reset}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
REPEATS="${REPEATS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
LONG_DECODE_LEN="${LONG_DECODE_LEN:-1024}"
SHORT_DECODE_LEN="${SHORT_DECODE_LEN:-128}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory}"
RUN_OWNER_TIMING="${RUN_OWNER_TIMING:-1}"
RUN_CAPACITY_PROBE="${RUN_CAPACITY_PROBE:-1}"

mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

run_torchrun() {
  local log_path="$1"
  shift
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TORCHRUN}" --standalone --nproc_per_node="${TP}" "$@"
  ) 2>&1 | tee "${log_path}"
}

capture_baseline() {
  local out_dir="${RAW_DIR}/hardware"
  mkdir -p "${out_dir}"
  nvidia-smi -L > "${out_dir}/nvidia-smi-L.txt"
  nvidia-smi > "${out_dir}/nvidia-smi.txt"
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits > "${out_dir}/nvidia-smi-query.csv"
  git -C "${ROOT}" status --short > "${out_dir}/git-status-short.txt"
  git -C "${ROOT}" rev-parse HEAD > "${out_dir}/git-head.txt"
  (
    cd "${ROOT}"
    python - <<'PY'
import json
import sys
import torch

payload = {
    "python": sys.version.replace("\n", " "),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    payload["device0_name"] = torch.cuda.get_device_name(0)
    payload["device0_capability"] = list(torch.cuda.get_device_capability(0))
try:
    payload["nccl_version"] = ".".join(str(part) for part in torch.cuda.nccl.version())
except Exception as exc:
    payload["nccl_version_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  ) > "${out_dir}/torch-env.json"
}

run_smoke() {
  local out_dir="${RAW_DIR}/smoke_${VARIANT}"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --output "${out_dir}/text_smoke.json" \
    --tensor-parallel-size "${TP}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --max-seq-len 1024 \
    --max-extend-tokens 4096 \
    --max-tokens 64 \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 \
    --fail-on-warning
}

run_macro() {
  local label="$1"
  local decode_len="$2"
  local out_dir="${RAW_DIR}/${label}_victory"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${decode_len}" \
    --batch-size "${BATCH_SIZE}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --output-dir "${out_dir}" \
    --keep-going
}

run_owner_timing() {
  local label="timing_${PROMPT_LEN}x${SHORT_DECODE_LEN}_owner_victory"
  local out_dir="${RAW_DIR}/${label}"
  mkdir -p "${out_dir}"
  (
    export MINISGL_DSV4_OWNER_TIMING=1
    export MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES="${MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES:-60000}"
    run_torchrun "${out_dir}/torchrun.log" \
      benchmark/offline/deepseek_v4_perf_matrix.py \
      --model-path "${MODEL_PATH}" \
      --variants "${VARIANT}" \
      --scenarios decode_throughput_bs8 \
      --prompt-len "${PROMPT_LEN}" \
      --decode-len "${SHORT_DECODE_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --repeats 1 \
      --warmup-repeats 0 \
      --page-size "${PAGE_SIZE}" \
      --num-pages "${NUM_PAGES}" \
      --output-dir "${out_dir}" \
      --keep-going
  )
}

run_capacity_probe() {
  local out_dir="${RAW_DIR}/capacity_auto_victory"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    "${MILESTONE_DIR}/scripts/capacity_probe.py" \
    --model-path "${MODEL_PATH}" \
    --output "${out_dir}/capacity_probe.json" \
    --tensor-parallel-size "${TP}" \
    --page-size "${PAGE_SIZE}" \
    --memory-ratio 0.9 \
    --max-running-req "${BATCH_SIZE}" \
    --max-seq-len "$((PROMPT_LEN + LONG_DECODE_LEN))" \
    --max-extend-tokens "${PROMPT_LEN}" \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 \
    --cuda-graph-capture-greedy-sample
}

capture_baseline
run_smoke
run_macro "${PROMPT_LEN}x${LONG_DECODE_LEN}" "${LONG_DECODE_LEN}"
run_macro "${PROMPT_LEN}x${SHORT_DECODE_LEN}" "${SHORT_DECODE_LEN}"

if [[ "${RUN_OWNER_TIMING}" == "1" ]]; then
  run_owner_timing
fi

if [[ "${RUN_CAPACITY_PROBE}" == "1" ]]; then
  run_capacity_probe
fi

python "${MILESTONE_DIR}/scripts/summarize_post0778_roofline.py" \
  --milestone-dir "${MILESTONE_DIR}"
