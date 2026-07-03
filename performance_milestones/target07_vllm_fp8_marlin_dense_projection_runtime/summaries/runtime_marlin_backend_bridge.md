# Runtime Marlin Backend Bridge

## Interpreter Matrix

| Interpreter | Result | Notes |
| --- | --- | --- |
| `/usr/bin/python` | blocked | Mini default env has torch `2.9.1+cu128` and no installed `vllm`. Adding `/workspace/vllm-dsv4-docker` to `PYTHONPATH` finds vLLM, but `vllm/_C.abi3.so` has a torch ABI mismatch. |
| `/workspace/venvs/vllm-dsv4/bin/python` | partial | vLLM imports, `gptq_marlin_repack` and `marlin_gemm` are present, and focused Marlin owner microbench runs. Mini benchmark modules import with repo `PYTHONPATH`. |
| `/workspace/venvs/vllm-dsv4/bin/torchrun` | blocked for full mini runtime | First TP8 smoke failed on bad inherited `sgl_kernel`; after optional-package probing was fixed, the promoted baseline hung in CUDA graph capture. Manual interrupt/fail-open then exposed a mini-owned Marlin WNA16 extension ABI mismatch under torch `2.11.0+cu128`. |

## Probe Artifacts

- Raw availability: `../raw/runtime_marlin_backend_availability.json`
- Focused runtime microbench: `focused_marlin_projection_runtime_microbench.md`
- TP8 smoke attempts: `../raw/tp8_text_smoke_attempts.json`

## Bridge Status

The vLLM Marlin backend is callable in focused CUDA code from the vLLM venv, but
the mini runtime bridge is not promotion-safe yet:

- default mini env cannot import the vLLM custom op ABI;
- vLLM venv can import mini, but its inherited `sgl_kernel` package is not a
  valid A100/sm80 build for torch `2.11.0+cu128`;
- after downgrading optional `sgl_kernel` probing to a clean unavailable state,
  the promoted baseline still did not complete TP8 CUDA graph capture;
- capture fail-open exposed a second ABI issue:
  `/root/.cache/minisgl/marlin_wna16/minisgl_marlin_wna16.so` was not loadable
  under the vLLM venv torch ABI.

Decision: keep the code path default-off and do not run macro/profile gates for
promotion. The next bridge target should provide a single coherent mini runtime
environment: either a mini-owned dense FP8 Marlin op bridge compiled against the
default mini torch, or a vLLM runner environment with matching `sgl_kernel` and
mini graph capture verified on the promoted baseline.
