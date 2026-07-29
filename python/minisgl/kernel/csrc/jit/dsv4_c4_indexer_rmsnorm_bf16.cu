/*
 * DeepSeek V4 SM80 C4 indexer RMSNorm publication stage.
 *
 * The qualified Mini path materializes FP32 squares before PyTorch's generic
 * 128-wide mean reduction, then performs rsqrt, input scaling, FP32 weight
 * scaling, and a BF16 store as distinct rounding boundaries.  This kernel
 * reproduces that reduction and rounding order with one warp per row.
 */

#include <minisgl/tensor.h>
#include <minisgl/utils.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr DLDataType kBF16DType{
    .code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
constexpr int kHeadDim = 128;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

struct RMSNormParams {
  __nv_bfloat16 *__restrict__ input;
  const float *__restrict__ weight;
  const int64_t *__restrict__ loc;
  float eps;
  int32_t rows;
};

__device__ __forceinline__ float add_rn(float lhs, float rhs) {
  return __fadd_rn(lhs, rhs);
}

__device__ __forceinline__ float mul_rn(float lhs, float rhs) {
  return __fmul_rn(lhs, rhs);
}

__global__ __launch_bounds__(kThreads, 4)
void c4_indexer_rmsnorm_bf16_kernel(
    const __grid_constant__ RMSNormParams params) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= params.rows || params.loc[row] < 0) {
    return;
  }

  const auto row_ptr =
      params.input + static_cast<int64_t>(row) * kHeadDim;
  float values[4];
  float partials[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = __bfloat162float(row_ptr[lane + i * 32]);
    partials[i] = mul_rn(values[i], values[i]);
  }
  float sum = add_rn(partials[0], partials[1]);
  sum = add_rn(sum, partials[2]);
  sum = add_rn(sum, partials[3]);

  // PyTorch Reduce.cuh uses ascending shuffle offsets for a contiguous
  // 128-element reduction with block_width=32.
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum = add_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
  }
  sum = __shfl_sync(0xffffffffu, sum, 0);
  const float mean = mul_rn(sum, 1.0f / static_cast<float>(kHeadDim));
  const float inv_rms = rsqrtf(add_rn(mean, params.eps));

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * 32;
    const float normalized = mul_rn(values[i], inv_rms);
    const float weighted = mul_rn(normalized, params.weight[d]);
    row_ptr[d] = __float2bfloat16_rn(weighted);
  }
}

struct DSV4C4IndexerRMSNormBF16Kernel {
  static void run(const tvm::ffi::TensorView input,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView loc, const double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};

    TensorMatcher({rows, kHeadDim})
        .with_dtype(details::DTypeRef{kBF16DType})
        .with_device<kDLCUDA>(device)
        .verify(input);
    TensorMatcher({kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(weight);
    TensorMatcher({rows})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(loc);

    if (rows.unwrap() == 0) {
      return;
    }
    RuntimeCheck(rows.unwrap() <= INT32_MAX, "row count exceeds int32");
    const auto params = RMSNormParams{
        .input = static_cast<__nv_bfloat16 *>(input.data_ptr()),
        .weight = static_cast<const float *>(weight.data_ptr()),
        .loc = static_cast<const int64_t *>(loc.data_ptr()),
        .eps = static_cast<float>(eps),
        .rows = static_cast<int32_t>(rows.unwrap()),
    };
    const auto dl_device = device.unwrap();
    auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));
    const auto blocks =
        static_cast<uint32_t>((rows.unwrap() + kWarpsPerBlock - 1) /
                              kWarpsPerBlock);
    c4_indexer_rmsnorm_bf16_kernel<<<blocks, kThreads, 0, stream>>>(params);
    const auto launch_result = ::cudaGetLastError();
    if (launch_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(launch_result));
    }
  }
};

} // namespace
