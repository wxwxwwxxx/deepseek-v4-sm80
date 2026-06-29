/*
 * DeepSeek V4 sm80 two-source sparse attention.
 *
 * This local CUDA kernel intentionally does not depend on FlashMLA.  It
 * consumes compressed C4/C128 KV rows and SWA KV rows as separate bf16 flat
 * caches, matching mini-sglang's sm80 cache policy.
 */

#include <minisgl/tensor.h>
#include <minisgl/utils.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr DLDataType kBF16DType{
    .code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
constexpr int kHeadDim = 512;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16 *ptr) {
  return __bfloat162float(*ptr);
}

struct SparseAttentionParams {
  const __nv_bfloat16 *__restrict__ q;
  const __nv_bfloat16 *__restrict__ compressed_cache;
  const __nv_bfloat16 *__restrict__ swa_cache;
  const int32_t *__restrict__ compressed_indices;
  const int32_t *__restrict__ compressed_lengths;
  const int32_t *__restrict__ swa_indices;
  const int32_t *__restrict__ swa_lengths;
  const float *__restrict__ sink;
  __nv_bfloat16 *__restrict__ out;
  int64_t q_stride_token;
  int64_t q_stride_head;
  int64_t compressed_index_stride;
  int64_t swa_index_stride;
  int64_t out_stride_token;
  int64_t out_stride_head;
  int num_heads;
  float softmax_scale;
  bool has_sink;
};

template <bool kHasCompressed>
__global__ __launch_bounds__(kThreads) void
sparse_attention_kernel(const __grid_constant__ SparseAttentionParams params) {
  const int row = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ float shared_l;
  __shared__ float shared_alpha;
  __shared__ float shared_beta;

  const int d0 = tid * 2;
  const int d1 = d0 + 1;
  const auto q_base =
      params.q + row * params.q_stride_token + head * params.q_stride_head;
  const float q0 = load_bf16(q_base + d0);
  const float q1 = load_bf16(q_base + d1);

  float m = params.has_sink ? params.sink[head] : -FLT_MAX;
  float l = params.has_sink ? 1.0f : 0.0f;

  const int raw_compressed_len =
      kHasCompressed ? params.compressed_lengths[row] : 0;
  const int raw_swa_len = params.swa_lengths[row];
  const int compressed_len =
      raw_compressed_len > 0 ? raw_compressed_len : 0;
  const int swa_len = raw_swa_len > 0 ? raw_swa_len : 0;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  auto score_for = [&](const __nv_bfloat16 *__restrict__ cache,
                       int32_t cache_row) {
    float partial = 0.0f;
    if (cache_row >= 0) {
      const auto kv_base = cache + static_cast<int64_t>(cache_row) * kHeadDim;
      partial = q0 * load_bf16(kv_base + d0) + q1 * load_bf16(kv_base + d1);
    }
    return block_reduce_sum(partial);
  };

  auto accumulate_online = [&](const __nv_bfloat16 *__restrict__ cache,
                               int32_t cache_row) {
    const float dot = score_for(cache, cache_row);
    if (tid == 0) {
      if (cache_row >= 0) {
        const float score = dot * params.softmax_scale;
        const float new_m = fmaxf(m, score);
        shared_alpha = expf(m - new_m);
        shared_beta = expf(score - new_m);
        l = l * shared_alpha + shared_beta;
        m = new_m;
      } else {
        shared_alpha = 1.0f;
        shared_beta = 0.0f;
      }
    }
    __syncthreads();
    const float alpha = shared_alpha;
    const float beta = shared_beta;
    acc0 *= alpha;
    acc1 *= alpha;
    if (cache_row >= 0 && beta != 0.0f) {
      const auto kv_base = cache + static_cast<int64_t>(cache_row) * kHeadDim;
      acc0 += beta * load_bf16(kv_base + d0);
      acc1 += beta * load_bf16(kv_base + d1);
    }
    __syncthreads();
  };

  if constexpr (kHasCompressed) {
    const auto row_indices =
        params.compressed_indices + row * params.compressed_index_stride;
    for (int i = 0; i < compressed_len; ++i) {
      accumulate_online(params.compressed_cache, row_indices[i]);
    }
  }
  const auto swa_row_indices = params.swa_indices + row * params.swa_index_stride;
  for (int i = 0; i < swa_len; ++i) {
    accumulate_online(params.swa_cache, swa_row_indices[i]);
  }

  if (tid == 0) {
    shared_l = l;
  }
  __syncthreads();
  const float inv_l = shared_l > 0.0f ? 1.0f / shared_l : 0.0f;

  const auto out_base =
      params.out + row * params.out_stride_token + head * params.out_stride_head;
  out_base[d0] = __float2bfloat16(acc0 * inv_l);
  out_base[d1] = __float2bfloat16(acc1 * inv_l);
}

template <bool kHasCompressed> struct DSV4SparseAttentionTwoSourceBF16Kernel {
  static constexpr auto kernel = sparse_attention_kernel<kHasCompressed>;

  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView compressed_cache,
                  const tvm::ffi::TensorView compressed_indices,
                  const tvm::ffi::TensorView compressed_lengths,
                  const tvm::ffi::TensorView swa_cache,
                  const tvm::ffi::TensorView swa_indices,
                  const tvm::ffi::TensorView swa_lengths,
                  const tvm::ffi::TensorView sink,
                  const tvm::ffi::TensorView out,
                  const double softmax_scale, const bool has_sink) {
    using namespace host;
    auto T = SymbolicSize{"tokens"};
    auto H = SymbolicSize{"heads"};
    auto CW = SymbolicSize{"compressed_width"};
    auto SW = SymbolicSize{"swa_width"};
    auto QS0 = SymbolicSize{"q_stride_token"};
    auto QS1 = SymbolicSize{"q_stride_head"};
    auto CS0 = SymbolicSize{"compressed_index_stride"};
    auto SS0 = SymbolicSize{"swa_index_stride"};
    auto OS0 = SymbolicSize{"out_stride_token"};
    auto OS1 = SymbolicSize{"out_stride_head"};
    auto device = SymbolicDevice{};

    TensorMatcher({T, H, kHeadDim})
        .with_strides({QS0, QS1, 1})
        .with_dtype(details::DTypeRef{kBF16DType})
        .with_device<kDLCUDA>(device)
        .verify(q);
    TensorMatcher({-1, kHeadDim})
        .with_dtype(details::DTypeRef{kBF16DType})
        .with_device<kDLCUDA>(device)
        .verify(compressed_cache);
    TensorMatcher({T, CW})
        .with_strides({CS0, 1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(compressed_indices);
    TensorMatcher({T})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(compressed_lengths);
    TensorMatcher({-1, kHeadDim})
        .with_dtype(details::DTypeRef{kBF16DType})
        .with_device<kDLCUDA>(device)
        .verify(swa_cache);
    TensorMatcher({T, SW})
        .with_strides({SS0, 1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(swa_indices);
    TensorMatcher({T})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(swa_lengths);
    TensorMatcher({-1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(sink);
    TensorMatcher({T, H, kHeadDim})
        .with_strides({OS0, OS1, 1})
        .with_dtype(details::DTypeRef{kBF16DType})
        .with_device<kDLCUDA>(device)
        .verify(out);

    RuntimeCheck(CW.unwrap() >= 1, "compressed index width must be >= 1");
    RuntimeCheck(SW.unwrap() >= 1, "SWA index width must be >= 1");
    RuntimeCheck(T.unwrap() >= 0, "tokens must be non-negative");
    RuntimeCheck(H.unwrap() > 0, "heads must be positive");

    const auto params = SparseAttentionParams{
        .q = static_cast<const __nv_bfloat16 *>(q.data_ptr()),
        .compressed_cache =
            static_cast<const __nv_bfloat16 *>(compressed_cache.data_ptr()),
        .swa_cache = static_cast<const __nv_bfloat16 *>(swa_cache.data_ptr()),
        .compressed_indices =
            static_cast<const int32_t *>(compressed_indices.data_ptr()),
        .compressed_lengths =
            static_cast<const int32_t *>(compressed_lengths.data_ptr()),
        .swa_indices = static_cast<const int32_t *>(swa_indices.data_ptr()),
        .swa_lengths = static_cast<const int32_t *>(swa_lengths.data_ptr()),
        .sink = static_cast<const float *>(sink.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .q_stride_token = QS0.unwrap(),
        .q_stride_head = QS1.unwrap(),
        .compressed_index_stride = CS0.unwrap(),
        .swa_index_stride = SS0.unwrap(),
        .out_stride_token = OS0.unwrap(),
        .out_stride_head = OS1.unwrap(),
        .num_heads = static_cast<int>(H.unwrap()),
        .softmax_scale = static_cast<float>(softmax_scale),
        .has_sink = has_sink,
    };

    if (T.unwrap() == 0) {
      return;
    }

    const auto dl_device = device.unwrap();
    auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));
    kernel<<<dim3(static_cast<uint32_t>(T.unwrap()),
                  static_cast<uint32_t>(H.unwrap())),
             kThreads, 0, stream>>>(params);
    const auto launch_result = ::cudaGetLastError();
    if (launch_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(launch_result));
    }
  }
};

} // namespace
