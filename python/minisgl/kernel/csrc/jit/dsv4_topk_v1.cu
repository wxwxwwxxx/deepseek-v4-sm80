/*
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * It's borrowed from SGLang's DeepSeek V4 topk_transform v1 JIT kernel and
 * adapted to mini-sglang's local tvm_ffi JIT wrapper. The key algorithmic
 * pieces kept here are the radix top-k selection, short-sequence sequential
 * fallback, and page-table translation.
 */

#include <minisgl/tensor.h>
#include <minisgl/utils.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <stdexcept>

namespace {

template <uint32_t kTopK> struct TopKConfig {
  static constexpr uint32_t kBlockSize = kTopK;
  static constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);
  static_assert(kTopK == 512 || kTopK == 1024,
                "DeepSeek V4 topk v1 only supports topk 512 or 1024.");
};

template <uint32_t kTopK> struct TopKParams {
  const float *__restrict__ scores;
  const int32_t *__restrict__ seq_lens;
  const int32_t *__restrict__ page_table;
  int32_t *__restrict__ page_indices;
  int32_t *__restrict__ raw_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
};

template <uint32_t kTopK> struct TopKGlobalLensParams {
  const float *__restrict__ scores;
  const int32_t *__restrict__ seq_lens;
  const int32_t *__restrict__ page_table;
  int32_t *__restrict__ page_indices;
  int32_t *__restrict__ raw_indices;
  int32_t *__restrict__ full_indices;
  int32_t *__restrict__ topk_lens;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
  int32_t ratio;
};

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000)
                     ? static_cast<uint16_t>(~bits)
                     : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t
page_to_indices(const int32_t *__restrict__ page_table, uint32_t i,
                uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

__device__ __forceinline__ int32_t full_from_page_index(int32_t page_index,
                                                        int32_t ratio) {
  return page_index >= 0 ? page_index * ratio + (ratio - 1) : -1;
}

template <uint32_t kTopK>
__device__ void naive_transform(const int32_t *__restrict__ page_table,
                                int32_t *__restrict__ indices,
                                int32_t *__restrict__ raw_indices,
                                const uint32_t length,
                                const uint32_t page_bits) {
  constexpr uint32_t kBlockSize = TopKConfig<kTopK>::kBlockSize;
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    raw_indices[tx] = tx;
  } else if (kTopK == kBlockSize || tx < kTopK) {
    indices[tx] = -1;
    raw_indices[tx] = -1;
  }
}

template <uint32_t kTopK>
__device__ void radix_topk(const float *__restrict__ input,
                           int32_t *__restrict__ output,
                           const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = TopKConfig<kTopK>::kBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE =
      TopKConfig<kTopK>::kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][SMEM_INPUT_SIZE];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
  auto &s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  if (tx < RADIX + 1)
    s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk &&
      s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < SMEM_INPUT_SIZE) {
          s_input_idx[0][pos] = idx;
          const auto bin32 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin32 >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input =
        raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk &&
        s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < SMEM_INPUT_SIZE) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin32 = convert_to_uint32(raw_input);
              const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <uint32_t kTopK>
__global__ __launch_bounds__(TopKConfig<kTopK>::kBlockSize) void
topk_transform_kernel(const __grid_constant__ TopKParams<kTopK> params) {
  const uint32_t work_id = blockIdx.x;
  const uint32_t seq_len = params.seq_lens[work_id];
  const auto score_ptr = params.scores + work_id * params.score_stride;
  const auto page_ptr = params.page_table + work_id * params.page_table_stride;
  const auto indices_ptr = params.page_indices + work_id * kTopK;
  const auto raw_indices_ptr = params.raw_indices + work_id * kTopK;

  if (seq_len <= kTopK) {
    naive_transform<kTopK>(page_ptr, indices_ptr, raw_indices_ptr, seq_len,
                           params.page_bits);
  } else {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk<kTopK>(score_ptr, s_topk_indices, seq_len);
    const auto tx = threadIdx.x;
    if (tx < kTopK) {
      indices_ptr[tx] =
          page_to_indices(page_ptr, s_topk_indices[tx], params.page_bits);
      raw_indices_ptr[tx] = s_topk_indices[tx];
    }
  }
}

template <uint32_t kTopK>
__global__ __launch_bounds__(TopKConfig<kTopK>::kBlockSize) void
topk_transform_global_lens_kernel(
    const __grid_constant__ TopKGlobalLensParams<kTopK> params) {
  const uint32_t work_id = blockIdx.x;
  const uint32_t seq_len = params.seq_lens[work_id];
  const auto score_ptr = params.scores + work_id * params.score_stride;
  const auto page_ptr = params.page_table + work_id * params.page_table_stride;
  const auto indices_ptr = params.page_indices + work_id * kTopK;
  const auto raw_indices_ptr = params.raw_indices + work_id * kTopK;
  const auto full_indices_ptr = params.full_indices + work_id * kTopK;
  const auto tx = threadIdx.x;

  if (tx == 0) {
    params.topk_lens[work_id] =
        static_cast<int32_t>(seq_len < kTopK ? seq_len : kTopK);
  }

  if (seq_len <= kTopK) {
    if (tx < seq_len) {
      const auto page_index = page_to_indices(page_ptr, tx, params.page_bits);
      indices_ptr[tx] = page_index;
      raw_indices_ptr[tx] = tx;
      full_indices_ptr[tx] = full_from_page_index(page_index, params.ratio);
    } else if (kTopK == TopKConfig<kTopK>::kBlockSize || tx < kTopK) {
      indices_ptr[tx] = -1;
      raw_indices_ptr[tx] = -1;
      full_indices_ptr[tx] = -1;
    }
  } else {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk<kTopK>(score_ptr, s_topk_indices, seq_len);
    if (tx < kTopK) {
      const auto raw_index = s_topk_indices[tx];
      const auto page_index =
          page_to_indices(page_ptr, raw_index, params.page_bits);
      indices_ptr[tx] = page_index;
      raw_indices_ptr[tx] = raw_index;
      full_indices_ptr[tx] = full_from_page_index(page_index, params.ratio);
    }
  }
}

template <uint32_t kTopK> struct DSV4TopKTransformKernel {
  static constexpr auto kernel = topk_transform_kernel<kTopK>;

  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView seq_lens,
                  const tvm::ffi::TensorView page_table,
                  const tvm::ffi::TensorView page_indices,
                  const uint32_t page_size,
                  const tvm::ffi::TensorView raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};

    TensorMatcher({B, -1})
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(scores);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(seq_lens);
    TensorMatcher({B, -1})
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(page_table);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(page_indices);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(raw_indices);

    if (!(page_size > 0 && (page_size & (page_size - 1)) == 0)) {
      throw std::runtime_error("page_size must be a positive power of two");
    }
    const auto page_bits = static_cast<uint32_t>(__builtin_ctz(page_size));

    const auto params = TopKParams<kTopK>{
        .scores = static_cast<const float *>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t *>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t *>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t *>(page_indices.data_ptr()),
        .raw_indices = static_cast<int32_t *>(raw_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
    };

    constexpr auto kDynamicSMEM = TopKConfig<kTopK>::kSMEM + sizeof(int32_t);
    static const auto attr_result = [] {
      return ::cudaFuncSetAttribute(
          reinterpret_cast<const void *>(kernel),
          ::cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSMEM);
    }();
    if (attr_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(attr_result));
    }
    const auto dl_device = device.unwrap();
    auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));
    kernel<<<static_cast<uint32_t>(B.unwrap()), TopKConfig<kTopK>::kBlockSize,
             kDynamicSMEM, stream>>>(params);
    const auto launch_result = ::cudaGetLastError();
    if (launch_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(launch_result));
    }
  }
};

template <uint32_t kTopK> struct DSV4TopKTransformGlobalLensKernel {
  static constexpr auto kernel = topk_transform_global_lens_kernel<kTopK>;

  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView seq_lens,
                  const tvm::ffi::TensorView page_table,
                  const tvm::ffi::TensorView page_indices,
                  const uint32_t page_size,
                  const tvm::ffi::TensorView raw_indices,
                  const tvm::ffi::TensorView full_indices,
                  const tvm::ffi::TensorView topk_lens,
                  const int32_t ratio) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};

    TensorMatcher({B, -1})
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(scores);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(seq_lens);
    TensorMatcher({B, -1})
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(page_table);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(page_indices);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(raw_indices);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(full_indices);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(topk_lens);

    if (!(page_size > 0 && (page_size & (page_size - 1)) == 0)) {
      throw std::runtime_error("page_size must be a positive power of two");
    }
    if (ratio <= 0) {
      throw std::runtime_error("ratio must be positive");
    }
    const auto page_bits = static_cast<uint32_t>(__builtin_ctz(page_size));

    const auto params = TopKGlobalLensParams<kTopK>{
        .scores = static_cast<const float *>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t *>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t *>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t *>(page_indices.data_ptr()),
        .raw_indices = static_cast<int32_t *>(raw_indices.data_ptr()),
        .full_indices = static_cast<int32_t *>(full_indices.data_ptr()),
        .topk_lens = static_cast<int32_t *>(topk_lens.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
        .ratio = ratio,
    };

    constexpr auto kDynamicSMEM = TopKConfig<kTopK>::kSMEM + sizeof(int32_t);
    static const auto attr_result = [] {
      return ::cudaFuncSetAttribute(
          reinterpret_cast<const void *>(kernel),
          ::cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSMEM);
    }();
    if (attr_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(attr_result));
    }
    const auto dl_device = device.unwrap();
    auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));
    kernel<<<static_cast<uint32_t>(B.unwrap()), TopKConfig<kTopK>::kBlockSize,
             kDynamicSMEM, stream>>>(params);
    const auto launch_result = ::cudaGetLastError();
    if (launch_result != ::cudaSuccess) {
      throw std::runtime_error(::cudaGetErrorString(launch_result));
    }
  }
};

} // namespace
