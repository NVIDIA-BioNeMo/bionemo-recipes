// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

// SPDX-License-Identifier: LicenseRef-Apache2

// Keep the native BF16 rounding points and accumulation order.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/extension.h>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float fp32(bf16 value) { return __bfloat162float(value); }
__device__ __forceinline__ float fp32(float value) { return value; }
__device__ __forceinline__ bf16 rounded(float value) { return __float2bfloat16_rn(value); }

// Each thread reads and writes one rotary pair.
template <typename Frequency>
__global__ void
inplace_rope_kernel(bf16 *__restrict__ tokens, const Frequency *__restrict__ frequencies,
                    int64_t total_pairs, int heads, int rotary_dim, int64_t token_stride,
                    int64_t head_stride, int64_t frequency_token_stride,
                    int64_t frequency_pair_stride, int64_t frequency_dim_stride, bool interleaved) {
  const int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total_pairs)
    return;
  const int half = rotary_dim / 2;
  const int dimension_pair = pair % half;
  const int head = (pair / half) % heads;
  const int64_t token = pair / (int64_t(half) * heads);
  const int first = interleaved ? 2 * dimension_pair : dimension_pair;
  const int second = interleaved ? first + 1 : first + half;
  const int64_t base = token * token_stride + head * head_stride;
  const float a = fp32(tokens[base + first]), b = fp32(tokens[base + second]);
  const int64_t fbase = token * frequency_token_stride;
  const float c0 = fp32(frequencies[fbase + first * frequency_dim_stride]);
  const float c1 = fp32(frequencies[fbase + second * frequency_dim_stride]);
  const float s0 = fp32(frequencies[fbase + frequency_pair_stride + first * frequency_dim_stride]);
  const float s1 = fp32(frequencies[fbase + frequency_pair_stride + second * frequency_dim_stride]);
  tokens[base + first] = rounded(a * c0 - b * s0);
  tokens[base + second] = rounded(b * c1 + a * s1);
}

__global__ void compact_projection_kernel(const bf16 *__restrict__ projection,
                                          const bf16 *__restrict__ weight,
                                          const int32_t *__restrict__ sequence_ids,
                                          bf16 *__restrict__ gate, bf16 *__restrict__ drive,
                                          int64_t tokens, int channels, int taps, int group_width) {
  constexpr int CT = 128, TT = 8;
  const int channel_tiles = (channels + CT - 1) / CT;
  const int64_t token_start = (int64_t(blockIdx.x) / channel_tiles) * TT;
  const int channel_start = (blockIdx.x % channel_tiles) * CT;
  for (int local = threadIdx.x; local < TT * CT; local += blockDim.x) {
    const int64_t token = token_start + local / CT;
    const int channel = channel_start + local % CT;
    if (token >= tokens || channel >= channels)
      continue;
    const int32_t sequence = sequence_ids[token];
    float a = 0.f, b = 0.f, v = 0.f;
#pragma unroll 4
    for (int lag = 0; lag < taps; ++lag) {
      const int64_t source = token - lag;
      if (source < 0 || sequence_ids[source] != sequence)
        continue;
      const int64_t base = source * (int64_t(3) * channels) + 3 * int64_t(channel);
      const int reverse = taps - 1 - lag;
      a = fmaf(fp32(projection[base]), fp32(weight[((3 * channel) / group_width) * taps + reverse]),
               a);
      b = fmaf(fp32(projection[base + 1]),
               fp32(weight[((3 * channel + 1) / group_width) * taps + reverse]), b);
      v = fmaf(fp32(projection[base + 2]),
               fp32(weight[((3 * channel + 2) / group_width) * taps + reverse]), v);
    }
    const int64_t destination = token * int64_t(channels) + channel;
    gate[destination] = rounded(a);
    drive[destination] = rounded(fp32(rounded(b)) * fp32(rounded(v)));
  }
}

// Reuse projection rows and the filtered drive in shared memory.
template <int K, int TT, typename W>
__global__ void
fused_fir_kernel(const bf16 *__restrict__ projection, const bf16 *__restrict__ projection_weight,
                 const W *__restrict__ mixer_weight, const bf16 *__restrict__ diagonal,
                 const int32_t *__restrict__ sequence_ids,
                 const int32_t *__restrict__ local_positions, bf16 *__restrict__ output,
                 int64_t tokens, int channels, int projection_group_width, int mixer_group_width,
                 bool flip_filter, bool has_diagonal) {
  constexpr int CT = 32, P = 3, DR = TT + K - 1, RR = DR + P - 1;
  extern __shared__ unsigned char storage[];
  bf16 *raw = reinterpret_cast<bf16 *>(storage);
  bf16 *drive = raw + RR * 3 * CT;
  bf16 *gate = drive + DR * CT;
  int32_t *sequence = reinterpret_cast<int32_t *>(gate + TT * CT);
  const int channel_tiles = (channels + CT - 1) / CT;
  const int64_t token_start = (int64_t(blockIdx.x) / channel_tiles) * TT;
  const int channel_start = (blockIdx.x % channel_tiles) * CT;
  const int64_t raw_start = token_start - (K - 1) - (P - 1);
  for (int row = threadIdx.x; row < RR; row += blockDim.x) {
    const int64_t token = raw_start + row;
    sequence[row] = token >= 0 && token < tokens ? sequence_ids[token] : -1;
  }
  for (int linear = threadIdx.x; linear < RR * 3 * CT; linear += blockDim.x) {
    const int row = linear / (3 * CT), component = linear % (3 * CT);
    const int channel = channel_start + component / 3;
    const int64_t token = raw_start + row;
    const int64_t offset = token * (int64_t(3) * channels) + 3 * int64_t(channel) + component % 3;
    raw[linear] =
        token >= 0 && token < tokens && channel < channels ? projection[offset] : rounded(0.f);
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < DR * CT; linear += blockDim.x) {
    const int row = linear / CT, local_channel = linear % CT;
    const int channel = channel_start + local_channel;
    const int64_t token = raw_start + row + P - 1;
    float a = 0.f, b = 0.f, v = 0.f;
    if (channel < channels && token >= 0 && token < tokens) {
#pragma unroll
      for (int lag = 0; lag < P; ++lag) {
        const int source_row = row + P - 1 - lag;
        const bool valid =
            raw_start + source_row >= 0 && sequence[source_row] == sequence[row + P - 1];
        const int base = source_row * (3 * CT) + 3 * local_channel;
        const float av = valid ? fp32(raw[base]) : 0.f;
        const float bv = valid ? fp32(raw[base + 1]) : 0.f;
        const float vv = valid ? fp32(raw[base + 2]) : 0.f;
        a = fmaf(
            av, fp32(projection_weight[((3 * channel) / projection_group_width) * P + P - 1 - lag]),
            a);
        b = fmaf(
            bv,
            fp32(projection_weight[((3 * channel + 1) / projection_group_width) * P + P - 1 - lag]),
            b);
        v = fmaf(
            vv,
            fp32(projection_weight[((3 * channel + 2) / projection_group_width) * P + P - 1 - lag]),
            v);
      }
    }
    drive[linear] = rounded(fp32(rounded(b)) * fp32(rounded(v)));
    if (row >= K - 1)
      gate[(row - K + 1) * CT + local_channel] = rounded(a);
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < TT * CT; linear += blockDim.x) {
    const int row = linear / CT, local_channel = linear % CT;
    const int channel = channel_start + local_channel;
    const int64_t token = token_start + row;
    if (channel >= channels || token >= tokens)
      continue;
    const int position = local_positions[token];
    const int filter = channel / mixer_group_width;
    float accumulator = 0.f;
#pragma unroll
    for (int lag = 0; lag < K; ++lag) {
      const float value =
          position >= lag ? fp32(drive[(row + K - 1 - lag) * CT + local_channel]) : 0.f;
      const int tap = flip_filter ? K - 1 - lag : lag;
      accumulator = fmaf(value, fp32(mixer_weight[filter * K + tap]), accumulator);
    }
    if (has_diagonal)
      accumulator = fmaf(fp32(drive[(row + K - 1) * CT + local_channel]), fp32(diagonal[channel]),
                         accumulator);
    output[token * int64_t(channels) + channel] =
        rounded(fp32(rounded(accumulator)) * fp32(gate[linear]));
  }
}

void validate_projection(const torch::Tensor &projection, const torch::Tensor &weight,
                         const torch::Tensor &sequence_ids, int64_t group_width) {
  TORCH_CHECK(projection.is_cuda() && projection.is_contiguous() && projection.dim() == 2 &&
                  projection.scalar_type() == at::kBFloat16 && projection.size(0) > 0 &&
                  projection.size(1) > 0 && projection.size(1) % 3 == 0,
              "projection must be nonempty contiguous CUDA BF16 [T,3C]");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2 &&
                  weight.scalar_type() == at::kBFloat16 && weight.device() == projection.device(),
              "projection weight must be contiguous BF16 on the input device");
  TORCH_CHECK(group_width > 0 && weight.size(0) * group_width == projection.size(1) &&
                  weight.size(1) >= 2 && weight.size(1) <= 4,
              "invalid projection groups/taps");
  TORCH_CHECK(sequence_ids.device() == projection.device() && sequence_ids.is_contiguous() &&
                  sequence_ids.scalar_type() == at::kInt && sequence_ids.dim() == 1 &&
                  sequence_ids.size(0) == projection.size(0),
              "sequence_ids must be contiguous CUDA int32 [T]");
}

template <int K, int TT, typename W>
void launch_fir(const torch::Tensor &projection, const torch::Tensor &projection_weight,
                const torch::Tensor &mixer_weight, const torch::Tensor &diagonal,
                const torch::Tensor &sequence_ids, const torch::Tensor &local_positions,
                torch::Tensor &output, int pg, int mg, bool flip, bool has_diagonal) {
  constexpr int CT = 32, DR = TT + K - 1, RR = DR + 2;
  constexpr int shared = (RR * 3 * CT + DR * CT + TT * CT) * sizeof(bf16) + RR * sizeof(int32_t);
  auto kernel = fused_fir_kernel<K, TT, W>;
  if constexpr (shared > 49152) {
    static thread_local int configured_device = -1;
    if (configured_device != projection.get_device()) {
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
      configured_device = projection.get_device();
    }
  }
  const int channels = projection.size(1) / 3;
  const int64_t blocks = ((projection.size(0) + TT - 1) / TT) * ((channels + CT - 1) / CT);
  TORCH_CHECK(blocks <= 2147483647, "launch exceeds CUDA grid limit");
  kernel<<<blocks, 256, shared, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16 *>(projection.data_ptr()),
      reinterpret_cast<const bf16 *>(projection_weight.data_ptr()),
      reinterpret_cast<const W *>(mixer_weight.data_ptr()),
      reinterpret_cast<const bf16 *>(diagonal.data_ptr()), sequence_ids.data_ptr<int32_t>(),
      local_positions.data_ptr<int32_t>(), reinterpret_cast<bf16 *>(output.data_ptr()),
      projection.size(0), channels, pg, mg, flip, has_diagonal);
}
} // namespace

torch::Tensor inplace_rope(torch::Tensor tokens, torch::Tensor frequencies, bool interleaved) {
  TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == at::kBFloat16 && tokens.dim() == 3 &&
                  tokens.size(0) > 0 && tokens.size(1) > 0 && tokens.size(2) > 0 &&
                  tokens.stride(2) == 1 && tokens.stride(1) >= tokens.size(2) &&
                  tokens.stride(0) >= (tokens.size(1) - 1) * tokens.stride(1) + tokens.size(2),
              "in-place RoPE requires non-overlapping CUDA BF16 THD rows with "
              "contiguous head dimensions");
  TORCH_CHECK(
      frequencies.device() == tokens.device() &&
          (frequencies.scalar_type() == at::kBFloat16 || frequencies.scalar_type() == at::kFloat) &&
          frequencies.dim() == 4 && frequencies.size(0) == tokens.size(0) &&
          frequencies.size(1) == 1 && frequencies.size(2) == 2 && frequencies.size(3) > 0 &&
          frequencies.size(3) <= tokens.size(2) && frequencies.size(3) % 2 == 0,
      "expected BF16/FP32 token-preindexed [T,1,2,even rotary_dim] "
      "frequencies");
  c10::cuda::CUDAGuard guard(tokens.device());
  const int64_t pairs = tokens.size(0) * tokens.size(1) * (frequencies.size(3) / 2);
  const int64_t blocks = (pairs + 255) / 256;
  TORCH_CHECK(blocks <= 2147483647, "launch exceeds CUDA grid limit");
  if (frequencies.scalar_type() == at::kBFloat16) {
    inplace_rope_kernel<bf16><<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<bf16 *>(tokens.data_ptr()),
        reinterpret_cast<const bf16 *>(frequencies.data_ptr()), pairs, tokens.size(1),
        frequencies.size(3), tokens.stride(0), tokens.stride(1), frequencies.stride(0),
        frequencies.stride(2), frequencies.stride(3), interleaved);
  } else {
    inplace_rope_kernel<float><<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<bf16 *>(tokens.data_ptr()), frequencies.data_ptr<float>(), pairs,
        tokens.size(1), frequencies.size(3), tokens.stride(0), tokens.stride(1),
        frequencies.stride(0), frequencies.stride(2), frequencies.stride(3), interleaved);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tokens;
}

std::vector<torch::Tensor> compact_projection(torch::Tensor projection, torch::Tensor weight,
                                              torch::Tensor sequence_ids, int64_t group_width) {
  validate_projection(projection, weight, sequence_ids, group_width);
  c10::cuda::CUDAGuard guard(projection.device());
  auto gate = torch::empty({projection.size(0), projection.size(1) / 3}, projection.options());
  auto drive = torch::empty_like(gate);
  const int channels = gate.size(1);
  const int64_t blocks = ((gate.size(0) + 7) / 8) * ((channels + 127) / 128);
  TORCH_CHECK(blocks <= 2147483647, "launch exceeds CUDA grid limit");
  compact_projection_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16 *>(projection.data_ptr()),
      reinterpret_cast<const bf16 *>(weight.data_ptr()), sequence_ids.data_ptr<int32_t>(),
      reinterpret_cast<bf16 *>(gate.data_ptr()), reinterpret_cast<bf16 *>(drive.data_ptr()),
      gate.size(0), channels, weight.size(1), group_width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gate, drive};
}

torch::Tensor fused_fir(torch::Tensor projection, torch::Tensor projection_weight,
                        torch::Tensor mixer_weight, std::optional<torch::Tensor> diagonal,
                        torch::Tensor sequence_ids, torch::Tensor local_positions,
                        int64_t projection_group_width, int64_t mixer_group_width) {
  validate_projection(projection, projection_weight, sequence_ids, projection_group_width);
  TORCH_CHECK(projection_weight.size(1) == 3, "expected three-tap projection");
  const int channels = projection.size(1) / 3;
  TORCH_CHECK(mixer_weight.device() == projection.device() && mixer_weight.is_contiguous() &&
                  mixer_weight.dim() == 2 &&
                  (mixer_weight.scalar_type() == at::kFloat ||
                   mixer_weight.scalar_type() == at::kBFloat16) &&
                  mixer_group_width > 0 && mixer_weight.size(0) * mixer_group_width == channels &&
                  (mixer_weight.size(1) == 7 || mixer_weight.size(1) == 128),
              "expected grouped seven- or 128-tap BF16/FP32 filter");
  TORCH_CHECK(local_positions.device() == projection.device() && local_positions.is_contiguous() &&
                  local_positions.scalar_type() == at::kInt && local_positions.dim() == 1 &&
                  local_positions.size(0) == projection.size(0),
              "expected contiguous CUDA int32 positions [T]");
  TORCH_CHECK(!diagonal || (diagonal->device() == projection.device() &&
                            diagonal->is_contiguous() && diagonal->scalar_type() == at::kBFloat16 &&
                            diagonal->dim() == 1 && diagonal->size(0) == channels),
              "expected contiguous BF16 diagonal [C] on input device");
  c10::cuda::CUDAGuard guard(projection.device());
  auto output = torch::empty({projection.size(0), channels}, projection.options());
  const auto diagonal_value = diagonal.value_or(projection_weight);
  const bool flip = mixer_weight.size(1) == 7;
  const bool has_diagonal = diagonal.has_value();
  if (mixer_weight.scalar_type() == at::kFloat) {
    if (flip) {
      launch_fir<7, 64, float>(projection, projection_weight, mixer_weight, diagonal_value,
                               sequence_ids, local_positions, output, projection_group_width,
                               mixer_group_width, flip, has_diagonal);
    } else {
      launch_fir<128, 64, float>(projection, projection_weight, mixer_weight, diagonal_value,
                                 sequence_ids, local_positions, output, projection_group_width,
                                 mixer_group_width, flip, has_diagonal);
    }
  } else if (flip) {
    launch_fir<7, 64, bf16>(projection, projection_weight, mixer_weight, diagonal_value,
                            sequence_ids, local_positions, output, projection_group_width,
                            mixer_group_width, flip, has_diagonal);
  } else {
    launch_fir<128, 64, bf16>(projection, projection_weight, mixer_weight, diagonal_value,
                              sequence_ids, local_positions, output, projection_group_width,
                              mixer_group_width, flip, has_diagonal);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("inplace_rope", &inplace_rope);
  module.def("compact_projection", &compact_projection);
  module.def("fused_fir", &fused_fir);
}
