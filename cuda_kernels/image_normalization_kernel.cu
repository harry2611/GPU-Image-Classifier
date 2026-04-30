
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void normalize_images_kernel(
    const float* input, float* output,
    const float* mean, const float* inv_std,
    int64_t total_elements, int64_t channels, int64_t channel_stride) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const int64_t channel_index = (index / channel_stride) % channels;
  output[index] = (input[index] - mean[channel_index]) * inv_std[channel_index];
}

// v2: float4 vectorized loads + division-free 3D grid
__global__ void normalize_images_kernel_v2(
    const float4* __restrict__ input4,
    float4* __restrict__ output4,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    int64_t channel_stride4) {

  const int64_t local_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_idx >= channel_stride4) return;

  const int64_t c = blockIdx.y;  // channel — free, no division
  const int64_t n = blockIdx.z;  // batch  — free, no division

  const float m = mean[c];
  const float s = inv_std[c];

  const int64_t global_idx = (n * gridDim.y + c) * channel_stride4 + local_idx;

  float4 val = input4[global_idx];
  val.x = (val.x - m) * s;
  val.y = (val.y - m) * s;
  val.z = (val.z - m) * s;
  val.w = (val.w - m) * s;
  output4[global_idx] = val;
}

}  // namespace

torch::Tensor normalize_images_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor std) {
  TORCH_CHECK(input.dim() == 4, "input must have shape [N, C, H, W]");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(mean.scalar_type() == torch::kFloat32, "mean must be float32");
  TORCH_CHECK(std.scalar_type() == torch::kFloat32, "std must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(mean.is_contiguous(), "mean must be contiguous");
  TORCH_CHECK(std.is_contiguous(), "std must be contiguous");
  TORCH_CHECK(mean.numel() == input.size(1), "mean must have one value per channel");
  TORCH_CHECK(std.numel() == input.size(1), "std must have one value per channel");

  auto output = torch::empty_like(input);
  auto inv_std = torch::reciprocal(std);

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t channel_stride = input.size(2) * input.size(3);

  if (channel_stride % 4 == 0) {
    const int64_t channel_stride4 = channel_stride / 4;
    constexpr int threads = 256;
    const int blocks_x = static_cast<int>((channel_stride4 + threads - 1) / threads);
    dim3 grid(blocks_x, static_cast<int>(C), static_cast<int>(N));

    normalize_images_kernel_v2<<<grid, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        channel_stride4);
  } else {
    const int64_t total_elements = input.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    normalize_images_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        total_elements, C, channel_stride);
  }

  cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(error));
  return output;
}
