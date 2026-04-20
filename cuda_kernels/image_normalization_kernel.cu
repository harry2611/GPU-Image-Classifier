#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void normalize_images_kernel(
    const float* input,
    float* output,
    const float* mean,
    const float* inv_std,
    int64_t total_elements,
    int64_t channels,
    int64_t channel_stride) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) {
    return;
  }

  const int64_t channel_index = (index / channel_stride) % channels;
  output[index] = (input[index] - mean[channel_index]) * inv_std[channel_index];
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

  const int64_t total_elements = input.numel();
  const int64_t channels = input.size(1);
  const int64_t channel_stride = input.size(2) * input.size(3);

  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

  normalize_images_kernel<<<blocks, threads_per_block>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      mean.data_ptr<float>(),
      inv_std.data_ptr<float>(),
      total_elements,
      channels,
      channel_stride);

  cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(error));

  return output;
}
