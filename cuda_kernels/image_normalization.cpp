#include <torch/extension.h>

torch::Tensor normalize_images_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor std);

torch::Tensor normalize_images(torch::Tensor input, torch::Tensor mean, torch::Tensor std) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(mean.is_cuda(), "mean must be a CUDA tensor");
  TORCH_CHECK(std.is_cuda(), "std must be a CUDA tensor");
  return normalize_images_cuda(input, mean, std);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "normalize_images",
      &normalize_images,
      "Normalize contiguous NCHW images with per-channel mean and std (CUDA)");
}
