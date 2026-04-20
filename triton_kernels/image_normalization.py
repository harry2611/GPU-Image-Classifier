from __future__ import annotations

import torch

TRITON_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - only exercised on Triton-capable environments.
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import path depends on host machine.
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = exc


if triton is not None:  # pragma: no branch

    @triton.jit
    def _image_normalization_kernel(
        input_ptr,
        output_ptr,
        mean_ptr,
        inv_std_ptr,
        total_elements,
        channel_stride,
        channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(axis=0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        channel_index = (offsets // channel_stride) % channels
        means = tl.load(mean_ptr + channel_index, mask=mask, other=0.0)
        inv_std = tl.load(inv_std_ptr + channel_index, mask=mask, other=1.0)
        normalized = (values - means) * inv_std
        tl.store(output_ptr + offsets, normalized, mask=mask)


def get_triton_status() -> tuple[bool, str]:
    if triton is None:
        return False, f"Triton is not installed or failed to import: {TRITON_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available, so Triton kernels cannot run."
    return True, "Triton is available."


def normalize_images_triton(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")
    _validate_inputs(images=images, mean=mean, std=std)

    output = torch.empty_like(images)
    inv_std = torch.reciprocal(std.to(device=images.device, dtype=torch.float32))
    mean = mean.to(device=images.device, dtype=torch.float32)
    total_elements = output.numel()
    channel_stride = images.shape[-1] * images.shape[-2]
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    _image_normalization_kernel[grid](
        images.contiguous(),
        output,
        mean.contiguous(),
        inv_std.contiguous(),
        total_elements,
        channel_stride,
        images.shape[1],
        BLOCK_SIZE=block_size,
    )
    return output


def _validate_inputs(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    if images.device.type != "cuda":
        raise ValueError("images must be a CUDA tensor for Triton normalization.")
    if images.dtype != torch.float32:
        raise ValueError("images must use torch.float32 for Triton normalization.")
    if images.dim() != 4:
        raise ValueError("images must have shape [N, C, H, W].")
    if mean.numel() != images.shape[1]:
        raise ValueError("mean must contain one value per channel.")
    if std.numel() != images.shape[1]:
        raise ValueError("std must contain one value per channel.")
