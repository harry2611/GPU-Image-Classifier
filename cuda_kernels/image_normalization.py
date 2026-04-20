from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / ".cache" / "torch_extensions" / "image_normalization"
EXTENSION_NAME = "image_normalization_cuda_ext"


def get_cuda_extension_status(verbose: bool = False) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available on this machine."
    if CUDA_HOME is None:
        return False, "CUDA_HOME is not configured, so the extension cannot be compiled."

    try:
        _load_cuda_extension(verbose=verbose)
    except Exception as exc:  # pragma: no cover - exercised only on CUDA machines.
        return False, f"Failed to build or load the CUDA extension: {exc}"

    return True, "CUDA extension is available."


def normalize_images_cuda_extension(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    extension = _load_cuda_extension(verbose=verbose)
    _validate_inputs(images=images, mean=mean, std=std)
    return extension.normalize_images(
        images.contiguous(),
        mean.to(device=images.device, dtype=torch.float32).contiguous(),
        std.to(device=images.device, dtype=torch.float32).contiguous(),
    )


@lru_cache(maxsize=1)
def _load_cuda_extension(verbose: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, so the extension cannot be loaded.")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not configured.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=EXTENSION_NAME,
        sources=[
            str(SOURCE_DIR / "image_normalization.cpp"),
            str(SOURCE_DIR / "image_normalization_kernel.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


def _validate_inputs(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    if images.device.type != "cuda":
        raise ValueError("images must be a CUDA tensor.")
    if images.dtype != torch.float32:
        raise ValueError("images must use torch.float32 for the custom CUDA kernel.")
    if images.dim() != 4:
        raise ValueError("images must have shape [N, C, H, W].")
    if mean.numel() != images.shape[1]:
        raise ValueError("mean must contain one value per channel.")
    if std.numel() != images.shape[1]:
        raise ValueError("std must contain one value per channel.")
