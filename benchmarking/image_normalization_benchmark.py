from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

import torch

from cuda_kernels.image_normalization import (
    get_cuda_extension_status,
    normalize_images_cuda_extension,
)
from evaluation.metrics import plot_benchmark_results
from triton_kernels.image_normalization import get_triton_status, normalize_images_triton
from utils.config import KernelBenchmarkConfig
from utils.io_utils import ensure_directories, write_csv, write_json
from utils.torch_utils import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkRow:
    backend: str
    device: str
    status: str
    mean_latency_ms: float | None
    median_latency_ms: float | None
    std_latency_ms: float | None
    effective_bandwidth_gbps: float | None
    speedup_vs_cpu: float | None
    max_abs_error: float | None
    notes: str


def run_image_normalization_benchmark(config: KernelBenchmarkConfig) -> dict[str, object]:
    seed_everything(config.random_state)
    output_paths = _prepare_output_directories(config.output_dir)
    shape_tag = (
        f"b{config.batch_size}_c{config.channels}_h{config.height}_w{config.width}"
    )

    cpu_images, mean_tensor, std_tensor = _build_benchmark_inputs(config)
    reference_cpu = _pytorch_normalize(cpu_images, mean_tensor, std_tensor)
    benchmark_rows: list[BenchmarkRow] = []

    cpu_row = _run_completed_benchmark(
        backend="torch_cpu",
        device=torch.device("cpu"),
        func=lambda: _pytorch_normalize(cpu_images, mean_tensor, std_tensor),
        reference_output=reference_cpu,
        bytes_processed=_bytes_processed(cpu_images),
        notes="PyTorch reference implementation on CPU.",
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
    )
    benchmark_rows.append(cpu_row)

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        cuda_images = cpu_images.to(cuda_device)
        cuda_mean = mean_tensor.to(cuda_device)
        cuda_std = std_tensor.to(cuda_device)
        reference_cuda = _pytorch_normalize(cuda_images, cuda_mean, cuda_std)

        benchmark_rows.append(
            _run_completed_benchmark(
                backend="torch_cuda",
                device=cuda_device,
                func=lambda: _pytorch_normalize(cuda_images, cuda_mean, cuda_std),
                reference_output=reference_cuda,
                bytes_processed=_bytes_processed(cuda_images),
                notes="PyTorch broadcasting on CUDA.",
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )
        )

        benchmark_rows.append(
            _run_optional_cuda_backend(
                backend="cuda_extension",
                availability_fn=lambda: get_cuda_extension_status(
                    verbose=config.verbose_backend_loading
                ),
                benchmark_fn=lambda: normalize_images_cuda_extension(
                    cuda_images,
                    cuda_mean,
                    cuda_std,
                    verbose=config.verbose_backend_loading,
                ),
                reference_output=reference_cuda,
                device=cuda_device,
                bytes_processed=_bytes_processed(cuda_images),
                notes="Custom CUDA extension compiled through torch.utils.cpp_extension.",
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )
        )
        benchmark_rows.append(
            _run_optional_cuda_backend(
                backend="triton",
                availability_fn=get_triton_status,
                benchmark_fn=lambda: normalize_images_triton(cuda_images, cuda_mean, cuda_std),
                reference_output=reference_cuda,
                device=cuda_device,
                bytes_processed=_bytes_processed(cuda_images),
                notes="Triton JIT kernel for NCHW image normalization.",
                warmup_iterations=config.warmup_iterations,
                benchmark_iterations=config.benchmark_iterations,
            )
        )
    else:
        benchmark_rows.extend(
            [
                BenchmarkRow(
                    backend="torch_cuda",
                    device="cuda",
                    status="skipped",
                    mean_latency_ms=None,
                    median_latency_ms=None,
                    std_latency_ms=None,
                    effective_bandwidth_gbps=None,
                    speedup_vs_cpu=None,
                    max_abs_error=None,
                    notes="CUDA is not available on this machine.",
                ),
                BenchmarkRow(
                    backend="cuda_extension",
                    device="cuda",
                    status="skipped",
                    mean_latency_ms=None,
                    median_latency_ms=None,
                    std_latency_ms=None,
                    effective_bandwidth_gbps=None,
                    speedup_vs_cpu=None,
                    max_abs_error=None,
                    notes="CUDA is not available, so the custom extension cannot run.",
                ),
                BenchmarkRow(
                    backend="triton",
                    device="cuda",
                    status="skipped",
                    mean_latency_ms=None,
                    median_latency_ms=None,
                    std_latency_ms=None,
                    effective_bandwidth_gbps=None,
                    speedup_vs_cpu=None,
                    max_abs_error=None,
                    notes="CUDA is not available, so the Triton kernel cannot run.",
                ),
            ]
        )

    benchmark_rows = _apply_speedups(benchmark_rows)
    serialized_rows = [asdict(row) for row in benchmark_rows]

    csv_path = output_paths["metrics"] / f"image_normalization_{shape_tag}_benchmark.csv"
    json_path = output_paths["metrics"] / f"image_normalization_{shape_tag}_benchmark.json"
    plot_path = output_paths["figures"] / f"image_normalization_{shape_tag}_benchmark.png"

    write_csv(csv_path, serialized_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": config.operation,
        "tensor_shape": [config.batch_size, config.channels, config.height, config.width],
        "warmup_iterations": config.warmup_iterations,
        "benchmark_iterations": config.benchmark_iterations,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
        },
        "results": serialized_rows,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }
    write_json(json_path, payload)

    try:
        plot_benchmark_results(
            benchmark_rows=serialized_rows,
            output_path=plot_path,
            title="Image Normalization Benchmark",
        )
    except Exception as exc:  # pragma: no cover - plotting depends on host environment.
        LOGGER.warning("Failed to generate the benchmark plot: %s", exc)

    LOGGER.info("Saved benchmark results to %s", json_path)
    return payload


def _prepare_output_directories(output_dir: Path) -> dict[str, Path]:
    directories = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
    }
    ensure_directories(directories.values())
    return directories


def _build_benchmark_inputs(
    config: KernelBenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.rand(
        (config.batch_size, config.channels, config.height, config.width),
        dtype=torch.float32,
    )
    mean_tensor = torch.linspace(0.35, 0.55, steps=config.channels, dtype=torch.float32)
    std_tensor = torch.linspace(0.20, 0.35, steps=config.channels, dtype=torch.float32)
    return images, mean_tensor, std_tensor


def _pytorch_normalize(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean_view = mean.view(1, -1, 1, 1)
    std_view = std.view(1, -1, 1, 1)
    return (images - mean_view) / std_view


def _run_completed_benchmark(
    backend: str,
    device: torch.device,
    func,
    reference_output: torch.Tensor,
    bytes_processed: int,
    notes: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> BenchmarkRow:
    for _ in range(warmup_iterations):
        func()
        _synchronize(device)

    latencies_ms: list[float] = []
    for _ in range(benchmark_iterations):
        _synchronize(device)
        start = perf_counter()
        candidate = func()
        _synchronize(device)
        latencies_ms.append((perf_counter() - start) * 1000.0)

    final_output = func()
    _synchronize(device)
    max_abs_error = _max_abs_error(final_output, reference_output)
    if max_abs_error > 1e-5:
        raise ValueError(f"{backend} deviated from the PyTorch reference output.")

    median_latency = median(latencies_ms)
    return BenchmarkRow(
        backend=backend,
        device=str(device),
        status="completed",
        mean_latency_ms=round(mean(latencies_ms), 4),
        median_latency_ms=round(median_latency, 4),
        std_latency_ms=round(pstdev(latencies_ms), 4),
        effective_bandwidth_gbps=round(_effective_bandwidth(bytes_processed, median_latency), 4),
        speedup_vs_cpu=None,
        max_abs_error=round(max_abs_error, 8),
        notes=notes,
    )


def _run_optional_cuda_backend(
    backend: str,
    availability_fn,
    benchmark_fn,
    reference_output: torch.Tensor,
    device: torch.device,
    bytes_processed: int,
    notes: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> BenchmarkRow:
    available, status_message = availability_fn()
    if not available:
        return BenchmarkRow(
            backend=backend,
            device=str(device),
            status="skipped",
            mean_latency_ms=None,
            median_latency_ms=None,
            std_latency_ms=None,
            effective_bandwidth_gbps=None,
            speedup_vs_cpu=None,
            max_abs_error=None,
            notes=status_message,
        )

    try:
        return _run_completed_benchmark(
            backend=backend,
            device=device,
            func=benchmark_fn,
            reference_output=reference_output,
            bytes_processed=bytes_processed,
            notes=notes,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )
    except Exception as exc:  # pragma: no cover - depends on optional CUDA stack.
        return BenchmarkRow(
            backend=backend,
            device=str(device),
            status="failed",
            mean_latency_ms=None,
            median_latency_ms=None,
            std_latency_ms=None,
            effective_bandwidth_gbps=None,
            speedup_vs_cpu=None,
            max_abs_error=None,
            notes=f"{notes} Backend failed at runtime: {exc}",
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_abs_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.max(torch.abs(candidate - reference)).item())


def _bytes_processed(images: torch.Tensor) -> int:
    return images.numel() * images.element_size() * 2


def _effective_bandwidth(bytes_processed: int, median_latency_ms: float) -> float:
    seconds = median_latency_ms / 1000.0
    return bytes_processed / seconds / 1e9


def _apply_speedups(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    cpu_row = next(
        (row for row in rows if row.backend == "torch_cpu" and row.status == "completed"),
        None,
    )
    if cpu_row is None or cpu_row.median_latency_ms in (None, 0.0):
        return rows

    cpu_latency = cpu_row.median_latency_ms
    updated_rows: list[BenchmarkRow] = []
    for row in rows:
        speedup = None
        if row.status == "completed" and row.median_latency_ms:
            speedup = round(cpu_latency / row.median_latency_ms, 4)
        updated_rows.append(
            BenchmarkRow(
                backend=row.backend,
                device=row.device,
                status=row.status,
                mean_latency_ms=row.mean_latency_ms,
                median_latency_ms=row.median_latency_ms,
                std_latency_ms=row.std_latency_ms,
                effective_bandwidth_gbps=row.effective_bandwidth_gbps,
                speedup_vs_cpu=speedup,
                max_abs_error=row.max_abs_error,
                notes=row.notes,
            )
        )
    return updated_rows
