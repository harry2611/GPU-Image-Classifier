from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("fashion_mnist", "cifar10")


@dataclass(slots=True)
class DatasetConfig:
    name: str = "fashion_mnist"
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    validation_size: float = 0.10
    random_state: int = 42
    batch_size: int = 128
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.name}")
        if not 0.0 < self.validation_size < 0.5:
            raise ValueError("validation_size must be between 0 and 0.5.")


@dataclass(slots=True)
class ClassicalExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    enabled_models: tuple[str, ...] = (
        "logistic_regression",
        "svm",
        "random_forest",
    )
    output_dir: Path = PROJECT_ROOT / "outputs"
    train_subset: int | None = None
    test_subset: int | None = None
    n_jobs: int = -1
    save_models: bool = True
