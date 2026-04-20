from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("fashion_mnist", "cifar10")
SUPPORTED_CLASSICAL_MODELS = (
    "logistic_regression",
    "svm",
    "random_forest",
)
SUPPORTED_PYTORCH_MODELS = ("simple_cnn", "resnet18")


@dataclass
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


@dataclass
class ClassicalExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    enabled_models: tuple[str, ...] = SUPPORTED_CLASSICAL_MODELS
    output_dir: Path = PROJECT_ROOT / "outputs"
    train_subset: int | None = None
    test_subset: int | None = None
    n_jobs: int = -1
    save_models: bool = True


@dataclass
class DeepLearningExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model_name: str = "simple_cnn"
    output_dir: Path = PROJECT_ROOT / "outputs"
    train_subset: int | None = None
    val_subset: int | None = None
    test_subset: int | None = None
    device: str = "auto"
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    early_stopping_patience: int = 3
    use_pretrained: bool = False
    freeze_backbone: bool = False
    save_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_PYTORCH_MODELS:
            raise ValueError(f"Unsupported PyTorch model: {self.model_name}")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.label_smoothing < 0.0:
            raise ValueError("label_smoothing cannot be negative.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.use_pretrained and self.model_name != "resnet18":
            raise ValueError("use_pretrained is only supported for the resnet18 model.")
        if self.freeze_backbone and self.model_name != "resnet18":
            raise ValueError("freeze_backbone is only supported for the resnet18 model.")
