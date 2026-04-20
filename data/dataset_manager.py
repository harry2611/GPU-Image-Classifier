from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.config import DatasetConfig
from utils.io_utils import ensure_directories

DATASET_REGISTRY = {
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

DATASET_STATS = {
    "fashion_mnist": {
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}


@dataclass
class ClassicalDatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    image_shape: tuple[int, ...]


@dataclass
class TorchDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    image_shape: tuple[int, ...]
    input_channels: int


def build_torch_transform(dataset_name: str) -> transforms.Compose:
    return build_torch_transform_for_split(dataset_name, train=False)


def build_torch_transform_for_split(dataset_name: str, train: bool) -> transforms.Compose:
    stats = DATASET_STATS[dataset_name]
    transform_steps: list[object] = []

    if train and dataset_name == "cifar10":
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )
    return transforms.Compose(transform_steps)


def download_dataset(config: DatasetConfig):
    dataset_cls = DATASET_REGISTRY[config.name]
    train_dataset = dataset_cls(root=str(config.data_dir), train=True, download=True)
    test_dataset = dataset_cls(root=str(config.data_dir), train=False, download=True)
    return train_dataset, test_dataset


def prepare_classical_dataset(
    config: DatasetConfig,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> ClassicalDatasetSplits:
    train_dataset, test_dataset = download_dataset(config)
    train_images, train_labels = _extract_arrays(train_dataset)
    test_images, test_labels = _extract_arrays(test_dataset)

    train_indices, val_indices = load_or_create_split_indices(config, train_labels)
    X_train = train_images[train_indices]
    y_train = train_labels[train_indices]
    X_val = train_images[val_indices]
    y_val = train_labels[val_indices]
    X_test = test_images
    y_test = test_labels

    X_train, y_train = _maybe_subsample(
        X_train,
        y_train,
        subset_size=train_subset,
        random_state=config.random_state,
    )
    X_test, y_test = _maybe_subsample(
        X_test,
        y_test,
        subset_size=test_subset,
        random_state=config.random_state + 1,
    )

    class_names = list(getattr(train_dataset, "classes", [])) or [
        str(label) for label in sorted(np.unique(train_labels))
    ]
    image_shape = tuple(X_train.shape[1:])

    return ClassicalDatasetSplits(
        X_train=_flatten_and_scale_pixels(X_train),
        y_train=y_train,
        X_val=_flatten_and_scale_pixels(X_val),
        y_val=y_val,
        X_test=_flatten_and_scale_pixels(X_test),
        y_test=y_test,
        class_names=class_names,
        image_shape=image_shape,
    )


def create_torch_dataloaders(
    config: DatasetConfig,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    train_subset: int | None = None,
    val_subset: int | None = None,
    test_subset: int | None = None,
    augment: bool = True,
) -> TorchDataLoaders:
    dataset_cls = DATASET_REGISTRY[config.name]
    train_transform = build_torch_transform_for_split(config.name, train=augment)
    eval_transform = build_torch_transform_for_split(config.name, train=False)

    train_dataset = dataset_cls(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = dataset_cls(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = dataset_cls(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_indices, val_indices = load_or_create_split_indices(
        config,
        np.asarray(train_dataset.targets),
    )
    train_targets = np.asarray(train_dataset.targets)
    test_targets = np.asarray(test_dataset.targets)
    train_indices = _maybe_subsample_indices(
        train_indices,
        train_targets[train_indices],
        subset_size=train_subset,
        random_state=config.random_state,
    )
    val_indices = _maybe_subsample_indices(
        val_indices,
        train_targets[val_indices],
        subset_size=val_subset,
        random_state=config.random_state + 1,
    )
    test_indices = _maybe_subsample_indices(
        np.arange(len(test_targets)),
        test_targets,
        subset_size=test_subset,
        random_state=config.random_state + 2,
    )

    effective_batch_size = batch_size or config.batch_size
    effective_num_workers = config.num_workers if num_workers is None else num_workers
    effective_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()),
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices.tolist()),
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices.tolist()),
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
    )

    sample_shape = tuple(_extract_sample_image_shape(test_loader.dataset))
    class_names = list(getattr(train_dataset, "classes", [])) or [
        str(label) for label in sorted(np.unique(train_targets))
    ]

    return TorchDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        image_shape=sample_shape,
        input_channels=sample_shape[0],
    )


def load_or_create_split_indices(
    config: DatasetConfig,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ensure_directories([config.processed_dir])
    split_path = _split_cache_path(config)

    if split_path.exists():
        payload = json.loads(split_path.read_text())
        return np.asarray(payload["train_indices"]), np.asarray(payload["val_indices"])

    index_array = np.arange(len(labels))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_size,
        random_state=config.random_state,
    )
    train_indices, val_indices = next(splitter.split(index_array, labels))

    split_path.write_text(
        json.dumps(
            {
                "dataset": config.name,
                "validation_size": config.validation_size,
                "random_state": config.random_state,
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
            },
            indent=2,
        )
    )
    return train_indices, val_indices


def _split_cache_path(config: DatasetConfig) -> Path:
    validation_tag = str(config.validation_size).replace(".", "_")
    return config.processed_dir / (
        f"{config.name}_val_{validation_tag}_seed_{config.random_state}_split.json"
    )


def _extract_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    images = dataset.data
    labels = dataset.targets

    image_array = images.numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    label_array = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    return image_array, label_array


def _extract_sample_image_shape(dataset) -> tuple[int, ...]:
    sample_image, _ = dataset[0]
    return tuple(sample_image.shape)


def _flatten_and_scale_pixels(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0


def _maybe_subsample(
    images: np.ndarray,
    labels: np.ndarray,
    subset_size: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subset_size is None or subset_size >= len(labels):
        return images, labels
    if subset_size <= 0:
        raise ValueError("subset_size must be positive when provided.")
    if subset_size < len(np.unique(labels)):
        raise ValueError(
            "subset_size must be at least the number of classes for stratified sampling."
        )

    index_array = np.arange(len(labels))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=subset_size,
        random_state=random_state,
    )
    selected_indices, _ = next(splitter.split(index_array, labels))
    return images[selected_indices], labels[selected_indices]


def _maybe_subsample_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    subset_size: int | None,
    random_state: int,
) -> np.ndarray:
    if subset_size is None or subset_size >= len(indices):
        return indices
    if subset_size <= 0:
        raise ValueError("subset_size must be positive when provided.")
    if subset_size < len(np.unique(labels)):
        raise ValueError(
            "subset_size must be at least the number of classes for stratified sampling."
        )

    index_positions = np.arange(len(indices))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=subset_size,
        random_state=random_state,
    )
    selected_positions, _ = next(splitter.split(index_positions, labels))
    return indices[selected_positions]
