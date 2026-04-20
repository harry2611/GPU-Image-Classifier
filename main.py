from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import (
    ClassicalExperimentConfig,
    DatasetConfig,
    DeepLearningExperimentConfig,
    KernelBenchmarkConfig,
    SUPPORTED_BENCHMARK_OPERATIONS,
    SUPPORTED_CLASSICAL_MODELS,
    SUPPORTED_DATASETS,
    SUPPORTED_PYTORCH_MODELS,
)
from utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Image Classification System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classical_parser = subparsers.add_parser(
        "classical-baselines",
        help="Run scikit-learn baselines on flattened image inputs.",
    )
    classical_parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="fashion_mnist",
        help="Dataset to download and evaluate.",
    )
    classical_parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_CLASSICAL_MODELS,
        default=list(SUPPORTED_CLASSICAL_MODELS),
        help="One or more classical models to run.",
    )
    classical_parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Optional stratified subset size for faster local experiments.",
    )
    classical_parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Optional stratified subset size for test-set evaluation.",
    )
    classical_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
        help="Fraction of the training split to reserve for validation.",
    )
    classical_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for split creation and estimator reproducibility.",
    )
    classical_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where metrics, models, and plots are stored.",
    )
    classical_parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU parallelism for models that support it.",
    )
    classical_parser.add_argument(
        "--skip-model-saving",
        action="store_true",
        help="Do not serialize trained sklearn models.",
    )

    pytorch_parser = subparsers.add_parser(
        "train-pytorch",
        help="Train a PyTorch image classifier with GPU-aware execution when available.",
    )
    pytorch_parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="fashion_mnist",
        help="Dataset to download and train on.",
    )
    pytorch_parser.add_argument(
        "--model",
        choices=SUPPORTED_PYTORCH_MODELS,
        default="simple_cnn",
        help="PyTorch architecture to train.",
    )
    pytorch_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )
    pytorch_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for train, validation, and test loaders.",
    )
    pytorch_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader instances.",
    )
    pytorch_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial AdamW learning rate.",
    )
    pytorch_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used by AdamW.",
    )
    pytorch_parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing value for cross-entropy.",
    )
    pytorch_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many non-improving validation epochs.",
    )
    pytorch_parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Use auto, cpu, cuda, mps, or a device string like cuda:0.",
    )
    pytorch_parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Optional stratified subset size for the train split.",
    )
    pytorch_parser.add_argument(
        "--val-subset",
        type=int,
        default=None,
        help="Optional stratified subset size for the validation split.",
    )
    pytorch_parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Optional stratified subset size for the test split.",
    )
    pytorch_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
        help="Fraction of the training split to reserve for validation.",
    )
    pytorch_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for split creation and training reproducibility.",
    )
    pytorch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where checkpoints, plots, and metrics are stored.",
    )
    pytorch_parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use torchvision pretrained weights when training ResNet18.",
    )
    pytorch_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the ResNet18 backbone and only train the classifier head.",
    )
    pytorch_parser.add_argument(
        "--skip-checkpoint-saving",
        action="store_true",
        help="Do not persist the best checkpoint to disk.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-kernels",
        help="Benchmark CPU, PyTorch CUDA, custom CUDA, and Triton kernels.",
    )
    benchmark_parser.add_argument(
        "--operation",
        choices=SUPPORTED_BENCHMARK_OPERATIONS,
        default="image_normalization",
        help="Kernel operation to benchmark.",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size of the synthetic image tensor.",
    )
    benchmark_parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of image channels in the synthetic tensor.",
    )
    benchmark_parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Height of the synthetic image tensor.",
    )
    benchmark_parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Width of the synthetic image tensor.",
    )
    benchmark_parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Warmup iterations to exclude from timing.",
    )
    benchmark_parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Measured iterations for each backend.",
    )
    benchmark_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when generating synthetic benchmark inputs.",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where benchmark artifacts are stored.",
    )
    benchmark_parser.add_argument(
        "--verbose-backend-loading",
        action="store_true",
        help="Show verbose output when loading CUDA or Triton backends.",
    )

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "classical-baselines":
        try:
            from training.classical_pipeline import run_classical_baselines
        except ModuleNotFoundError as exc:
            parser.exit(
                status=1,
                message=(
                    f"Missing dependency '{exc.name}'. Install the project dependencies first "
                    "with `pip install -e .`.\n"
                ),
            )
        dataset_config = DatasetConfig(
            name=args.dataset,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
        experiment_config = ClassicalExperimentConfig(
            dataset=dataset_config,
            enabled_models=tuple(args.models),
            output_dir=args.output_dir,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            n_jobs=args.n_jobs,
            save_models=not args.skip_model_saving,
        )
        run_classical_baselines(experiment_config)
        return

    if args.command == "train-pytorch":
        try:
            from training.pytorch_pipeline import run_pytorch_training
        except ModuleNotFoundError as exc:
            parser.exit(
                status=1,
                message=(
                    f"Missing dependency '{exc.name}'. Install the project dependencies first "
                    "with `pip install -e .`.\n"
                ),
            )
        dataset_config = DatasetConfig(
            name=args.dataset,
            validation_size=args.validation_size,
            random_state=args.random_state,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        experiment_config = DeepLearningExperimentConfig(
            dataset=dataset_config,
            model_name=args.model,
            output_dir=args.output_dir,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.early_stopping_patience,
            use_pretrained=args.use_pretrained,
            freeze_backbone=args.freeze_backbone,
            save_checkpoint=not args.skip_checkpoint_saving,
        )
        run_pytorch_training(experiment_config)
        return

    if args.command == "benchmark-kernels":
        try:
            from benchmarking.image_normalization_benchmark import (
                run_image_normalization_benchmark,
            )
        except ModuleNotFoundError as exc:
            parser.exit(
                status=1,
                message=(
                    f"Missing dependency '{exc.name}'. Install the project dependencies first "
                    "with `pip install -e .`.\n"
                ),
            )

        benchmark_config = KernelBenchmarkConfig(
            operation=args.operation,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            channels=args.channels,
            height=args.height,
            width=args.width,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
            random_state=args.random_state,
            verbose_backend_loading=args.verbose_backend_loading,
        )

        if benchmark_config.operation == "image_normalization":
            run_image_normalization_benchmark(benchmark_config)


if __name__ == "__main__":
    main()
