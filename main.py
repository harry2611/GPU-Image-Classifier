from __future__ import annotations

import argparse
from pathlib import Path

from models.classical_models import SUPPORTED_CLASSICAL_MODELS
from utils.config import ClassicalExperimentConfig, DatasetConfig, SUPPORTED_DATASETS
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


if __name__ == "__main__":
    main()
