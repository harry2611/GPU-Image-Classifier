from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from data.dataset_manager import prepare_classical_dataset
from evaluation.metrics import compute_classification_metrics, plot_confusion_matrix
from models.classical_models import build_classical_models
from utils.config import ClassicalExperimentConfig
from utils.io_utils import ensure_directories, write_csv, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SplitEvaluation:
    metrics: dict[str, object]
    inference_time_seconds: float
    confusion_matrix_path: str


@dataclass(slots=True)
class ModelEvaluationResult:
    model_name: str
    train_time_seconds: float
    validation: SplitEvaluation
    test: SplitEvaluation


def run_classical_baselines(config: ClassicalExperimentConfig) -> dict[str, object]:
    output_paths = _prepare_output_directories(config.output_dir)
    dataset = prepare_classical_dataset(
        config.dataset,
        train_subset=config.train_subset,
        test_subset=config.test_subset,
    )
    LOGGER.info(
        "Prepared %s with train=%d, val=%d, test=%d, feature_dim=%d",
        config.dataset.name,
        len(dataset.y_train),
        len(dataset.y_val),
        len(dataset.y_test),
        dataset.X_train.shape[1],
    )

    available_models = build_classical_models(
        random_state=config.dataset.random_state,
        n_jobs=config.n_jobs,
    )
    selected_models = {
        name: estimator
        for name, estimator in available_models.items()
        if name in config.enabled_models
    }

    detailed_results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model_name, estimator in selected_models.items():
        LOGGER.info("Training classical baseline: %s", model_name)
        training_start = perf_counter()
        estimator.fit(dataset.X_train, dataset.y_train)
        train_time_seconds = perf_counter() - training_start

        if config.save_models:
            model_path = output_paths["models"] / f"{config.dataset.name}_{model_name}.joblib"
            joblib.dump(estimator, model_path)

        validation_result = _evaluate_split(
            estimator=estimator,
            split_name="validation",
            X=dataset.X_val,
            y=dataset.y_val,
            class_names=dataset.class_names,
            figure_dir=output_paths["figures"],
            model_name=model_name,
        )
        test_result = _evaluate_split(
            estimator=estimator,
            split_name="test",
            X=dataset.X_test,
            y=dataset.y_test,
            class_names=dataset.class_names,
            figure_dir=output_paths["figures"],
            model_name=model_name,
        )

        result = ModelEvaluationResult(
            model_name=model_name,
            train_time_seconds=train_time_seconds,
            validation=validation_result,
            test=test_result,
        )
        detailed_results.append(asdict(result))
        summary_rows.append(
            {
                "model": model_name,
                "train_time_seconds": round(train_time_seconds, 4),
                "validation_accuracy": round(validation_result.metrics["accuracy"], 4),
                "validation_f1_weighted": round(validation_result.metrics["f1_weighted"], 4),
                "validation_roc_auc_ovr_weighted": _round_optional_metric(
                    validation_result.metrics["roc_auc_ovr_weighted"]
                ),
                "test_accuracy": round(test_result.metrics["accuracy"], 4),
                "test_f1_weighted": round(test_result.metrics["f1_weighted"], 4),
                "test_roc_auc_ovr_weighted": _round_optional_metric(
                    test_result.metrics["roc_auc_ovr_weighted"]
                ),
                "validation_inference_time_seconds": round(
                    validation_result.inference_time_seconds, 4
                ),
                "test_inference_time_seconds": round(test_result.inference_time_seconds, 4),
            }
        )

    best_model = max(summary_rows, key=lambda row: row["validation_accuracy"])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset.name,
        "image_shape": list(dataset.image_shape),
        "class_names": dataset.class_names,
        "split_sizes": {
            "train": len(dataset.y_train),
            "validation": len(dataset.y_val),
            "test": len(dataset.y_test),
        },
        "best_model_by_validation_accuracy": best_model["model"],
        "results": detailed_results,
    }

    write_json(output_paths["metrics"] / "classical_results.json", payload)
    write_csv(output_paths["metrics"] / "classical_summary.csv", summary_rows)

    LOGGER.info("Best validation model: %s", best_model["model"])
    LOGGER.info("Saved summary table to %s", output_paths["metrics"] / "classical_summary.csv")

    return payload


def _prepare_output_directories(output_dir: Path) -> dict[str, Path]:
    directories = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "models": output_dir / "models",
    }
    ensure_directories(directories.values())
    return directories


def _evaluate_split(
    estimator,
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    figure_dir: Path,
    model_name: str,
) -> SplitEvaluation:
    inference_start = perf_counter()
    y_pred = estimator.predict(X)
    y_scores = _extract_scores(estimator, X)
    inference_time_seconds = perf_counter() - inference_start

    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        y_scores=y_scores,
        class_names=class_names,
    )

    confusion_matrix_path = figure_dir / f"{model_name}_{split_name}_confusion_matrix.png"
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=confusion_matrix_path,
        title=f"{model_name.replace('_', ' ').title()} - {split_name.title()} Confusion Matrix",
    )

    return SplitEvaluation(
        metrics=metrics,
        inference_time_seconds=inference_time_seconds,
        confusion_matrix_path=str(confusion_matrix_path),
    )


def _extract_scores(estimator, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def _round_optional_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
