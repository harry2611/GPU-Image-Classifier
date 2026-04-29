from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn, optim

from data.dataset_manager import create_torch_dataloaders
from evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_history,
)
from models.pytorch_models import build_pytorch_model
from utils.config import DeepLearningExperimentConfig
from utils.io_utils import ensure_directories, write_csv, write_json
from utils.torch_utils import count_trainable_parameters, resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class NeuralSplitEvaluation:
    loss: float
    metrics: dict[str, object]
    inference_time_seconds: float
    confusion_matrix_path: str | None = None


def run_pytorch_training(config: DeepLearningExperimentConfig) -> dict[str, object]:
    seed_everything(config.dataset.random_state)
    output_paths = _prepare_output_directories(config.output_dir)
    dataloaders = create_torch_dataloaders(
        config.dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train_subset=config.train_subset,
        val_subset=config.val_subset,
        test_subset=config.test_subset,
        augment=True,
    )
    device = resolve_device(config.device)
    class_names = dataloaders.class_names
    num_classes = len(class_names)

    model = build_pytorch_model(
        model_name=config.model_name,
        num_classes=num_classes,
        input_channels=dataloaders.input_channels,
        use_pretrained=config.use_pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    LOGGER.info(
        "Training %s on %s with %d trainable parameters",
        config.model_name,
        device,
        count_trainable_parameters(model),
    )

    history_rows: list[dict[str, float]] = []
    best_epoch = 1
    best_validation_accuracy = float("-inf")
    best_validation_metrics: dict[str, object] | None = None
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch_index in range(1, config.epochs + 1):
        train_epoch = _train_one_epoch(
            model=model,
            data_loader=dataloaders.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_names=class_names,
            use_amp=use_amp,
        )
        validation_epoch = _evaluate_model(
            model=model,
            data_loader=dataloaders.val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )
        scheduler.step()

        history_row = {
            "epoch": epoch_index,
            "train_loss": round(train_epoch.loss, 4),
            "train_accuracy": round(train_epoch.metrics["accuracy"], 4),
            "train_f1_weighted": round(train_epoch.metrics["f1_weighted"], 4),
            "validation_loss": round(validation_epoch.loss, 4),
            "validation_accuracy": round(validation_epoch.metrics["accuracy"], 4),
            "validation_f1_weighted": round(validation_epoch.metrics["f1_weighted"], 4),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
        }
        history_rows.append(history_row)
        LOGGER.info(
            "Epoch %d/%d | train_acc=%.4f | val_acc=%.4f | val_f1=%.4f",
            epoch_index,
            config.epochs,
            train_epoch.metrics["accuracy"],
            validation_epoch.metrics["accuracy"],
            validation_epoch.metrics["f1_weighted"],
        )

        if validation_epoch.metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_epoch.metrics["accuracy"]
            best_validation_metrics = validation_epoch.metrics
            best_epoch = epoch_index
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if config.save_checkpoint:
                checkpoint_path = (
                    output_paths["models"]
                    / f"{config.dataset.name}_{config.model_name}_best.pt"
                )
                torch.save(
                    {
                        "epoch": epoch_index,
                        "model_state_dict": best_state_dict,
                        "model_name": config.model_name,
                        "dataset": config.dataset.name,
                        "class_names": class_names,
                    },
                    checkpoint_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                LOGGER.info("Early stopping triggered after epoch %d", epoch_index)
                break

    model.load_state_dict(best_state_dict)

    final_validation = _evaluate_model(
        model=model,
        data_loader=dataloaders.val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        split_name="validation",
        figure_dir=output_paths["figures"],
        model_name=config.model_name,
    )
    test_result = _evaluate_model(
        model=model,
        data_loader=dataloaders.test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        split_name="test",
        figure_dir=output_paths["figures"],
        model_name=config.model_name,
    )

    history_path = output_paths["metrics"] / f"{config.dataset.name}_{config.model_name}_history.csv"
    summary_path = output_paths["metrics"] / f"{config.dataset.name}_{config.model_name}_summary.csv"
    results_path = output_paths["metrics"] / f"{config.dataset.name}_{config.model_name}_results.json"
    training_curve_path = (
        output_paths["figures"] / f"{config.dataset.name}_{config.model_name}_training_curves.png"
    )

    write_csv(history_path, history_rows)
    write_csv(
        summary_path,
        [
            {
                "dataset": config.dataset.name,
                "model": config.model_name,
                "device": str(device),
                "epochs_completed": len(history_rows),
                "best_epoch": best_epoch,
                "trainable_parameters": count_trainable_parameters(model),
                "validation_accuracy": round(final_validation.metrics["accuracy"], 4),
                "validation_f1_weighted": round(final_validation.metrics["f1_weighted"], 4),
                "test_accuracy": round(test_result.metrics["accuracy"], 4),
                "test_f1_weighted": round(test_result.metrics["f1_weighted"], 4),
                "test_roc_auc_ovr_weighted": _round_optional_metric(
                    test_result.metrics["roc_auc_ovr_weighted"]
                ),
            }
        ],
    )
    plot_training_history(
        history_rows=history_rows,
        output_path=training_curve_path,
        title=f"{config.model_name.replace('_', ' ').title()} Training History",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset.name,
        "model_name": config.model_name,
        "use_pretrained": config.use_pretrained,
        "freeze_backbone": config.freeze_backbone,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history_rows),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": count_trainable_parameters(model),
        "class_names": class_names,
        "split_sizes": {
            "train": len(dataloaders.train_loader.dataset),
            "validation": len(dataloaders.val_loader.dataset),
            "test": len(dataloaders.test_loader.dataset),
        },
        "history": history_rows,
        "best_validation_metrics": best_validation_metrics,
        "final_validation": asdict(final_validation),
        "test": asdict(test_result),
        "artifacts": {
            "history_csv": str(history_path),
            "summary_csv": str(summary_path),
            "training_curve": str(training_curve_path),
        },
    }
    write_json(results_path, payload)
    LOGGER.info("Saved PyTorch results to %s", results_path)
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


def _train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_names: list[str],
    use_amp: bool,
) -> NeuralSplitEvaluation:
    model.train()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        )
        with autocast_context:
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits.detach(), dim=1)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )
    return NeuralSplitEvaluation(
        loss=total_loss / max(sample_count, 1),
        metrics=metrics,
        inference_time_seconds=0.0,
    )


def _evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    split_name: str | None = None,
    figure_dir: Path | None = None,
    model_name: str | None = None,
) -> NeuralSplitEvaluation:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    inference_start = perf_counter()
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            logits = model(images)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    inference_time_seconds = perf_counter() - inference_start

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_scores = np.concatenate(all_probabilities)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        class_names=class_names,
    )

    confusion_matrix_path: str | None = None
    if split_name is not None and figure_dir is not None and model_name is not None:
        confusion_path = figure_dir / f"{model_name}_{split_name}_confusion_matrix.png"
        plot_confusion_matrix(
            confusion=metrics["confusion_matrix"],
            class_names=class_names,
            output_path=confusion_path,
            title=f"{model_name.replace('_', ' ').title()} - {split_name.title()} Confusion Matrix",
        )
        confusion_matrix_path = str(confusion_path)

    return NeuralSplitEvaluation(
        loss=total_loss / max(sample_count, 1),
        metrics=metrics,
        inference_time_seconds=inference_time_seconds,
        confusion_matrix_path=confusion_matrix_path,
    )


def _round_optional_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
