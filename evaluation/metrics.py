from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_scores: np.ndarray | None = None,
) -> dict[str, object]:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_scores is not None:
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(
                    y_true,
                    y_scores,
                    multi_class="ovr",
                    average="weighted",
                )
            )
        except ValueError:
            metrics["roc_auc_ovr_weighted"] = None
    else:
        metrics["roc_auc_ovr_weighted"] = None

    return metrics


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(
                column_index,
                row_index,
                format(confusion[row_index, column_index], "d"),
                ha="center",
                va="center",
                color="white" if confusion[row_index, column_index] > threshold else "black",
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(
    history_rows: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    if not history_rows:
        return

    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_loss"] for row in history_rows]
    validation_loss = [row["validation_loss"] for row in history_rows]
    train_accuracy = [row["train_accuracy"] for row in history_rows]
    validation_accuracy = [row["validation_accuracy"] for row in history_rows]
    validation_f1 = [row["validation_f1_weighted"] for row in history_rows]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="Validation Loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, validation_accuracy, label="Validation Accuracy", linewidth=2)
    axes[1].plot(epochs, validation_f1, label="Validation F1", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy and F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
