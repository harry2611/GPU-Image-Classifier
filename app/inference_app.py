"""Streamlit inference demo for the GPU Image Classifier project.

Run with:
    streamlit run app/inference_app.py

Loads any saved PyTorch checkpoints from ``outputs/models/`` and lets the user
upload an image to get top-3 predictions. The model is selected from a
sidebar dropdown, and preprocessing matches the per-channel mean/std that the
training pipeline used so predictions are on the same scale as during training.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from data.dataset_manager import DATASET_STATS
from models.pytorch_models import build_pytorch_model
from utils.torch_utils import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main() -> None:
    st.set_page_config(
        page_title="GPU Image Classifier — Demo",
        page_icon=":mag:",
        layout="centered",
    )
    st.title("GPU Image Classifier")
    st.caption(
        "Upload an image and the trained model will predict the class. "
        "Models loaded from `outputs/models/`."
    )

    checkpoints = _discover_checkpoints(MODELS_DIR)
    if not checkpoints:
        st.error(
            "No PyTorch checkpoints found under `outputs/models/`. Train a model first, "
            "for example: `python3 main.py train-pytorch --dataset fashion_mnist --model simple_cnn --epochs 5`."
        )
        return

    with st.sidebar:
        st.header("Model")
        checkpoint_label = st.selectbox(
            "Checkpoint",
            options=list(checkpoints.keys()),
            help="Pick which trained model to use for inference.",
        )
        device_choice = st.selectbox(
            "Device",
            options=["auto", "cpu", "mps", "cuda"],
            index=0,
            help="`auto` picks CUDA, then MPS, then CPU.",
        )
        st.divider()
        st.markdown(
            "**Training stats used for normalization** match the dataset the "
            "checkpoint was trained on:"
        )

    checkpoint_path = checkpoints[checkpoint_label]
    model, metadata, device = _load_model_and_metadata(checkpoint_path, device_choice)

    with st.sidebar:
        st.json(
            {
                "model": metadata["model_name"],
                "dataset": metadata["dataset"],
                "device": str(device),
                "classes": len(metadata["class_names"]),
                "input_channels": metadata["input_channels"],
                "image_size": metadata["image_size"],
            },
            expanded=False,
        )

    st.subheader("1) Upload an image")
    uploaded = st.file_uploader(
        "PNG, JPG, or BMP",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info(
            "Tip: Fashion-MNIST models work best on grayscale clothing images at "
            "28x28; ResNet18 trained on CIFAR-10 expects natural RGB images."
        )
        return

    raw_image = Image.open(io.BytesIO(uploaded.read()))
    st.image(raw_image, caption="Uploaded image", width=240)

    st.subheader("2) Prediction")
    with st.spinner("Running inference..."):
        prediction = _predict(model=model, metadata=metadata, image=raw_image, device=device)

    top_class = prediction["top_class"]
    top_probability = prediction["top_probability"]

    st.metric(
        label="Top prediction",
        value=top_class,
        delta=f"{top_probability * 100:.1f}% confidence",
    )

    st.write("Top 3 predictions")
    top_table = {
        "class": [item["class"] for item in prediction["top_k"]],
        "probability": [round(item["probability"], 4) for item in prediction["top_k"]],
    }
    st.dataframe(top_table, hide_index=True, use_container_width=True)
    st.bar_chart(
        data={"probability": top_table["probability"]},
        x_label="class",
        y_label="probability",
    )


def _discover_checkpoints(models_dir: Path) -> dict[str, Path]:
    if not models_dir.exists():
        return {}
    return {path.name: path for path in sorted(models_dir.glob("*.pt"))}


@st.cache_resource(show_spinner=False)
def _load_model_and_metadata(
    checkpoint_path: Path,
    device_choice: str,
) -> tuple[torch.nn.Module, dict[str, object], torch.device]:
    device = resolve_device(device_choice)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_name = state["model_name"]
    dataset_name = state["dataset"]
    class_names = list(state["class_names"])

    stats = DATASET_STATS[dataset_name]
    input_channels = len(stats["mean"])
    image_size = 28 if dataset_name == "fashion_mnist" else 32

    model = build_pytorch_model(
        model_name=model_name,
        num_classes=len(class_names),
        input_channels=input_channels,
        use_pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)

    metadata = {
        "model_name": model_name,
        "dataset": dataset_name,
        "class_names": class_names,
        "input_channels": input_channels,
        "image_size": image_size,
        "stats": stats,
    }
    return model, metadata, device


def _predict(
    model: torch.nn.Module,
    metadata: dict[str, object],
    image: Image.Image,
    device: torch.device,
) -> dict[str, object]:
    image_tensor = _preprocess_image(
        image=image,
        input_channels=int(metadata["input_channels"]),
        image_size=int(metadata["image_size"]),
        mean=metadata["stats"]["mean"],
        std=metadata["stats"]["std"],
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(-probabilities)[:3]
    class_names = list(metadata["class_names"])
    top_k = [
        {
            "class": class_names[int(index)],
            "probability": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]
    return {
        "top_class": top_k[0]["class"],
        "top_probability": top_k[0]["probability"],
        "top_k": top_k,
    }


def _preprocess_image(
    image: Image.Image,
    input_channels: int,
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    if input_channels == 1:
        prepared = image.convert("L")
    else:
        prepared = image.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
    return transform(prepared)


if __name__ == "__main__":
    main()
