from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def build_pytorch_model(
    model_name: str,
    num_classes: int,
    input_channels: int,
    use_pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet18":
        return _build_resnet18(
            num_classes=num_classes,
            input_channels=input_channels,
            use_pretrained=use_pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unsupported PyTorch model: {model_name}")


def _build_resnet18(
    num_classes: int,
    input_channels: int,
    use_pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    model = resnet18(weights=weights)

    if input_channels != model.conv1.in_channels:
        original_conv = model.conv1
        replacement_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if weights is not None:
                if input_channels == 1:
                    replacement_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                else:
                    averaged_weights = original_conv.weight.mean(dim=1, keepdim=True)
                    replacement_conv.weight.copy_(
                        averaged_weights.repeat(1, input_channels, 1, 1) / input_channels
                    )

        model.conv1 = replacement_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if input_channels != 3:
            for parameter in model.conv1.parameters():
                parameter.requires_grad = True

    return model
