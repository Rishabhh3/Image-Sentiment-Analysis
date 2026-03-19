import torch
import torch.nn as nn


class SentimentCNN(nn.Module):
    """Custom CNN for image sentiment classification.

    Architecture (mirrors the Keras model in the notebook):

    * 3 × (Conv2d → BN → ReLU → MaxPool2d)
    * AdaptiveAvgPool to a fixed 7×7 spatial size
    * Flatten
    * Linear(256) → ReLU → Dropout
    * Linear(num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Adaptive pool to a fixed 7×7 → robust to any input resolution
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)



# ---------------------------------------------------------------------------
# Transfer-learning backbones
# ---------------------------------------------------------------------------


def _make_transfer_model(backbone: str, num_classes: int, dropout: float) -> nn.Module:
    """Return a pretrained backbone with a custom classification head."""
    import torchvision.models as models

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            "Choose from: 'custom_cnn', 'resnet18', 'efficientnet_b0'."
        )
    return model


def build_model(
    backbone: str = "custom_cnn",
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = False,
) -> nn.Module:
    """Factory function that returns the requested model.

    Args:
        backbone: Architecture name.  One of ``"custom_cnn"``,
            ``"resnet18"``, or ``"efficientnet_b0"``.
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the final linear layer.
        pretrained: When ``True`` and backbone is not ``"custom_cnn"``,
            load ImageNet pre-trained weights.

    Returns:
        Initialised ``nn.Module``.
    """
    if backbone == "custom_cnn":
        return SentimentCNN(num_classes=num_classes, dropout=dropout)
    if pretrained:
        return _make_transfer_model(backbone, num_classes, dropout)
    # Untrained transfer backbone
    import torchvision.models as models

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'.")
    return model
