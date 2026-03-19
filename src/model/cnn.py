"""
CNN model architecture for Image Sentiment Analysis.

The architecture mirrors the original Keras model from the notebook:
    Conv(16) → MaxPool → Conv(32) → MaxPool → Conv(16) → MaxPool
    → Flatten → Dense(256) → Dense(num_classes)

Supports both binary (2 classes) and multi-class sentiment classification.
"""

import torch
import torch.nn as nn


class SentimentCNN(nn.Module):
    """Convolutional Neural Network for image sentiment classification."""

    def __init__(self, num_classes: int = 2, img_size: int = 224):
        """
        Args:
            num_classes: Number of output sentiment classes (default: 2).
            img_size:    Height/width of the input images (default: 224).
        """
        super().__init__()

        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1: 16 filters
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 filters
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16 filters
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Compute the flattened feature size dynamically
        self._flattened_size = self._get_flattened_size(img_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def _get_flattened_size(self, img_size: int) -> int:
        """Compute the flattened feature map size for a given input resolution."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.features(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 2, img_size: int = 224) -> SentimentCNN:
    """Convenience factory function."""
    return SentimentCNN(num_classes=num_classes, img_size=img_size)
