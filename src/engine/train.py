"""
Training engine for one epoch of the Image Sentiment Analysis model.

Separating the training loop into its own module makes it easy to swap
optimisers, schedulers or loss functions without touching the rest of the
codebase.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run a single training epoch.

    Args:
        model:     The neural network to train.
        loader:    DataLoader for the training split.
        optimizer: Optimiser (e.g. Adam).
        criterion: Loss function (e.g. CrossEntropyLoss).
        device:    Target device (CPU / CUDA).

    Returns:
        avg_loss (float): Mean loss over all mini-batches.
        accuracy (float): Fraction of correctly classified samples (0-1).
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
