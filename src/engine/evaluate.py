"""
Evaluation engine for the Image Sentiment Analysis model.

Computes loss, accuracy, precision, and recall on a held-out data split,
mirroring the metrics used in the original TensorFlow notebook.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on the provided DataLoader.

    Args:
        model:     The neural network to evaluate.
        loader:    DataLoader for the validation / test split.
        criterion: Loss function (e.g. CrossEntropyLoss).
        device:    Target device (CPU / CUDA).

    Returns:
        A dict with keys: 'loss', 'accuracy', 'precision', 'recall'.
        Precision and recall are macro-averaged across all classes.
    """
    model.eval()

    running_loss = 0.0
    num_classes: int | None = None
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            if num_classes is None:
                num_classes = outputs.size(1)

            _, predicted = outputs.max(dim=1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    total = sum(t.numel() for t in all_labels)
    avg_loss = running_loss / total

    preds = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)

    accuracy = preds.eq(labels_cat).sum().item() / total

    # Macro-averaged precision and recall (use model output classes, not label max)
    if num_classes is None:
        num_classes = int(labels_cat.max().item()) + 1
    precision_list = []
    recall_list = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (labels_cat == cls)).sum().item()
        fp = ((preds == cls) & (labels_cat != cls)).sum().item()
        fn = ((preds != cls) & (labels_cat == cls)).sum().item()

        precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recall_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    precision = sum(precision_list) / num_classes
    recall = sum(recall_list) / num_classes

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
