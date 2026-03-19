import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import CustomException, logger


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device = None,
) -> Dict[str, float]:
    """Evaluate *model* on *loader* and return a metrics dictionary.

    Args:
        model: Trained model.
        loader: DataLoader for the evaluation set.
        class_names: Ordered list of class label strings.
        device: torch device to use.  Auto-detected when ``None``.

    Returns:
        Dictionary with keys ``loss``, ``accuracy``, ``precision``,
        ``recall`` and ``f1`` (macro-averaged).
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        n = len(all_labels)
        avg_loss = total_loss / n

        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        macro = report["macro avg"]

        metrics = {
            "loss": avg_loss,
            "accuracy": report["accuracy"],
            "precision": macro["precision"],
            "recall": macro["recall"],
            "f1": macro["f1-score"],
        }

        print("\n── Evaluation Results ──────────────────────────")
        print(f"  Loss      : {avg_loss:.4f}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=class_names,
                zero_division=0,
            )
        )
        print("Confusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        header = "        " + "  ".join(f"{c:>10}" for c in class_names)
        print(header)
        for row_name, row in zip(class_names, cm):
            print(f"{row_name:>8}  " + "  ".join(f"{v:>10}" for v in row))

        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    except Exception as e:
        raise CustomException(str(e), sys) from e
