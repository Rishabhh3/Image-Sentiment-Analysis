import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import CustomException, logger


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full pass over *loader* and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate *model* on *loader* without gradient computation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="    val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    class_names: List[str],
) -> nn.Module:
    """Full training loop with early stopping and TensorBoard logging.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        config: Training configuration dict containing keys such as
            ``epochs``, ``learning_rate``, ``weight_decay``, ``save_dir``,
            ``log_dir`` and ``early_stopping_patience``.
        class_names: Ordered list of class label strings.

    Returns:
        The model loaded with the best weights found during training.
    """
    try:
        train_cfg = config.get("train", config)
        epochs: int = train_cfg.get("epochs", 20)
        lr: float = train_cfg.get("learning_rate", 1e-3)
        weight_decay: float = train_cfg.get("weight_decay", 1e-4)
        save_dir: str = train_cfg.get("save_dir", "models")
        log_dir: str = train_cfg.get("log_dir", "logs")
        patience: int = train_cfg.get("early_stopping_patience", 5)

        os.makedirs(save_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        writer = SummaryWriter(log_dir=log_dir)

        best_val_loss = float("inf")
        best_model_path = os.path.join(save_dir, "best_model.pth")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            elapsed = time.time() - start
            logger.info(
                "Epoch [%d/%d] | train_loss: %.4f | train_acc: %.4f "
                "| val_loss: %.4f | val_acc: %.4f | %.1fs",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                elapsed,
            )
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
                f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )

            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars(
                "Accuracy", {"train": train_acc, "val": val_acc}, epoch
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "class_names": class_names,
                    },
                    best_model_path,
                )
                logger.info("Saved best model → %s", best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        patience,
                    )
                    print(f"\nEarly stopping after {patience} epochs without improvement.")
                    break

        writer.close()

        # Load best weights before returning
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\nBest model (val_loss={best_val_loss:.4f}) saved to '{best_model_path}'")
        return model

    except Exception as e:
        raise CustomException(str(e), sys) from e
