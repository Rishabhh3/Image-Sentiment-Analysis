"""
train.py — Entry point for training the Image Sentiment CNN.

Usage:
    python train.py

The script will:
1. Load the dataset from the path configured in src/config.py
2. Build the CNN model
3. Train for the configured number of epochs
4. Save the best model checkpoint to models/best_model.pth
5. Log loss and accuracy to the console and to logs/
"""

import os
import sys

import torch
import torch.nn as nn

from src.config import DATA_CONFIG
from src.data.loader import get_data_loaders
from src.engine.evaluate import evaluate
from src.engine.train import train_one_epoch
from src.model.cnn import build_model
from src.utils import logger

# ---------------------------------------------------------------------------
# Training hyper-parameters (override via DATA_CONFIG or edit here)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_SAVE_DIR = "models"
CHECKPOINT_NAME = "best_model.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    config = {"data": DATA_CONFIG}
    train_loader, val_loader, class_names = get_data_loaders(config)
    logger.info(f"Classes: {class_names}")
    print(f"Classes: {class_names}")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = build_model(
        num_classes=len(class_names),
        img_size=DATA_CONFIG["img_size"],
    ).to(device)
    logger.info(f"Model:\n{model}")

    # ------------------------------------------------------------------
    # 3. Loss and optimiser
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        log_msg = (
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}"
        )
        print(log_msg)
        logger.info(log_msg)

        # Save the best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_NAME)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "class_names": class_names,
                    "img_size": DATA_CONFIG["img_size"],
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model to {checkpoint_path}")
            print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")

    print("\nTraining complete.")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
