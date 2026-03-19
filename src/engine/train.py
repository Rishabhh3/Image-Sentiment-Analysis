import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch.nn as nn

logger = logging.getLogger(__name__)


class Trainer:
    """Production-level training engine for sentiment analysis model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        use_mixed_precision: bool = False,
    ):
        # Auto-detect best device for Mac
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        
        logger.info(f"Training on device: {device}")

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save model checkpoint."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(self, train_loader, val_loader, checkpoint_dir: Path = None) -> None:
        """Full training loop."""
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")