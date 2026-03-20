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

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        use_mixed_precision: bool = False, # mixed precision training can speed up training and reduce memory usage by using 16-bit floating point numbers instead of 32-bit, but it can sometimes cause instability, so it's optional and disabled by default
    ):
        # Auto-detect best device for Mac
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None 
        # GradScaler helps prevent underflow when using mixed precision by dynamically scaling the loss value, which allows us to take advantage of the speed and memory benefits of mixed precision without sacrificing stability.
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate) 
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        ) # if model stops imporving for 3 epochs, reduce learning rate by half to help model converge to a better minimum, and verbose=True will print a message every time the learning rate is reduced
        self.criterion = nn.CrossEntropyLoss()
        # used in binary classification, and heavily penalizes the model when it is confidently wrong
        self.best_val_loss = float("inf")
        
        logger.info(f"Training on device: {device}")


    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        ''' We are using tqdm to create a progress bar for the training loop, which gives us a visual indication of how long each epoch is taking
          and how many batches have been processed. '''
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()              # zero out the gradients from the previous batch
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)  # compute the loss between the model's predictions and the true labels for this batch
            loss.backward()                         # backpropagate the loss to compute gradients for all model parameters
            self.optimizer.step()                   # update the model parameters using the computed gradients and the Adam optimization algorithm

            total_loss += loss.item()           
            _, predicted = torch.max(outputs.data, 1)       # get predicted class label
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

        with torch.no_grad():   # locks model in val mode, turning off dropout and batch norm updates
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
                "model_state_dict": self.model.state_dict(),    # saves the learned parameters of the model, which can be loaded later to restore the model's state
                "optimizer_state_dict": self.optimizer.state_dict(), # saves the state of the optimizer, including the values of the learning rate
                 # and momentum, which allows to resume training with the same optimization settings if we need to stop and restart training
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

            if val_metrics["loss"] < self.best_val_loss: # if the validation loss has improved, save the model checkpoint and update the best validation loss
                self.best_val_loss = val_metrics["loss"]
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")