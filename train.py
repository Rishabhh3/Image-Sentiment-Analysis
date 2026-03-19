"""CLI entry point for training the Image Sentiment Analysis model.

Usage::

    python train.py
    python train.py --data_path Data --epochs 30 --backbone resnet18 --pretrained
"""

import argparse
import os

from src.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from src.data.loader import get_data_loaders
from src.engine.train import train
from src.models.model import build_model
from src.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Image Sentiment Classifier")
    parser.add_argument("--data_path", type=str, default=DATA_CONFIG["data_path"])
    parser.add_argument("--img_size", type=int, default=DATA_CONFIG["img_size"])
    parser.add_argument("--batch_size", type=int, default=DATA_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--backbone", type=str, default=MODEL_CONFIG["backbone"],
                        choices=["custom_cnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet-pretrained weights (transfer learning)")
    parser.add_argument("--save_dir", type=str, default=TRAIN_CONFIG["save_dir"])
    parser.add_argument("--log_dir", type=str, default=TRAIN_CONFIG["log_dir"])
    parser.add_argument("--augment", action="store_true", default=DATA_CONFIG["augment"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "data": {
            "data_path": args.data_path,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_workers": DATA_CONFIG["num_workers"],
            "train_split": DATA_CONFIG["train_split"],
            "val_split": DATA_CONFIG["val_split"],
            "test_split": DATA_CONFIG["test_split"],
            "random_seed": DATA_CONFIG["random_seed"],
            "augment": args.augment,
        },
        "train": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": TRAIN_CONFIG["weight_decay"],
            "save_dir": args.save_dir,
            "log_dir": args.log_dir,
            "early_stopping_patience": TRAIN_CONFIG["early_stopping_patience"],
        },
    }

    logger.info("Starting training with config: %s", config)
    print("Loading data …")
    train_loader, val_loader, _, class_names = get_data_loaders(config)
    print(f"Classes: {class_names}")

    num_classes = len(class_names)
    model = build_model(
        backbone=args.backbone,
        num_classes=num_classes,
        dropout=MODEL_CONFIG["dropout"],
        pretrained=args.pretrained,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.backbone} | Parameters: {total_params:,}")

    trained_model = train(model, train_loader, val_loader, config, class_names)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
