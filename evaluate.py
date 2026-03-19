"""CLI entry point for evaluating a trained Image Sentiment Analysis model.

Usage::

    python evaluate.py --checkpoint models/best_model.pth
    python evaluate.py --checkpoint models/best_model.pth --split test
"""

import argparse

import torch

from src.config import DATA_CONFIG, MODEL_CONFIG
from src.data.loader import get_data_loaders
from src.engine.evaluate import evaluate
from src.inference.predict import load_model
from src.models.model import build_model
from src.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Image Sentiment Classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pth",
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_CONFIG["data_path"]
    )
    parser.add_argument("--img_size", type=int, default=DATA_CONFIG["img_size"])
    parser.add_argument("--batch_size", type=int, default=DATA_CONFIG["batch_size"])
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--backbone", type=str, default=MODEL_CONFIG["backbone"],
        choices=["custom_cnn", "resnet18", "efficientnet_b0"]
    )
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
            "augment": False,
        },
    }

    train_loader, val_loader, test_loader, class_names = get_data_loaders(config)
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_map[args.split]

    model = build_model(
        backbone=args.backbone,
        num_classes=len(class_names),
        dropout=MODEL_CONFIG["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names_from_ckpt = load_model(model, args.checkpoint, device)

    # Prefer class names stored in the checkpoint
    if class_names_from_ckpt:
        class_names = class_names_from_ckpt

    print(f"\nEvaluating '{args.split}' split with classes: {class_names}")
    metrics = evaluate(model, loader, class_names, device)
    logger.info("Final metrics on '%s' split: %s", args.split, metrics)


if __name__ == "__main__":
    main()
