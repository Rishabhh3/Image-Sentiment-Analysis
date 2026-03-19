"""
evaluate.py — Entry point for evaluating a trained model checkpoint.

Usage:
    python evaluate.py [--checkpoint models/best_model.pth]

Loads the model from the given checkpoint and prints precision, recall, and
accuracy on the validation set.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

from src.config import DATA_CONFIG
from src.data.loader import get_data_loaders
from src.engine.evaluate import evaluate as run_evaluate
from src.inference.predict import load_model
from src.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Image Sentiment CNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("models", "best_model.pth"),
        help="Path to the model checkpoint (default: models/best_model.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"❌  Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load model from checkpoint
    # ------------------------------------------------------------------
    model, class_names, img_size = load_model(args.checkpoint, device)
    print(f"Loaded model from: {args.checkpoint}")
    print(f"Classes: {class_names}")

    # ------------------------------------------------------------------
    # Build validation loader (use same config as training)
    # ------------------------------------------------------------------
    data_cfg = DATA_CONFIG.copy()
    data_cfg["img_size"] = img_size
    config = {"data": data_cfg}

    _, val_loader, _ = get_data_loaders(config)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    metrics = run_evaluate(model, val_loader, criterion, device)

    print("\n── Evaluation Results ──────────────────────")
    print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print("────────────────────────────────────────────")

    logger.info(f"Evaluation results: {metrics}")


if __name__ == "__main__":
    main()
