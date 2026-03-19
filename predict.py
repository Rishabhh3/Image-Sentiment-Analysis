"""CLI entry point for running inference with a trained model.

Usage::

    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth
"""

import argparse

import torch

from src.config import DATA_CONFIG, MODEL_CONFIG
from src.inference.predict import load_model, predict
from src.models.model import build_model
from src.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sentiment of a single image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pth",
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=MODEL_CONFIG["backbone"],
        choices=["custom_cnn", "resnet18", "efficientnet_b0"],
    )
    parser.add_argument(
        "--img_size", type=int, default=DATA_CONFIG["img_size"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        backbone=args.backbone,
        num_classes=2,
        dropout=MODEL_CONFIG["dropout"],
    )
    model, class_names = load_model(model, args.checkpoint, device)

    if not class_names:
        class_names = ["happy", "sad"]
        print(
            "Warning: no class names found in checkpoint, "
            f"defaulting to {class_names}"
        )

    result = predict(args.image, model, class_names, args.img_size, device)

    print(f"\nPredicted Sentiment : {result['class'].capitalize()}")
    print(f"Confidence          : {result['confidence']:.4f}")
    print("\nClass Probabilities :")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:>15} : {prob:.4f}")


if __name__ == "__main__":
    main()
