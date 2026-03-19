"""
predict.py — Entry point for single-image sentiment prediction.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth

Output example:
    Predicted Sentiment: happy
    Confidence        : 0.9231
    Class Probabilities:
      happy : 0.9231
      sad   : 0.0769
"""

import argparse
import os
import sys

from src.inference.predict import predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the sentiment of a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file.",
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

    if not os.path.isfile(args.image):
        print(f"❌  Image not found: {args.image}")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"❌  Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    result = predict(args.image, args.checkpoint)

    print(f"\nPredicted Sentiment: {result['label']}")
    print(f"Confidence        : {result['confidence']}")
    print("Class Probabilities:")
    for cls, prob in result["class_probs"].items():
        print(f"  {cls:<10}: {prob}")


if __name__ == "__main__":
    main()
