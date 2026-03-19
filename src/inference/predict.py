import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from src.data.transforms import get_val_transforms
from src.utils import CustomException, logger


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[nn.Module, List[str]]:
    """Load model weights from a checkpoint file.

    Args:
        model: An initialised (but untrained) model instance.
        checkpoint_path: Path to a ``.pth`` checkpoint saved by the
            training loop.
        device: Target device.  Auto-detected when ``None``.

    Returns:
        Tuple of ``(model, class_names)`` where *model* has been moved to
        *device* and loaded with the checkpoint weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names: List[str] = checkpoint.get("class_names", [])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint from '%s' (epoch %d, val_loss=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", float("nan")),
    )
    return model, class_names


def predict(
    image_path: str,
    model: nn.Module,
    class_names: List[str],
    img_size: int = 224,
    device: torch.device = None,
) -> Dict:
    """Run inference on a single image and return the prediction.

    Args:
        image_path: Path to the input image file.
        model: A model already loaded with trained weights (e.g. via
            :func:`load_model`).
        class_names: Ordered list of class label strings.
        img_size: Size to which the image is resized before inference.
        device: Target device.  Auto-detected when ``None``.

    Returns:
        Dictionary with keys:

        * ``"class"``       – predicted class label (str)
        * ``"class_index"`` – predicted class index (int)
        * ``"confidence"``  – softmax probability of the predicted class
        * ``"probabilities"`` – dict mapping each class to its probability
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: '{image_path}'")

        transform = get_val_transforms(img_size)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)  # shape: [1, C, H, W]

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        probabilities = {
            cls: float(probs[i]) for i, cls in enumerate(class_names)
        }

        result = {
            "class": class_names[pred_idx] if class_names else str(pred_idx),
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
        }
        logger.info("Prediction for '%s': %s", image_path, result)
        return result

    except Exception as e:
        raise CustomException(str(e), sys) from e
