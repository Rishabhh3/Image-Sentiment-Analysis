"""
Single-image inference for Image Sentiment Analysis.

Loads a trained model checkpoint and predicts the sentiment label
for a single image file, printing the predicted class and confidence.
"""

import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load a saved model checkpoint.

    The checkpoint is expected to be a dict saved by ``train.py`` with keys:
        - 'model_state_dict'
        - 'class_names'
        - 'img_size'

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device:          Target device (CPU / CUDA).

    Returns:
        model (nn.Module), class_names (list[str]), img_size (int)
    """
    # Import here to avoid a circular dependency at module level
    from src.model.cnn import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(num_classes=len(class_names), img_size=img_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size


def predict(
    image_path: str,
    checkpoint_path: str,
    device: torch.device | None = None,
) -> dict:
    """
    Predict the sentiment label for a single image.

    Args:
        image_path:      Path to the image file (jpg, png, …).
        checkpoint_path: Path to the trained model checkpoint.
        device:          Device to run inference on (auto-detected if None).

    Returns:
        A dict with keys:
            - 'label'      : predicted class name (str)
            - 'confidence' : probability of the predicted class (float)
            - 'class_probs': dict mapping class_name → probability
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, img_size = load_model(checkpoint_path, device)

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    confidence, pred_idx = probs.max(dim=0)

    class_probs = {
        name: round(probs[i].item(), 4)
        for i, name in enumerate(class_names)
    }

    return {
        "label": class_names[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "class_probs": class_probs,
    }
