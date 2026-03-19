import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.ingestion import get_image_paths_and_labels, split_data
from src.utils import logger, CustomException


class ImageDataset(Dataset):
    """PyTorch Dataset for loading labelled images from file paths."""

    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        except Exception as e:
            raise CustomException(e, sys) from e


def _build_transforms(img_size: int):
    """Return (train_transform, val_transform) for a given image size."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_data_loaders(config: dict):
    """
    Build and return train / validation DataLoaders.

    Args:
        config: Dictionary with a ``'data'`` key containing:
            - data_path   (str)   : Root folder with class sub-directories.
            - img_size    (int)   : Image will be resized to (img_size × img_size).
            - batch_size  (int)   : Batch size for both loaders.
            - num_workers (int)   : DataLoader worker processes.
            - train_split (float) : Fraction of data for training (default 0.8).

    Returns:
        train_loader (DataLoader)
        val_loader   (DataLoader)
        class_names  (list[str])
    """
    try:
        data_cfg = config.get("data", config)

        data_path = data_cfg["data_path"]
        img_size = data_cfg.get("img_size", 224)
        batch_size = data_cfg.get("batch_size", 32)
        num_workers = data_cfg.get("num_workers", 2)
        train_split = data_cfg.get("train_split", 0.8)

        image_paths, labels, class_names = get_image_paths_and_labels(data_path)

        train_paths, train_labels, val_paths, val_labels = split_data(
            image_paths, labels, train_split=train_split
        )

        train_transform, val_transform = _build_transforms(img_size)

        train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "DataLoaders ready — train batches: %d, val batches: %d",
            len(train_loader),
            len(val_loader),
        )
        return train_loader, val_loader, class_names

    except Exception as e:
        raise CustomException(e, sys) from e
