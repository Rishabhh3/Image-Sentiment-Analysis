"""
Data loading utilities for Image Sentiment Analysis.

Reads images from a directory structured as:
    Data/
        happy/   <- class 0
        sad/     <- class 1

Returns PyTorch DataLoader objects for training and validation.
"""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int = 224):
    """Return train and validation image transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


def get_data_loaders(config: dict):
    """
    Build and return train/validation DataLoaders.

    Args:
        config: dict with a 'data' sub-dict containing:
            - data_path   : root folder (e.g. 'Data')
            - img_size    : image resize target (default 224)
            - batch_size  : samples per batch (default 32)
            - num_workers : dataloader worker count (default 2)
            - train_split : fraction for training (default 0.8)

    Returns:
        train_loader (DataLoader), val_loader (DataLoader), class_names (list[str])
    """
    data_cfg = config.get("data", config)

    data_path = data_cfg.get("data_path", "Data")
    img_size = data_cfg.get("img_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    train_split = data_cfg.get("train_split", 0.8)

    train_tf, val_tf = get_transforms(img_size)

    # Load two separate ImageFolder instances so each subset can carry its
    # own transform (train augmentation vs. val-only normalization).
    train_full = datasets.ImageFolder(root=data_path, transform=train_tf)
    val_full = datasets.ImageFolder(root=data_path, transform=val_tf)

    class_names = train_full.classes

    n_total = len(train_full)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    # Derive a reproducible index split from the training dataset.
    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(
        train_full, [n_train, n_val], generator=generator
    )
    # Re-create the split with the same seed to obtain the val indices,
    # this time applied to the val_full dataset (different transforms).
    generator2 = torch.Generator().manual_seed(42)
    _, val_subset = random_split(
        val_full, [n_train, n_val], generator=generator2
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names
