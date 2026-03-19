import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils import logger


def get_data_loaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train / val / test DataLoaders from a directory of class sub-folders.

    Expected directory layout::

        data_path/
            happy/
                img1.jpg
                img2.png
                ...
            sad/
                img1.jpg
                ...

    Args:
        config: Dictionary that must contain a ``data`` sub-dict with at
            minimum the keys ``data_path``, ``img_size``, ``batch_size``
            and ``num_workers``.  Optional keys are ``train_split``,
            ``val_split``, ``test_split``, ``random_seed`` and ``augment``.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, class_names)``.
    """
    data_cfg = config.get("data", config)

    data_path: str = data_cfg["data_path"]
    img_size: int = data_cfg.get("img_size", 224)
    batch_size: int = data_cfg.get("batch_size", 32)
    num_workers: int = data_cfg.get("num_workers", 2)
    train_split: float = data_cfg.get("train_split", 0.7)
    val_split: float = data_cfg.get("val_split", 0.2)
    seed: int = data_cfg.get("random_seed", 42)
    augment: bool = data_cfg.get("augment", True)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: '{data_path}'. "
            "Please create it and add class sub-folders (e.g. Data/happy, Data/sad)."
        )

    # Load full dataset with val transforms first (used for val / test splits)
    full_dataset = ImageFolder(data_path, transform=get_val_transforms(img_size))
    class_names: List[str] = full_dataset.classes
    n_total = len(full_dataset)

    if n_total == 0:
        raise ValueError(f"No images found in '{data_path}'.")

    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply augmentation transforms to the training subset only
    if augment:
        train_dataset = _SubsetWithTransform(
            full_dataset, train_subset.indices, get_train_transforms(img_size)
        )
    else:
        train_dataset = train_subset

    logger.info(
        "Dataset split — train: %d | val: %d | test: %d | classes: %s",
        len(train_dataset),
        len(val_subset),
        len(test_subset),
        class_names,
    )

    train_loader = DataLoader(
        train_dataset,
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
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a subset of an ImageFolder, applying a custom transform."""

    def __init__(self, dataset: ImageFolder, indices: List[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img_path, label = self.dataset.imgs[self.indices[idx]]
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
