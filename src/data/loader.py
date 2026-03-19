import sys

import torch
from PIL import Image   # physically opens the image file and loads it into memory as a PIL Image object in python
from torch.utils.data import DataLoader, Dataset # It handles the logic: if I ask for image - load it,prepare it and give me image and answer
from torchvision import transforms # before feeding to NN, it should be uniform, it chains together operation and preprocess the image

from src.data.ingestion import get_image_paths_and_labels, split_data
from src.utils import logger, CustomException
from src.data.transform import get_train_transforms, get_val_transforms
from src.config import DATA_CONFIG

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
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    return train_transform, val_transform


def get_data_loaders(config: dict):
    try:
        '''Your config.py file contains the baseline rules for your project.
        If someone runs your code and forgets to specify an image size or a batch size, the code won't crash.
        It will safely fall back to the default values you set in DATA_CONFIG.'''

        data_cfg = config.get("data", config)
        data_path = DATA_CONFIG["data_path"]  # default value from config.py, can be overridden by config argument
        img_size = data_cfg.get("img_size", DATA_CONFIG["img_size"])
        batch_size = data_cfg.get("batch_size", DATA_CONFIG["batch_size"])
        num_workers = data_cfg.get("num_workers", DATA_CONFIG["num_workers"])
        train_split = data_cfg.get("train_split", DATA_CONFIG["train_split"])

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
            # already used DATA_CONFIG['BATCH_SIZE'] in batch size, so no need to use DATA_CONFIG['BATCH_SIZE'] again here
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
