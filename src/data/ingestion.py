import os
import sys
import random

from src.utils import logger, CustomException

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_FILE_SIZE = 10 * 1024  # 10 KB


def _is_valid_image(file_path: str) -> bool:
    """Return True if the file has a valid extension and meets the minimum size."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False
    if os.path.getsize(file_path) < MIN_FILE_SIZE:
        return False
    return True


def get_image_paths_and_labels(data_path: str):
    """
    Scan *data_path* for class sub-directories and collect valid image paths.

    Returns:
        image_paths (list[str]): Absolute paths to valid images.
        labels      (list[int]): Integer label for each image (class index).
        class_names (list[str]): Sorted list of class names.
    """
    try:
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        class_names = sorted(
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        )
        # sorted so that the label is always in same order
        # class names = happy and sad

        if not class_names:
            raise ValueError(f"No class sub-directories found in: {data_path}")

        image_paths, labels = [], []

        ''' This loop iterates through each class sub-directory, assigns a label index based on the sorted order of class names,
          and collects valid image file paths along with their corresponding labels. It also logs warnings for any invalid 
          or small images that are skipped. '''
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_path, class_name)
            for fname in os.listdir(class_dir):
                file_path = os.path.join(class_dir, fname)
                if os.path.isfile(file_path) and _is_valid_image(file_path):
                    image_paths.append(file_path)
                    labels.append(label_idx)
                    # if valid image, add to list of paths and integer ID
                else:
                    logger.warning("Skipping invalid or small image: %s", file_path)

        logger.info(
            "Found %d valid images across %d classes: %s",
            len(image_paths),
            len(class_names),
            class_names,
        )
        return image_paths, labels, class_names

    except Exception as e:
        raise CustomException(e, sys) from e


def split_data(image_paths: list, labels: list, train_split: float = 0.8, seed: int = 42):
    """
    Randomly split *image_paths* and *labels* into train and validation sets.

    Args:
        image_paths  : List of image file paths.
        labels       : Corresponding integer labels.
        train_split  : Fraction of data to use for training (default 0.8).
        seed         : Random seed for reproducibility.

    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    try:
        if not 0 < train_split < 1:
            raise ValueError(f"train_split must be in (0, 1), got {train_split}")

        combined = list(zip(image_paths, labels)) # locks them together so they are shuffled in the same way and convert into list of tuples
        # otherswise if we shuffle them separately, the paths and labels will no longer correspond to each other
        random.seed(seed)
        random.shuffle(combined) # without this, the data will always be in the same order (e.g. all happy followed by all sad) 
                                # so the model might just learn to predict the majority class and not learn anything useful.

        split_idx = int(len(combined) * train_split)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]

        train_paths, train_labels = (list(x) for x in zip(*train_data)) if train_data else ([], [])
        # the * does opposite of zip, it unzips the list of tuples back into separate lists for paths and labels

        val_paths, val_labels = (list(x) for x in zip(*val_data)) if val_data else ([], [])
        ''' We are unzipping them for model training because the DataLoader expects separate lists of paths and labels. like X, y
        after splitting the model wants separate list of paths and labels'''


        logger.info(
            "Data split — train: %d samples, val: %d samples",
            len(train_paths),
            len(val_paths),
        )
        return list(train_paths), list(train_labels), list(val_paths), list(val_labels)

    except Exception as e:
        raise CustomException(e, sys) from e
