import torchvision.transforms as T


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """Return augmentation + normalisation transforms for the training set."""
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(img_size: int = 224) -> T.Compose:
    """Return deterministic transforms for the validation / test sets."""
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
