DATA_CONFIG = {
    "data_path": "Data",       # Root folder: Data/happy, Data/sad
    "img_size": 224,           # Image resize dimension (height = width)
    "batch_size": 32,          # Batch size for DataLoader
    "num_workers": 2,          # Workers for DataLoader (set 0 on Windows)
    "train_split": 0.7,        # Fraction used for training
    "val_split": 0.2,          # Fraction used for validation
    "test_split": 0.1,         # Fraction used for testing
    "random_seed": 42,         # Seed for reproducibility
    "augment": True,           # Apply data augmentation during training
}

MODEL_CONFIG = {
    "num_classes": 2,          # Number of sentiment classes
    "dropout": 0.5,            # Dropout probability in classifier head
    "pretrained": False,       # Use ImageNet-pretrained backbone
    "backbone": "custom_cnn",  # Options: "custom_cnn" | "resnet18" | "efficientnet_b0"
}

TRAIN_CONFIG = {
    "epochs": 20,              # Number of training epochs
    "learning_rate": 1e-3,     # Initial learning rate for Adam optimiser
    "weight_decay": 1e-4,      # L2 regularisation weight decay
    "save_dir": "models",      # Directory to save trained model checkpoints
    "log_dir": "logs",         # Directory for TensorBoard logs
    "early_stopping_patience": 5,  # Stop if val-loss doesn't improve for N epochs
}
