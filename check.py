from src.data.loader import get_data_loaders

config = {
    'data': {
        'data_path': 'Data',
        'img_size': 224,
        'batch_size': 4,
        'num_workers': 0
    }
}

try:
    train_loader, val_loader, classes = get_data_loaders(config)
    print(f"Classes found: {classes}")
    
    # Take one batch from the training loader
    images, labels = next(iter(train_loader))
    
    print(f"Successfully loaded a batch of {len(images)} images.")
    print(f"Image tensor shape: {images.shape}") # Expect [4, 3, 224, 224]
    print(f"Labels in this batch: {labels}")      # Expect 0s and 1s
    print("✅ Loader is fully functional!")

except Exception as e:
    print(f"❌ Loader Test Failed: {e}")