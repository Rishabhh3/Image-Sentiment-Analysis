import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Import custom modules
from src.config import DATA_CONFIG
from src.utils import logger, CustomException
from src.data.ingestion import get_image_paths_and_labels, split_data
from src.data.loader import get_data_loaders
from src.models.model import SentimentModel, get_device, freeze_backbone, save_model
from src.engine.train import Trainer
from src.engine.evaluate import Evaluator
from src.inference.predict import SentimentPredictor


def setup_directories():
    """Create required directories for saving models and logs."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    logger.info("Directories setup complete")


def main():
    """
    Complete end-to-end pipeline for Image Sentiment Analysis:
    1. Data Ingestion: Load and validate images
    2. Data Splitting: Split into train/validation sets
    3. Data Loading: Create PyTorch DataLoaders with augmentation
    4. Model Initialization: Create ResNet50-based sentiment model
    5. Training: Train the model with validation
    6. Evaluation: Evaluate on validation set with metrics
    7. Visualization: Generate confusion matrix
    """
    
    parser = argparse.ArgumentParser(description="Image Sentiment Analysis Pipeline")
    parser.add_argument("--mode", default="train", choices=["train", "evaluate", "predict"],
                        help="Pipeline mode: train, evaluate, or predict")
    parser.add_argument("--model_path", default="models/sentiment_model.pth",
                        help="Path to save/load model")
    parser.add_argument("--image_path", default=None,
                        help="Path to image for prediction (used in predict mode)")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze ResNet backbone and train only final layer")
    
    args = parser.parse_args()
    
    # ============ SETUP ============
    setup_directories()
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # ============ STEP 1: DATA INGESTION ============
    logger.info("=" * 50)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 50)
    
    try:
        image_paths, labels, class_names = get_image_paths_and_labels(DATA_CONFIG["data_path"])
        logger.info(f"Total images found: {len(image_paths)}")
        logger.info(f"Classes: {class_names}")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return
    
    # ============ STEP 2: DATA SPLITTING ============
    logger.info("=" * 50)
    logger.info("STEP 2: DATA SPLITTING")
    logger.info("=" * 50)
    
    try:
        train_paths, train_labels, val_paths, val_labels = split_data(
            image_paths, labels, train_split=DATA_CONFIG["train_split"]
        )
        logger.info(f"Train samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        return
    
    # ============ STEP 3: DATA LOADING ============
    logger.info("=" * 50)
    logger.info("STEP 3: DATA LOADING & AUGMENTATION")
    logger.info("=" * 50)
    
    try:
        config = {"data": DATA_CONFIG}
        train_loader, val_loader, _ = get_data_loaders(config)
        logger.info(f"Train loader batches: {len(train_loader)}")
        logger.info(f"Validation loader batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return
    
    # ============ STEP 4: MODEL INITIALIZATION ============
    logger.info("=" * 50)
    logger.info("STEP 4: MODEL INITIALIZATION")
    logger.info("=" * 50)
    
    try:
        model = SentimentModel(num_classes=len(class_names), pretrained=True)
        if args.freeze_backbone:
            logger.info("Freezing ResNet50 backbone")
            model = freeze_backbone(model)
        model.to(device)
        logger.info("Model initialized: ResNet50 with 2 output classes")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return
    
    # ============ TRAINING MODE ============
    if args.mode == "train":
        logger.info("=" * 50)
        logger.info("STEP 5: TRAINING")
        logger.info("=" * 50)
        
        try:
            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                use_mixed_precision=False
            )
            
            train_loss_history = []
            train_acc_history = []
            val_loss_history = []
            val_acc_history = []
            
            for epoch in range(args.num_epochs):
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch [{epoch+1}/{args.num_epochs}]")
                logger.info(f"{'='*50}")
                
                # Training
                train_metrics = trainer.train_epoch(train_loader)
                logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Accuracy: {train_metrics['accuracy']:.4f}")
                train_loss_history.append(train_metrics['loss'])
                train_acc_history.append(train_metrics['accuracy'])
                
                # Validation
                val_metrics = trainer.validate(val_loader)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
                val_loss_history.append(val_metrics['loss'])
                val_acc_history.append(val_metrics['accuracy'])
                
                # Update learning rate
                trainer.scheduler.step(val_metrics['loss'])
            
            # ============ STEP 6: EVALUATION ============
            logger.info("=" * 50)
            logger.info("STEP 6: EVALUATION ON VALIDATION SET")
            logger.info("=" * 50)
            
            evaluator = Evaluator(device=device)
            val_metrics = evaluator.evaluate(model, val_loader, trainer.criterion)
            
            logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1']:.4f}")
            
            # ============ STEP 7: VISUALIZATION ============
            logger.info("=" * 50)
            logger.info("STEP 7: VISUALIZATION & METRICS")
            logger.info("=" * 50)
            
            # Plot training curves
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot(train_loss_history, label='Train Loss')
            axes[0].plot(val_loss_history, label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Over Epochs')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(train_acc_history, label='Train Accuracy')
            axes[1].plot(val_acc_history, label='Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Over Epochs')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig("logs/training_curves.png")
            logger.info("Training curves saved to logs/training_curves.png")
            
            # Plot confusion matrix
            confusion_matrix_plot = evaluator.plot_confusion_matrix(class_names)
            confusion_matrix_plot.savefig("logs/confusion_matrix.png")
            logger.info("Confusion matrix saved to logs/confusion_matrix.png")
            
            # ============ SAVE MODEL ============
            logger.info("=" * 50)
            logger.info("SAVING MODEL")
            logger.info("=" * 50)
            
            save_model(model, args.model_path)
            logger.info(f"Model saved to {args.model_path}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            return
    
    # ============ EVALUATION MODE (WITHOUT TRAINING) ============
    elif args.mode == "evaluate":
        logger.info("=" * 50)
        logger.info("EVALUATION MODE")
        logger.info("=" * 50)
        
        try:
            # Load trained model
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
            model.eval()
            logger.info(f"Model loaded from {args.model_path}")
            
            # Evaluate on validation set
            evaluator = Evaluator(device=device)
            criterion = torch.nn.CrossEntropyLoss()
            val_metrics = evaluator.evaluate(model, val_loader, criterion)
            
            logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1']:.4f}")
            
            # Plot confusion matrix
            confusion_matrix_plot = evaluator.plot_confusion_matrix(class_names)
            confusion_matrix_plot.savefig("logs/confusion_matrix_eval.png")
            logger.info("Confusion matrix saved to logs/confusion_matrix_eval.png")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return
    
    # ============ PREDICTION MODE ============
    elif args.mode == "predict":
        logger.info("=" * 50)
        logger.info("PREDICTION MODE")
        logger.info("=" * 50)
        
        if not args.image_path:
            logger.error("Image path required for prediction mode. Use --image_path <path>")
            return
        
        try:
            # Load trained model
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
            logger.info(f"Model loaded from {args.model_path}")
            
            # Make prediction
            predictor = SentimentPredictor(model, device=device)
            result = predictor.predict(args.image_path)
            
            logger.info(f"Image: {args.image_path}")
            logger.info(f"Predicted Sentiment: {result['sentiment'].upper()}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Probabilities: {result['probabilities']}")
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return
    
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
