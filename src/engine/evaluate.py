import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt


class Evaluator:
    """Evaluate model performance on sentiment analysis tasks."""
    
    def __init__(self, device=None): 
        # Automatically use Mac GPU (MPS), NVIDIA GPU (CUDA), or fallback to CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.predictions = []
        self.ground_truth = []
    
    def evaluate(self, model, dataloader, criterion):
        """Evaluate model on a dataset."""
        model.eval()    # set the model to evaluation mode, which turns off dropout and batch normalization layers,
                        # ensuring that the model's behavior is consistent during evaluation
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader: 
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy()) # move the predicted labels back to CPU and convert to numpy array, then extend the list of all predictions with these new predictions
                all_labels.extend(labels.cpu().numpy()) 
        
        self.predictions = np.array(all_preds)  # convert the list of all predictions to a numpy array for easier manipulation and evaluation
        self.ground_truth = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(self.ground_truth, self.predictions), 
            'precision': precision_score(self.ground_truth, self.predictions, average='weighted', zero_division=0),

            # recall_score with average='weighted' calculates the recall for each class and then computes a weighted average of these recalls,
            # where the weight for each class is proportional to the number of true instances of that class in the dataset. 
            # The zero_division=0 parameter ensures that if there are any cases where precision or recall cannot be calculated (e.g., when there are no predicted samples for a class), it will return 0 instead of raising an error.
            'recall': recall_score(self.ground_truth, self.predictions, average='weighted', zero_division=0),
            'f1': f1_score(self.ground_truth, self.predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, class_names=None): 
        """Plot confusion matrix."""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        return plt