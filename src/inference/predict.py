import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

class SentimentPredictor:
    def __init__(self, model, device=None):
        # Auto-select best device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'  # Metal Performance Shaders for Mac
            else:
                device = 'cpu'
        
        self.model = model
        self.device = device
        self.model.to(self.device) # move the model to the selected device (GPU or CPU) for inference
        self.model.eval() # set the model to evaluation mode, which turns off dropout and batch normalization layers
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.sentiment_labels = ['sad', 'happy']
    
    def predict(self, image_path):
        """Predict sentiment for a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device) 
            
            with torch.no_grad():
                output = self.model(image_tensor) # get the raw output from the model, which is typically a tensor of shape (1, num_classes) containing the logits for each class
                probabilities = torch.softmax(output, dim=1) # apply the softmax function to convert the logits into probabilities for each class

                predicted_class = torch.argmax(probabilities, dim=1).item() # get the index of the class with the highest probability, which corresponds to the predicted sentiment label
                confidence = probabilities[0][predicted_class].item() # get the confidence score for the predicted class, which is the probability of the predicted class according to the model
            
            return {
                'sentiment': self.sentiment_labels[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.sentiment_labels, probabilities[0])
                    # create a dictionary mapping each sentiment label to its corresponding probability, which allows us to see confidence for each possible sentiment
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, image_paths):
        """Predict sentiment for multiple images"""
        return [self.predict(image_path) for image_path in image_paths]