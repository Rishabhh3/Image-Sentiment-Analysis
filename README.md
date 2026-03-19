#  Image Sentiment Analysis

A deep learning project that classifies images based on their **sentiment (Positive / Negative)** using a Convolutional Neural Network (CNN) built entirely with **PyTorch**.

---

##  Overview

Image Sentiment Analysis is a computer vision task where the goal is to determine the **emotional tone conveyed by an image**.

This project builds a clean, step-by-step end-to-end pipeline so that each component is easy to understand and modify independently:

| Step | File / Module | What it does |
|------|--------------|--------------|
| 1 | `src/data/loader.py` | Load & transform images into PyTorch `DataLoader` |
| 2 | `src/model/cnn.py` | CNN architecture definition |
| 3 | `src/engine/train.py` | Single-epoch training loop |
| 4 | `src/engine/evaluate.py` | Loss, accuracy, precision & recall |
| 5 | `src/inference/predict.py` | Single-image prediction |
| 6 | `train.py` | **Run training** (entry point) |
| 7 | `evaluate.py` | **Run evaluation** (entry point) |
| 8 | `predict.py` | **Run prediction** (entry point) |

---

## 🧠 Model Architecture

```
Input (3 × 224 × 224)
  └─ Conv2d(3→16, 3×3) + ReLU + MaxPool(2×2)
  └─ Conv2d(16→32, 3×3) + ReLU + MaxPool(2×2)
  └─ Conv2d(32→16, 3×3) + ReLU + MaxPool(2×2)
  └─ Flatten
  └─ Linear(→256) + ReLU + Dropout(0.5)
  └─ Linear(256→num_classes)
```

## 📂 Project Structure

```
Image-Sentiment-Analysis/
├── Data/                        # Dataset (images organized by class folder)
│   ├── happy/
│   └── sad/
├── models/                      # Saved model checkpoints
├── logs/                        # Training logs
├── notebook/
│   └── image_sentiment_classification.ipynb  # Original TensorFlow reference
├── src/
│   ├── config.py                # Dataset / training configuration
│   ├── utils.py                 # Logger and custom exceptions
│   ├── data/
│   │   └── loader.py            # PyTorch Dataset + DataLoader helpers
│   ├── model/
│   │   └── cnn.py               # SentimentCNN architecture
│   ├── engine/
│   │   ├── train.py             # train_one_epoch()
│   │   └── evaluate.py          # evaluate()
│   └── inference/
│       └── predict.py           # predict() for a single image
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── predict.py                   # Prediction entry point
├── check.py                     # Quick data-loader smoke test
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Rishabhh3/Image-Sentiment-Analysis.git
cd Image-Sentiment-Analysis
pip install -r requirements.txt
```

---

## 📊 Dataset

Place your images in a `Data/` folder, one sub-folder per class:

```
Data/
├── happy/   ← class 0
└── sad/     ← class 1
```

Dataset preprocessing includes:

* Resizing to 224 × 224
* Normalization (ImageNet mean/std)
* Random horizontal flip (training only)
* 80/20 train-validation split

---

## 🏋️ Training the Model

```bash
python train.py
```

Training configuration lives in `src/config.py`. The best model is saved to `models/best_model.pth`.

Training includes:

* Loss: Cross-Entropy
* Optimizer: Adam
* Metrics: Loss, Accuracy, Precision, Recall

---

## 📊 Evaluating the Model

```bash
python evaluate.py
# or with a custom checkpoint:
python evaluate.py --checkpoint models/best_model.pth
```

---

## 🔍 Prediction

```bash
python predict.py --image path/to/image.jpg
```

Output:

```
Predicted Sentiment: happy
Confidence        : 0.9231
Class Probabilities:
  happy     : 0.9231
  sad       : 0.0769
```

---

## 🚧 Future Improvements

* Use **Transfer Learning (ResNet / EfficientNet)**
* Add **Grad-CAM visualization** for interpretability
* Deploy as a **web app (Streamlit / Flask)**
* Combine with **text sentiment (multimodal AI)**

---

## 💡 Applications

* Social media analytics
* Marketing & brand perception
* Content recommendation
* Human-computer interaction

