#  Image Sentiment Analysis

A deep learning project that classifies images based on their **sentiment (Happy / Sad)** using a Convolutional Neural Network (CNN) built with **PyTorch**.

This project is inspired by a deep CNN image classification pipeline and adapted for **visual sentiment understanding** – a task that goes beyond object detection to infer emotional context from images.

---

##  Overview

Image Sentiment Analysis is a computer vision task where the goal is to determine the **emotional tone conveyed by an image**.

Unlike traditional classification, this involves:

* Understanding **visual features**
* Extracting **contextual cues**
* Mapping them to **human emotions**

This project builds an end-to-end pipeline:

* Data preprocessing & augmentation
* Model training (Custom CNN or transfer-learning backbone)
* Evaluation (Precision / Recall / F1 / Accuracy)
* Prediction on unseen images (CLI + Streamlit web app)

---

## 🧠 Model Architecture

Three model options are available:

| Backbone | Description |
|----------|-------------|
| `custom_cnn` | 3 × Conv→BN→ReLU→MaxPool blocks, then Dense(256)→Dropout→Linear |
| `resnet18` | Pretrained ResNet-18 with custom classification head |
| `efficientnet_b0` | Pretrained EfficientNet-B0 with custom classification head |

---

## 📂 Project Structure

```
Image-Sentiment-Analysis/
├── app.py                   # Streamlit web app
├── train.py                 # CLI entry point – training
├── evaluate.py              # CLI entry point – evaluation
├── predict.py               # CLI entry point – single image inference
├── check.py                 # Sanity-check for the data loader
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py            # Centralised configuration
│   ├── utils.py             # Logging & custom exceptions
│   ├── data/
│   │   ├── loader.py        # PyTorch Dataset / DataLoader factory
│   │   └── transforms.py    # Train (augmented) & val/test transforms
│   ├── models/
│   │   └── model.py         # build_model() factory + SentimentCNN
│   ├── engine/
│   │   ├── train.py         # Training loop with early stopping & TensorBoard
│   │   └── evaluate.py      # Precision / Recall / F1 / Accuracy + confusion matrix
│   └── inference/
│       └── predict.py       # load_model() + predict() helpers
│
├── notebook/
│   └── image_sentiment_classification.ipynb
│
├── Data/          # (not tracked) class sub-directories e.g. Data/happy, Data/sad
└── models/        # (not tracked) saved .pth checkpoints
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

Organise images into class sub-directories under `Data/`:

```
Data/
├── happy/
│   ├── img1.jpg
│   └── ...
└── sad/
    ├── img1.jpg
    └── ...
```

The data loader automatically:

* Resizes images and normalises pixel values
* Applies random augmentation (flip, rotation, colour jitter) during training
* Splits data into 70 % train / 20 % val / 10 % test (configurable in `src/config.py`)
* Uses a fixed random seed for reproducible splits

---

## 🏋️ Training the Model

```bash
# Default custom CNN
python train.py

# Transfer learning with ResNet-18
python train.py --backbone resnet18 --pretrained --epochs 30

# All options
python train.py --help
```

Training includes:

* Loss: Cross-Entropy
* Optimizer: Adam with ReduceLROnPlateau scheduler
* Metrics: Loss & Accuracy logged to TensorBoard
* Early stopping (patience configurable)
* Best checkpoint auto-saved to `models/best_model.pth`

View TensorBoard logs:

```bash
tensorboard --logdir logs
```

---

## 📈 Evaluation

```bash
python evaluate.py --checkpoint models/best_model.pth --split test
```

Outputs Precision, Recall, F1, Accuracy and a confusion matrix.

---

## 🔍 Prediction (CLI)

```bash
python predict.py --image path/to/image.jpg
```

Output:

```
Predicted Sentiment : Happy
Confidence          : 0.9231

Class Probabilities :
          happy : 0.9231
            sad : 0.0769
```

---

## 🌐 Web App (Streamlit)

```bash
streamlit run app.py
```

Upload any image via the browser UI to get an instant sentiment prediction with confidence scores.

---

## ✅ Data Loader Check

```bash
python check.py
```

Verifies the data pipeline loads correctly and prints batch shapes.

---

## 🚧 Future Improvements

* Add **Grad-CAM visualisation** for model interpretability
* Extend to **multi-class sentiment** (Positive / Neutral / Negative)
* Combine with **text sentiment (multimodal AI)**
* Containerise with **Docker** for easy deployment

---

## 💡 Applications

* Social media analytics
* Marketing & brand perception
* Content recommendation
* Human-computer interaction
