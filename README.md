#  Image Sentiment Analysis

A deep learning project that classifies images based on their **sentiment (e.g., Positive / Negative / Neutral)** using a Convolutional Neural Network (CNN).

This project is inspired by a deep CNN image classification pipeline and adapted for **visual sentiment understanding**, a task that goes beyond object detection to infer emotional context from images.

---

##  Overview

Image Sentiment Analysis is a computer vision task where the goal is to determine the **emotional tone conveyed by an image**.

Unlike traditional classification, this involves:

* Understanding **visual features**
* Extracting **contextual cues**
* Mapping them to **human emotions**

This project builds an end-to-end pipeline:

* Data preprocessing
* Model training (CNN)
* Evaluation
* Prediction on unseen images

---

## 🧠 Model Architecture

The model is based on a **Convolutional Neural Network (CNN)**:

* Convolution layers → Feature extraction
* ReLU activation → Non-linearity
* Pooling layers → Dimensionality reduction
* Fully connected layers → Classification
* Output layer → Sentiment class probabilities


## 📂 Project Structure

```
Image-Sentiment-Analysis/
│── data/                  # Dataset (images categorized by sentiment)
│── models/                # Saved trained models
│── src/
|── image_sentiment_classfication.ipynb
│── requirements.txt
│── README.md
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

* Images are labeled into sentiment categories:

  * Positive 😊
  * Negative 😞

Dataset preprocessing includes:

* Resizing images
* Normalization
* Train-test split
* Data augmentation (optional)

---

## 🏋️ Training the Model

```bash
python src/train.py
```

Training includes:

* Loss: Cross-Entropy
* Optimizer: Adam
* Metrics: Accuracy

---

## 🔍 Prediction

```bash
python src/predict.py --image path_to_image
```

Output:

```
Predicted Sentiment: Positive
Confidence: 0.87
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
